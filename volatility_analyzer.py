"""
Volatility-adjusted confidence module for forecasting.

This module analyzes the sentiment and volume of recent news related to a forecast
to identify high-volatility information environments. In such cases, it automatically
shrinks predictions towards the midpoint to reflect increased uncertainty.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import MetaculusQuestion, clean_indents


@dataclass
class VolatilityAnalysisResult:
    """
    Contains the results of volatility analysis for a forecasting question.
    
    This structure captures information volatility metrics and recommended 
    confidence adjustments.
    """
    question: MetaculusQuestion
    analyzed_keywords: List[str]
    news_volume: int  # Number of relevant news articles found
    sentiment_volatility: float  # 0.0 (stable) to 1.0 (highly volatile)
    conflicting_reports_score: float  # 0.0 (no conflicts) to 1.0 (high conflicts)
    overall_volatility_score: float  # 0.0 (stable) to 1.0 (highly volatile)
    volatility_level: str  # "Low", "Medium", "High"
    confidence_adjustment_factor: float  # Multiplier for shrinking to midpoint (0.0 to 1.0)
    midpoint_shrinkage_amount: float  # How much to shrink towards 50%
    detailed_analysis: str  # LLM's detailed analysis
    news_sample: List[Dict[str, Any]]  # Sample of analyzed news articles


class VolatilityAnalyzer:
    """
    Analyzes information volatility in the news environment to adjust forecast confidence.
    
    This analyzer examines recent news related to a forecasting question to identify
    periods of high volatility, conflicting information, and rapid changes that should
    reduce confidence in specific predictions.
    """
    
    def __init__(
        self,
        get_llm: Callable[[str, str], Any],
        asknews_client: AsyncAskNewsSDK,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the volatility analyzer."""
        self._get_llm = get_llm
        self._asknews_client = asknews_client
        self._logger = logger or logging.getLogger(__name__)
        
        # Configuration parameters
        self.max_articles_to_analyze = 50
        self.analysis_window_days = 30  # Look at news from past 30 days
        self.high_volume_threshold = 20  # Articles per week for high volume
        self.high_volatility_threshold = 0.7  # Threshold for high volatility classification
        
    async def analyze_information_volatility(
        self,
        question: MetaculusQuestion,
        keywords: Optional[List[str]] = None
    ) -> VolatilityAnalysisResult:
        """
        Analyze the information environment volatility for a forecasting question.
        
        Args:
            question: The forecasting question to analyze
            keywords: Optional list of keywords to focus the news search
            
        Returns:
            VolatilityAnalysisResult with detailed volatility metrics and adjustments
        """
        self._logger.info(f"Starting volatility analysis for question: {question.page_url}")
        
        # Step 1: Extract relevant keywords if not provided
        if keywords is None:
            keywords = await self._extract_keywords_from_question(question)
        
        # Step 2: Gather recent news related to the question
        news_articles = await self._gather_recent_news(keywords)
        
        # Step 3: Analyze sentiment volatility and conflicts
        sentiment_volatility, conflicting_score, detailed_analysis = await self._analyze_news_volatility(
            question, news_articles, keywords
        )
        
        # Step 4: Calculate volume metrics
        news_volume = len(news_articles)
        volume_score = min(news_volume / (self.high_volume_threshold * (self.analysis_window_days / 7)), 1.0)
        
        # Step 5: Compute overall volatility score
        overall_volatility = self._compute_overall_volatility_score(
            volume_score, sentiment_volatility, conflicting_score
        )
        
        # Step 6: Determine confidence adjustment
        volatility_level, adjustment_factor, shrinkage_amount = self._calculate_confidence_adjustment(
            overall_volatility
        )
        
        self._logger.info(f"Volatility analysis completed. Level: {volatility_level}, "
                         f"Volume: {news_volume}, Overall score: {overall_volatility:.3f}")
        
        return VolatilityAnalysisResult(
            question=question,
            analyzed_keywords=keywords,
            news_volume=news_volume,
            sentiment_volatility=sentiment_volatility,
            conflicting_reports_score=conflicting_score,
            overall_volatility_score=overall_volatility,
            volatility_level=volatility_level,
            confidence_adjustment_factor=adjustment_factor,
            midpoint_shrinkage_amount=shrinkage_amount,
            detailed_analysis=detailed_analysis,
            news_sample=news_articles[:10]  # Keep sample for inspection
        )
    
    async def _extract_keywords_from_question(self, question: MetaculusQuestion) -> List[str]:
        """Extract relevant keywords from the forecasting question for news search."""
        prompt = clean_indents(f"""
            Extract 5-8 key search terms from this forecasting question that would be most useful 
            for finding relevant news coverage. Focus on the main entities, events, and topics.
            
            Question: {question.question_text}
            
            Background: {question.background_info or "No background provided"}
            
            Return only a comma-separated list of keywords, no explanations.
            Example: "artificial intelligence, OpenAI, GPT-5, machine learning, AI safety"
        """)
        
        response = await self._get_llm("keyword_extractor_llm", "llm").invoke(prompt)
        keywords = [k.strip() for k in response.split(",") if k.strip()]
        return keywords[:8]  # Limit to 8 keywords
    
    async def _gather_recent_news(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Gather recent news articles related to the keywords."""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.analysis_window_days)
        
        all_articles = []
        
        # Search for news using multiple keyword combinations
        search_queries = [
            " ".join(keywords[:3]),  # Top 3 keywords
            " OR ".join(keywords[:5]) if len(keywords) > 1 else keywords[0],  # Broader search
        ]
        
        for query in search_queries:
            try:
                self._logger.info(f"Searching news for: {query}")
                response = await self._asknews_client.news.search_news(
                    query=query,
                    n_articles=25,
                    return_type="both",
                    method="kw",
                    start_timestamp=int(start_date.timestamp()),
                    end_timestamp=int(end_date.timestamp()),
                )
                
                if hasattr(response, 'as_dicts'):
                    articles = response.as_dicts
                elif hasattr(response, 'articles'):
                    articles = response.articles
                else:
                    articles = response if isinstance(response, list) else []
                
                all_articles.extend(articles)
                
            except Exception as e:
                self._logger.warning(f"Error searching news for '{query}': {e}")
                continue
        
        # Deduplicate and limit
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get('title', '')
            if title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
                if len(unique_articles) >= self.max_articles_to_analyze:
                    break
        
        self._logger.info(f"Gathered {len(unique_articles)} unique articles for volatility analysis")
        return unique_articles
    
    async def _analyze_news_volatility(
        self, 
        question: MetaculusQuestion, 
        articles: List[Dict[str, Any]], 
        keywords: List[str]
    ) -> Tuple[float, float, str]:
        """
        Analyze news articles for sentiment volatility and conflicting information.
        
        Returns:
            Tuple of (sentiment_volatility, conflicting_score, detailed_analysis)
        """
        if not articles:
            return 0.0, 0.0, "No recent news articles found for analysis."
        
        # Prepare articles text for analysis (limit length)
        articles_text = []
        for article in articles[:20]:  # Analyze up to 20 articles
            title = article.get('title', '')
            summary = article.get('summary', article.get('content', ''))[:500]  # Limit length
            date = article.get('pub_date', '')
            articles_text.append(f"[{date}] {title}\n{summary}")
        
        combined_text = "\n\n---\n\n".join(articles_text)
        
        prompt = self._build_volatility_analysis_prompt(question, combined_text, keywords)
        
        analysis_result = await self._get_llm("volatility_analyzer_llm", "llm").invoke(prompt)
        
        # Parse the analysis result
        sentiment_volatility, conflicting_score = self._parse_volatility_metrics(analysis_result)
        
        return sentiment_volatility, conflicting_score, analysis_result
    
    def _build_volatility_analysis_prompt(
        self,
        question: MetaculusQuestion,
        news_text: str,
        keywords: List[str]
    ) -> str:
        """Build prompt for analyzing news volatility."""
        return clean_indents(f"""
            You are a Volatility Analysis Specialist tasked with assessing the stability 
            of the information environment around a forecasting question.
            
            ## Forecasting Question
            {question.question_text}
            
            ## Keywords Being Tracked
            {', '.join(keywords)}
            
            ## Recent News Coverage
            {news_text}
            
            ## Your Task
            Analyze the news coverage above to determine information environment volatility.
            Focus on:
            
            ### 1. Sentiment Volatility Assessment
            - Are there rapid changes in tone/sentiment about the topic?
            - Do articles contradict each other in their outlook?
            - Is there evidence of shifting narratives or changing expert opinions?
            
            ### 2. Conflicting Information Analysis  
            - Are there direct contradictions between different news sources?
            - Do articles present conflicting facts, data, or expert opinions?
            - Are there competing narratives about likely outcomes?
            
            ### 3. Volume and Intensity Patterns
            - Is there unusually high coverage volume that might indicate instability?
            - Are there signs of rapid news cycles or "breaking news" patterns?
            
            **Required Output Format:**
            **Sentiment Volatility Score:** [0.0 to 1.0, where 1.0 is extremely volatile]
            **Conflicting Information Score:** [0.0 to 1.0, where 1.0 is highly conflicting]
            
            **Detailed Analysis:**
            [Your comprehensive analysis explaining the volatility assessment, specific examples 
            of conflicts or rapid changes, and how this affects forecast confidence]
            
            **Key Volatility Indicators:**
            - [List specific examples from the news that indicate volatility]
            
            **Confidence Impact:**
            [Explanation of how this volatility should affect forecasting confidence]
        """)
    
    def _parse_volatility_metrics(self, analysis_text: str) -> Tuple[float, float]:
        """Parse volatility metrics from LLM analysis."""
        sentiment_volatility = 0.0
        conflicting_score = 0.0
        
        # Look for sentiment volatility score
        sentiment_match = re.search(r'Sentiment Volatility Score.*?(\d+\.?\d*)', analysis_text, re.IGNORECASE)
        if sentiment_match:
            try:
                sentiment_volatility = float(sentiment_match.group(1))
                sentiment_volatility = max(0.0, min(1.0, sentiment_volatility))  # Clamp to [0,1]
            except ValueError:
                pass
        
        # Look for conflicting information score
        conflict_match = re.search(r'Conflicting Information Score.*?(\d+\.?\d*)', analysis_text, re.IGNORECASE)
        if conflict_match:
            try:
                conflicting_score = float(conflict_match.group(1))
                conflicting_score = max(0.0, min(1.0, conflicting_score))  # Clamp to [0,1]
            except ValueError:
                pass
        
        # Fallback: analyze text for volatility indicators
        if sentiment_volatility == 0.0 and conflicting_score == 0.0:
            text_lower = analysis_text.lower()
            
            # Check for high volatility indicators
            high_volatility_terms = ['highly volatile', 'extremely unstable', 'rapid changes', 'conflicting']
            medium_volatility_terms = ['some volatility', 'moderate changes', 'mixed signals']
            
            if any(term in text_lower for term in high_volatility_terms):
                sentiment_volatility = 0.8
                conflicting_score = 0.7
            elif any(term in text_lower for term in medium_volatility_terms):
                sentiment_volatility = 0.5
                conflicting_score = 0.4
            else:
                sentiment_volatility = 0.2
                conflicting_score = 0.2
        
        return sentiment_volatility, conflicting_score
    
    def _compute_overall_volatility_score(
        self, 
        volume_score: float, 
        sentiment_volatility: float, 
        conflicting_score: float
    ) -> float:
        """Compute overall volatility score from component scores."""
        # Weighted combination of different volatility factors
        # Conflicting information is most important, sentiment volatility second, volume third
        overall_score = (
            0.5 * conflicting_score +     # 50% weight on conflicts
            0.3 * sentiment_volatility +  # 30% weight on sentiment volatility
            0.2 * volume_score           # 20% weight on volume
        )
        
        return min(1.0, overall_score)  # Cap at 1.0
    
    def _calculate_confidence_adjustment(
        self, 
        overall_volatility: float
    ) -> Tuple[str, float, float]:
        """
        Calculate confidence adjustment parameters based on volatility score.
        
        Returns:
            Tuple of (volatility_level, adjustment_factor, shrinkage_amount)
        """
        if overall_volatility >= 0.7:
            # High volatility: significant shrinkage towards midpoint
            level = "High"
            adjustment_factor = 0.3  # Use only 30% of original confidence
            shrinkage_amount = 0.7   # Shrink 70% towards 50%
        elif overall_volatility >= 0.4:
            # Medium volatility: moderate shrinkage
            level = "Medium" 
            adjustment_factor = 0.6  # Use 60% of original confidence
            shrinkage_amount = 0.4   # Shrink 40% towards 50%
        else:
            # Low volatility: minimal adjustment
            level = "Low"
            adjustment_factor = 0.9  # Use 90% of original confidence
            shrinkage_amount = 0.1   # Shrink 10% towards 50%
        
        return level, adjustment_factor, shrinkage_amount
    
    @staticmethod
    def apply_volatility_adjustment(
        original_prediction: float, 
        volatility_result: VolatilityAnalysisResult
    ) -> float:
        """
        Apply volatility adjustment to shrink prediction towards midpoint.
        
        Args:
            original_prediction: Original prediction probability (0.0 to 1.0)
            volatility_result: Results from volatility analysis
            
        Returns:
            Adjusted prediction probability
        """
        if volatility_result.volatility_level == "Low":
            # Minimal adjustment for low volatility
            return original_prediction
        
        # Calculate midpoint (50% for binary questions)
        midpoint = 0.5
        shrinkage = volatility_result.midpoint_shrinkage_amount
        
        # Apply shrinkage towards midpoint
        adjusted_prediction = original_prediction * (1 - shrinkage) + midpoint * shrinkage
        
        return adjusted_prediction
    
    @staticmethod 
    def format_volatility_explanation(volatility_result: VolatilityAnalysisResult) -> str:
        """Format volatility analysis for inclusion in forecast explanation."""
        if volatility_result.volatility_level == "Low":
            return ""  # No explanation needed for low volatility
        
        return clean_indents(f"""
            
            ## Information Volatility Assessment
            
            **Volatility Level:** {volatility_result.volatility_level}
            **News Volume:** {volatility_result.news_volume} articles analyzed
            **Overall Volatility Score:** {volatility_result.overall_volatility_score:.2f}/1.0
            
            **Key Factors:**
            - Sentiment Volatility: {volatility_result.sentiment_volatility:.2f}/1.0
            - Conflicting Reports: {volatility_result.conflicting_reports_score:.2f}/1.0
            
            **Confidence Adjustment Applied:**
            Due to the {volatility_result.volatility_level.lower()} volatility in the information 
            environment, this forecast has been adjusted towards the midpoint (50%) to account for 
            increased uncertainty. The prediction has been shrunk by 
            {volatility_result.midpoint_shrinkage_amount:.0%} towards 50% to reflect the unstable 
            information landscape.
            
            **Volatility Analysis Summary:**
            {volatility_result.detailed_analysis}
        """)
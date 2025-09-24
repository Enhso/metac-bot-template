"""
Volatility-aware ensemble forecaster that incorporates information environment volatility
analysis to dynamically adjust prediction confidence.
"""

import asyncio
import logging
import time
from typing import List, Optional

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import MetaculusQuestion, ForecastReport

from data_models import EnhancedResearchDossier, VolatilityAnalysisResult
from forecasters.contradiction_aware_ensemble import ContradictionAwareEnsembleForecaster
from volatility_analyzer import VolatilityAnalyzer

logger = logging.getLogger(__name__)


class VolatilityAwareEnsembleForecaster(ContradictionAwareEnsembleForecaster):
    """
    Enhanced ensemble forecaster that incorporates volatility-adjusted confidence.
    
    This forecaster extends the contradiction-aware ensemble by adding analysis of
    information environment volatility. When high volatility is detected (e.g.,
    conflicting news reports, rapid sentiment changes), predictions are automatically
    adjusted towards the midpoint to reflect increased uncertainty.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._volatility_analyzer = None
        logger.info("Initialized VolatilityAwareEnsembleForecaster with information volatility analysis")
    
    def _get_volatility_analyzer(self) -> VolatilityAnalyzer:
        """Get or create volatility analyzer instance."""
        if self._volatility_analyzer is None:
            asknews_client = self._create_asknews_client()
            self._volatility_analyzer = VolatilityAnalyzer(
                get_llm=lambda name, kind: self.get_llm(name, "llm"),
                asknews_client=asknews_client,
                logger=logger
            )
        return self._volatility_analyzer
    
    async def _generate_enhanced_research_dossier(self, question: MetaculusQuestion) -> EnhancedResearchDossier:
        """
        Generate comprehensive research dossier with bias, contradiction, AND volatility analysis.
        """
        # Start with the contradiction-aware dossier
        dossier = await super()._generate_enhanced_research_dossier(question)
        
        # Add volatility analysis
        logger.info(f"Starting volatility analysis for URL {question.page_url}")
        volatility_start_time = time.time()
        
        try:
            volatility_analyzer = self._get_volatility_analyzer()
            volatility_analysis = await volatility_analyzer.analyze_information_volatility(question)
            dossier.volatility_analysis = volatility_analysis
            
            volatility_time = time.time() - volatility_start_time
            logger.info(f"Volatility analysis completed in {volatility_time:.2f}s for URL {question.page_url}. "
                       f"Level: {volatility_analysis.volatility_level}, "
                       f"Score: {volatility_analysis.overall_volatility_score:.3f}")
        
        except Exception as e:
            logger.error(f"Error during volatility analysis for URL {question.page_url}: {e}")
            # Create a default low-volatility result as fallback
            dossier.volatility_analysis = VolatilityAnalysisResult(
                question=question,
                analyzed_keywords=[],
                news_volume=0,
                sentiment_volatility=0.0,
                conflicting_reports_score=0.0,
                overall_volatility_score=0.0,
                volatility_level="Low",
                confidence_adjustment_factor=1.0,
                midpoint_shrinkage_amount=0.0,
                detailed_analysis="Volatility analysis failed; assuming low volatility."
            )
        
        return dossier
    
    def _build_volatility_aware_prompt(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        persona_prompt: str
    ) -> str:
        """
        Build a prompt that incorporates awareness of biases, contradictions, AND volatility.
        """
        base_prompt = self._build_contradiction_aware_prompt(enhanced_dossier, persona_prompt)
        
        # Add volatility section if present
        volatility_section = ""
        if enhanced_dossier.volatility_analysis:
            volatility_info = enhanced_dossier.volatility_analysis
            volatility_section = f"""
            
            ### 5. Information Environment Volatility Analysis
            **Volatility Level:** {volatility_info.volatility_level}
            **Overall Volatility Score:** {volatility_info.overall_volatility_score:.2f}/1.0
            **News Volume Analyzed:** {volatility_info.news_volume} articles
            **Sentiment Volatility:** {volatility_info.sentiment_volatility:.2f}/1.0
            **Conflicting Reports Score:** {volatility_info.conflicting_reports_score:.2f}/1.0
            **Keywords Analyzed:** {', '.join(volatility_info.analyzed_keywords)}
            
            **Volatility Analysis:**
            {volatility_info.detailed_analysis}
            
            **Recommended Confidence Adjustment:**
            Due to {volatility_info.volatility_level.lower()} information volatility, consider adjusting 
            prediction confidence. The analysis suggests shrinking predictions by 
            {volatility_info.midpoint_shrinkage_amount:.0%} towards the midpoint (50%) to account for 
            the unstable information environment."""
        
        # Insert volatility section before the synthesis task
        if "## Your Enhanced Synthesis Task" in base_prompt:
            parts = base_prompt.split("## Your Enhanced Synthesis Task", 1)
            enhanced_prompt = (parts[0] + volatility_section + "\n\n## Your Enhanced Synthesis Task" + 
                             parts[1] if len(parts) == 2 else base_prompt + volatility_section)
        else:
            enhanced_prompt = base_prompt + volatility_section
        
        # Update the synthesis instructions to include volatility awareness
        enhanced_prompt = enhanced_prompt.replace(
            "Integrate all analysis components above into a final prediction that accounts for both cognitive biases and contradictory information.",
            "Integrate all analysis components above into a final prediction that accounts for cognitive biases, contradictory information, AND information environment volatility."
        )
        
        # Add volatility awareness to the steps
        if "### Step 1: Bias and Contradiction-Aware Evidence Integration" in enhanced_prompt:
            enhanced_prompt = enhanced_prompt.replace(
                "### Step 1: Bias and Contradiction-Aware Evidence Integration\n        - Synthesize the research while addressing both identified biases and contradictions",
                "### Step 1: Bias, Contradiction, and Volatility-Aware Evidence Integration\n        - Synthesize the research while addressing identified biases, contradictions, and volatility indicators"
            )
        
        return enhanced_prompt
    
    async def _generate_persona_analysis_with_volatility(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        persona_name: str,
        persona_prompt: str
    ) -> str:
        """Generate persona analysis incorporating volatility awareness."""
        
        # Use volatility-aware prompt
        prompt = self._build_volatility_aware_prompt(enhanced_dossier, persona_prompt)
        
        # Get persona-specific LLM
        persona_llm_key = self._get_persona_llm(persona_name)
        
        # Generate refined prediction with volatility awareness
        refined_prediction_text = await self.get_llm(persona_llm_key, "llm").invoke(prompt)
        
        return refined_prediction_text
    
    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """
        Orchestrates the volatility-aware ensemble forecasting process.
        """
        async with self._concurrency_limiter:
            overall_start_time = time.time()
            
            # --- ENHANCED RESEARCH PHASE (with volatility analysis) ---
            logger.info(f"Starting enhanced research phase for URL {question.page_url}")
            research_start_time = time.time()
            enhanced_dossier = await self._generate_enhanced_research_dossier(question)
            research_time = time.time() - research_start_time
            
            # --- VOLATILITY-AWARE PERSONA ANALYSIS PHASE ---
            logger.info(f"Starting volatility-aware persona analysis for URL {question.page_url}")
            persona_start_time = time.time()
            
            persona_reports = []
            for persona_name in self.PERSONAS:
                persona_prompt = self.PERSONAS[persona_name]
                logger.info(f"Generating {persona_name} volatility-aware analysis for URL {question.page_url}")
                
                persona_step_start = time.time()
                refined_prediction_text = await self._generate_persona_analysis_with_volatility(
                    enhanced_dossier, persona_name, persona_prompt
                )
                persona_step_time = time.time() - persona_step_start
                logger.info(f"{persona_name} volatility-aware analysis completed in {persona_step_time:.2f}s")
                
                persona_reports.append((persona_name, refined_prediction_text))
            
            persona_total_time = time.time() - persona_start_time
            logger.info(f"All volatility-aware persona analyses completed in {persona_total_time:.2f}s")
            
            # --- SYNTHESIS WITH VOLATILITY ADJUSTMENT ---
            logger.info(f"Starting synthesis with volatility adjustment for URL {question.page_url}")
            synthesis_start_time = time.time()
            
            # First, synthesize ensemble forecasts normally
            reasoned_prediction = await self._synthesize_ensemble_forecasts(question, persona_reports)
            
            # Then, apply volatility adjustment if needed
            if enhanced_dossier.volatility_analysis:
                original_prediction = reasoned_prediction.prediction_value
                
                # Apply volatility adjustment based on the type of prediction
                adjusted_prediction = self._apply_volatility_adjustment_to_prediction(
                    original_prediction, enhanced_dossier.volatility_analysis
                )
                
                # Update the prediction if adjustment was applied
                if adjusted_prediction != original_prediction:
                    reasoned_prediction.prediction_value = adjusted_prediction
                    logger.info(f"Applied volatility adjustment: {original_prediction} → {adjusted_prediction}")
            
            synthesis_time = time.time() - synthesis_start_time
            logger.info(f"Synthesis with volatility adjustment completed in {synthesis_time:.2f}s")
            
            # --- FORMAT FINAL REPORT ---
            final_explanation = self._format_volatility_aware_explanation(
                reasoned_prediction, enhanced_dossier
            )
            
            # Construct final report
            final_report = self._construct_final_report(question, reasoned_prediction.prediction_value, final_explanation)
            
            # Publish if required
            if self.publish_reports_to_metaculus:
                await final_report.publish_report_to_metaculus()
            
            # Log efficiency summary
            total_time = time.time() - overall_start_time
            logger.info(f"VOLATILITY-AWARE FORECAST COMPLETED for URL {question.page_url}: "
                       f"Total={total_time:.2f}s (Research={research_time:.2f}s, "
                       f"Personas={persona_total_time:.2f}s, Synthesis={synthesis_time:.2f}s)")
            
            return final_report
    
    def _apply_volatility_adjustment_to_prediction(self, original_prediction, volatility_analysis: VolatilityAnalysisResult):
        """Apply volatility adjustment to prediction based on prediction type."""
        from volatility_analyzer import VolatilityAnalyzer
        
        if volatility_analysis.volatility_level == "Low":
            return original_prediction  # No adjustment needed
        
        # For binary and multiple choice questions, treat as probability
        if isinstance(original_prediction, (float, int)):
            return VolatilityAnalyzer.apply_volatility_adjustment(float(original_prediction), volatility_analysis)
        
        # For numeric distributions, adjust the central tendency towards prior expectations
        # This is a simplified approach - could be more sophisticated
        if hasattr(original_prediction, 'median') or hasattr(original_prediction, 'mean'):
            # For now, return as-is for numeric predictions
            # Could implement more sophisticated adjustment later
            return original_prediction
        
        # For multiple choice predictions, adjust probabilities towards uniform distribution
        if hasattr(original_prediction, 'probabilities') or isinstance(original_prediction, (list, dict)):
            # This would need more sophisticated handling based on the prediction format
            return original_prediction
        
        return original_prediction
    
    def _format_volatility_aware_explanation(
        self, 
        reasoned_prediction, 
        enhanced_dossier: EnhancedResearchDossier
    ) -> str:
        """Format explanation including volatility analysis."""
        
        # Start with base explanation
        base_explanation = f"# Volatility-Aware Ensemble Forecast\n\n{reasoned_prediction.reasoning}"
        
        # Add volatility section if analysis was performed
        volatility_section = ""
        if enhanced_dossier.volatility_analysis:
            from volatility_analyzer import VolatilityAnalyzer
            volatility_section = VolatilityAnalyzer.format_volatility_explanation(
                enhanced_dossier.volatility_analysis
            )
        
        # Add bias summary if available
        bias_summary = ""
        if enhanced_dossier.bias_analysis and enhanced_dossier.bias_analysis.detected_biases:
            bias_summary = (f"\n\n## Cognitive Bias Mitigation\n"
                           f"Detected biases: {', '.join(enhanced_dossier.bias_analysis.detected_biases)}. "
                           f"Corrections applied to improve accuracy.")
        
        # Add contradiction summary if available
        contradiction_summary = ""
        if enhanced_dossier.contradiction_analysis:
            contradiction_info = enhanced_dossier.contradiction_analysis
            contradiction_summary = (f"\n\n## Contradictory Information Analysis\n"
                                    f"Detected {len(contradiction_info.detected_contradictions)} contradictions. "
                                    f"Overall coherence: {contradiction_info.overall_coherence_assessment}. "
                                    f"Key uncertainties identified and addressed.")
        
        return base_explanation + volatility_section + bias_summary + contradiction_summary
    
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | object | None]:
        """
        Returns default LLM configurations including volatility analyzer LLM.
        """
        defaults = super()._llm_config_defaults()
        
        # Add volatility analyzer LLM (use same as contradiction analyzer)
        defaults["volatility_analyzer_llm"] = defaults["contradiction_analyzer_llm"]
        
        return defaults

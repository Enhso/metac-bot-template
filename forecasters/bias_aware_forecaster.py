"""
Enhanced Self-Critique Forecaster with Cognitive Bias Self-Correction

This module extends the existing self-critique forecasting framework to include
systematic cognitive bias detection and correction. It implements a parallel
LLM call using a "Cognitive Bias Red Team" persona to improve forecast accuracy.
"""

import asyncio
import logging
import time
from typing import Optional

from forecasting_tools import (
    ForecastReport,
    MetaculusQuestion,
    ReasonedPrediction,
)

from critique_strategy import CritiqueAndRefineStrategy
from data_models import ResearchDossier, BiasAnalysisResult, EnhancedResearchDossier
from forecasters.self_critique import SelfCritiqueForecaster


logger = logging.getLogger(__name__)


class BiasAwareSelfCritiqueForecaster(SelfCritiqueForecaster):
    """
    Enhanced forecaster that incorporates systematic cognitive bias detection
    and correction into the self-critique process.
    
    This forecaster extends the standard workflow with a parallel cognitive bias
    analysis step that reviews the initial rationale for systematic errors and
    provides specific correction recommendations.
    
    The workflow becomes:
    1. Initial research (unchanged)
    2. Initial prediction (unchanged) 
    3. Adversarial critique (unchanged)
    4. Cognitive bias analysis (NEW - parallel step)
    5. Targeted research (unchanged)
    6. Bias-aware refined prediction (ENHANCED - incorporates bias corrections)
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the bias-aware forecaster with cognitive bias checking."""
        super().__init__(*args, **kwargs)
        logger.info("Initialized BiasAwareSelfCritiqueForecaster with cognitive bias self-correction")
    
    async def _generate_research_dossier(self, question: MetaculusQuestion) -> EnhancedResearchDossier:
        """
        Generate an enhanced research dossier that includes cognitive bias analysis.
        
        This extends the standard research dossier with systematic bias detection
        to create a more comprehensive analysis foundation.
        """
        # Generate the standard research dossier first
        standard_dossier = await super()._generate_research_dossier(question)
        
        # Perform cognitive bias analysis on the initial prediction
        asknews_client = self._create_asknews_client()
        strategy = CritiqueAndRefineStrategy(
            lambda name, kind: self.get_llm(name, "llm"), 
            asknews_client=asknews_client,
            logger=logger
        )
        
        bias_start_time = time.time()
        logger.info(f"Starting cognitive bias analysis for URL {question.page_url}")
        
        # Analyze the initial prediction for cognitive biases
        bias_analysis_text = await strategy.perform_cognitive_bias_analysis(
            question=question,
            rationale_text=standard_dossier.initial_prediction_text,
            reasoning_context=f"Initial Research Context:\n{standard_dossier.initial_research}"
        )
        
        # Parse the bias analysis to extract structured information
        bias_result = self._parse_bias_analysis(
            question=question,
            analyzed_rationale=standard_dossier.initial_prediction_text,
            bias_analysis_text=bias_analysis_text
        )
        
        bias_time = time.time() - bias_start_time
        logger.info(f"Cognitive bias analysis completed in {bias_time:.2f}s for URL {question.page_url}")
        
        # Create enhanced dossier with bias analysis
        enhanced_dossier = EnhancedResearchDossier(
            question=standard_dossier.question,
            initial_research=standard_dossier.initial_research,
            initial_prediction_text=standard_dossier.initial_prediction_text,
            critique_text=standard_dossier.critique_text,
            targeted_research=standard_dossier.targeted_research,
            bias_analysis=bias_result
        )
        
        return enhanced_dossier
    
    def _parse_bias_analysis(
        self,
        question: MetaculusQuestion,
        analyzed_rationale: str,
        bias_analysis_text: str
    ) -> BiasAnalysisResult:
        """
        Parse the bias analysis text to extract structured information.
        
        This method extracts key information from the bias analysis to create
        a structured result that can be used for further processing.
        """
        # Simple parsing - in production, this could be more sophisticated
        detected_biases = []
        priority_corrections = []
        severity_assessment = "Medium"  # Default
        confidence_adjustment_recommended = False
        
        # Extract detected biases (look for bias names in the text)
        bias_keywords = [
            "anchoring", "availability", "confirmation", "overconfidence",
            "base rate neglect", "representativeness", "recency", "survivorship",
            "planning fallacy", "attribution"
        ]
        
        text_lower = bias_analysis_text.lower()
        for bias in bias_keywords:
            if bias in text_lower:
                detected_biases.append(bias.title() + " Bias")
        
        # Check for severity indicators
        if "high" in text_lower and "risk" in text_lower:
            severity_assessment = "High"
        elif "low" in text_lower and "risk" in text_lower:
            severity_assessment = "Low"
        
        # Check for confidence adjustment recommendations
        if any(phrase in text_lower for phrase in ["adjust confidence", "recalibrate", "less certain", "more uncertain"]):
            confidence_adjustment_recommended = True
        
        # Extract priority corrections (simplified - look for correction section)
        if "critical biases" in text_lower or "priority" in text_lower:
            priority_corrections = detected_biases[:2]  # Take top 2 as priority
        
        return BiasAnalysisResult(
            question=question,
            analyzed_rationale=analyzed_rationale,
            bias_analysis_text=bias_analysis_text,
            detected_biases=detected_biases,
            severity_assessment=severity_assessment,
            priority_corrections=priority_corrections,
            confidence_adjustment_recommended=confidence_adjustment_recommended
        )
    
    async def _generate_final_prediction(
        self,
        dossier: EnhancedResearchDossier,
        persona_prompt: Optional[str] = None
    ) -> ReasonedPrediction:
        """
        Generate the final prediction using bias-aware refinement.
        
        This method uses the enhanced refinement process that explicitly
        incorporates cognitive bias corrections.
        """
        asknews_client = self._create_asknews_client()
        strategy = CritiqueAndRefineStrategy(
            lambda name, kind: self.get_llm(name, "llm"), 
            asknews_client=asknews_client,
            logger=logger
        )
        
        # Use bias-aware refinement if bias analysis is available
        if dossier.bias_analysis is not None:
            logger.info(f"Generating bias-aware refined prediction for URL {dossier.question.page_url}")
            refined_prediction_text = await strategy.generate_bias_aware_refined_prediction(
                question=dossier.question,
                initial_research=dossier.initial_research,
                initial_prediction_text=dossier.initial_prediction_text,
                critique_text=dossier.critique_text,
                targeted_research=dossier.targeted_research,
                bias_analysis=dossier.bias_analysis.bias_analysis_text,
                persona_prompt=persona_prompt,
            )
        else:
            # Fallback to standard refinement if bias analysis is not available
            logger.warning(f"No bias analysis available, using standard refinement for URL {dossier.question.page_url}")
            refined_prediction_text = await strategy.generate_refined_prediction(
                question=dossier.question,
                initial_research=dossier.initial_research,
                initial_prediction_text=dossier.initial_prediction_text,
                critique_text=dossier.critique_text,
                targeted_research=dossier.targeted_research,
                persona_prompt=persona_prompt,
            )
        
        # Extract prediction from the refined text
        prediction = self._extract_prediction_using_centralized_logic(dossier.question, refined_prediction_text)
        
        return ReasonedPrediction(prediction_value=prediction, reasoning=refined_prediction_text)
    
    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """
        Run the complete bias-aware forecasting process for a single question.
        
        This orchestrates the full workflow including cognitive bias analysis.
        """
        async with self._concurrency_limiter:
            overall_start_time = time.time()
            
            logger.info(f"Starting bias-aware forecasting for URL {question.page_url}")
            
            # Generate enhanced research dossier with bias analysis
            dossier = await self._generate_research_dossier(question)
            
            # Generate bias-aware final prediction
            reasoned_prediction = await self._generate_final_prediction(dossier)
            
            # Log bias analysis summary if available
            if dossier.bias_analysis:
                bias_info = dossier.bias_analysis
                logger.info(f"Bias analysis summary for URL {question.page_url}: "
                           f"Detected {len(bias_info.detected_biases)} biases "
                           f"(Severity: {bias_info.severity_assessment}, "
                           f"Confidence adjustment: {bias_info.confidence_adjustment_recommended})")
            
            # Format the final explanation
            bias_summary = ""
            if dossier.bias_analysis and dossier.bias_analysis.detected_biases:
                bias_summary = f"\n\n## Cognitive Bias Mitigation\nThis forecast has been analyzed for cognitive biases. " \
                             f"Detected potential biases: {', '.join(dossier.bias_analysis.detected_biases)}. " \
                             f"Specific corrections have been applied to improve accuracy."
            
            final_explanation = f"# Bias-Aware Forecast\n\n{reasoned_prediction.reasoning}{bias_summary}"
            
            # Construct the final report
            final_report = self._construct_final_report(question, reasoned_prediction.prediction_value, final_explanation)
            
            # Publish if required
            if self.publish_reports_to_metaculus:
                await final_report.publish_report_to_metaculus()
            
            total_time = time.time() - overall_start_time
            logger.info(f"Completed bias-aware forecasting in {total_time:.2f}s for URL {question.page_url}")
            
            return final_report
    
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str]:
        """
        Returns default LLM configuration including the bias checker LLM.
        """
        defaults = super()._llm_config_defaults()
        
        # Add bias checker LLM (can use the same as critique_llm or a specialized model)
        defaults["bias_checker_llm"] = defaults.get("critique_llm", defaults["default"])
        
        return defaults
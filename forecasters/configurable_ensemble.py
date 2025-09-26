"""
Configurable Ensemble Forecaster

This module implements a unified ensemble forecaster that uses composition instead of
inheritance to combine different analysis capabilities. It replaces the brittle
inheritance chain: EnsembleForecaster -> BiasAware -> ContradictionAware -> VolatilityAware

The ConfigurableEnsembleForecaster can be configured with any combination of:
- Cognitive bias analysis
- Contradictory information detection and resolution  
- Information environment volatility assessment

This provides maximum flexibility while eliminating code duplication and the
"all-or-nothing" limitation of the inheritance approach.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import (
    BinaryQuestion,
    ForecastReport,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    ReasonedPrediction,
)

from cognitive_bias_checker import CognitiveBiasChecker
from contradictory_information_analyzer import ContradictoryInformationAnalyzer
from volatility_analyzer import VolatilityAnalyzer
from critique_strategy import CritiqueAndRefineStrategy
from data_models import EnhancedResearchDossier, BiasAnalysisResult, VolatilityAnalysisResult
from forecasters.ensemble import EnsembleForecaster
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class ForecasterConfig:
    """
    Configuration for the ConfigurableEnsembleForecaster.
    
    This replaces the rigid inheritance hierarchy with flexible composition.
    Each analyzer can be independently enabled/disabled.
    """
    enable_bias_analysis: bool = False
    enable_contradiction_analysis: bool = False
    enable_volatility_analysis: bool = False
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled features for logging."""
        features = []
        if self.enable_bias_analysis:
            features.append("cognitive_bias_analysis")
        if self.enable_contradiction_analysis:
            features.append("contradiction_detection")
        if self.enable_volatility_analysis:
            features.append("volatility_assessment")
        return features or ["standard_ensemble"]
    
    def __str__(self) -> str:
        """String representation for logging."""
        features = self.get_enabled_features()
        return f"ForecasterConfig({', '.join(features)})"


class ConfigurableEnsembleForecaster(EnsembleForecaster):
    """
    Unified ensemble forecaster with configurable analysis capabilities.
    
    This forecaster uses composition instead of inheritance to combine different
    analysis capabilities. It can be configured to include any combination of:
    
    - Cognitive bias analysis and correction
    - Contradictory information detection and resolution
    - Information environment volatility assessment
    
    The workflow is:
    1. Generate base research dossier (from parent class)
    2. Apply configured analysis steps to enhance the dossier
    3. Generate persona analyses using the enhanced dossier
    4. Synthesize final forecast with full analytical awareness
    
    This eliminates the rigid inheritance chain and provides maximum flexibility.
    """
    
    def __init__(
        self, 
        *args: Any, 
        forecaster_config: Optional[ForecasterConfig] = None,
        **kwargs: Any
    ):
        """
        Initialize the configurable forecaster.
        
        Args:
            forecaster_config: Configuration specifying which analyzers to use
            *args, **kwargs: Standard EnsembleForecaster arguments
        """
        super().__init__(*args, **kwargs)
        
        self._config = forecaster_config or ForecasterConfig()
        
        # Initialize analyzers lazily
        self._bias_checker: Optional[CognitiveBiasChecker] = None
        self._contradiction_analyzer: Optional[ContradictoryInformationAnalyzer] = None
        self._volatility_analyzer: Optional[VolatilityAnalyzer] = None
        
        logger.info(f"Initialized {self.__class__.__name__} with {self._config}")
    
    def _llm_wrapper(self, name: str, kind: str) -> Any:
        """Wrapper function for get_llm to match expected signature."""
        return self.get_llm(name, "llm")
    
    def _get_bias_checker(self) -> CognitiveBiasChecker:
        """Lazy initialization of cognitive bias checker."""
        if self._bias_checker is None:
            self._bias_checker = CognitiveBiasChecker(
                get_llm=self._llm_wrapper,
                logger=logger
            )
        return self._bias_checker
    
    def _get_contradiction_analyzer(self) -> ContradictoryInformationAnalyzer:
        """Lazy initialization of contradiction analyzer."""
        if self._contradiction_analyzer is None:
            self._contradiction_analyzer = ContradictoryInformationAnalyzer(
                get_llm=self._llm_wrapper,
                logger=logger
            )
        return self._contradiction_analyzer
    
    def _get_volatility_analyzer(self) -> VolatilityAnalyzer:
        """Lazy initialization of volatility analyzer."""
        if self._volatility_analyzer is None:
            asknews_client = self._create_asknews_client()
            if asknews_client is None:
                raise ValueError("AskNews client is required for volatility analysis")
            self._volatility_analyzer = VolatilityAnalyzer(
                get_llm=self._llm_wrapper,
                asknews_client=asknews_client,
                logger=logger
            )
        return self._volatility_analyzer
    
    async def _generate_research_dossier(self, question: MetaculusQuestion) -> EnhancedResearchDossier:
        """
        Generate an enhanced research dossier by applying configured analysis steps.
        
        This method:
        1. Generates the base research dossier using the parent class
        2. Applies enabled analysis steps to enhance the dossier
        3. Returns the fully enhanced dossier
        """
        logger.info(f"Starting configurable dossier generation for URL {question.page_url}")
        start_time = time.time()
        
        # Start with base research dossier from parent class
        base_dossier = await super()._generate_research_dossier(question)
        
        # Convert to enhanced dossier
        enhanced_dossier = EnhancedResearchDossier(
            question=base_dossier.question,
            initial_research=base_dossier.initial_research,
            initial_prediction_text=base_dossier.initial_prediction_text,
            critique_text=base_dossier.critique_text,
            targeted_research=base_dossier.targeted_research,
            bias_analysis=None,
            contradiction_analysis=None,
            volatility_analysis=None
        )
        
        # Apply configured analysis steps
        if self._config.enable_bias_analysis:
            enhanced_dossier = await self._add_bias_analysis(enhanced_dossier, question)
        
        if self._config.enable_contradiction_analysis:
            enhanced_dossier = await self._add_contradiction_analysis(enhanced_dossier, question)
        
        if self._config.enable_volatility_analysis:
            enhanced_dossier = await self._add_volatility_analysis(enhanced_dossier, question)
        
        total_time = time.time() - start_time
        enabled_features = self._config.get_enabled_features()
        logger.info(f"Configurable dossier generation completed in {total_time:.2f}s for URL {question.page_url}. "
                   f"Applied: {', '.join(enabled_features)}")
        
        return enhanced_dossier
    
    async def _add_bias_analysis(
        self, 
        dossier: EnhancedResearchDossier, 
        question: MetaculusQuestion
    ) -> EnhancedResearchDossier:
        """Add cognitive bias analysis to the dossier."""
        logger.info(f"Starting cognitive bias analysis for URL {question.page_url}")
        bias_start_time = time.time()
        
        # Perform bias analysis on the initial prediction
        bias_checker = self._get_bias_checker()
        bias_analysis_text = await bias_checker.analyze_for_cognitive_biases(
            question=question,
            rationale_text=dossier.initial_prediction_text,
            reasoning_context=f"Initial Research Context:\n{dossier.initial_research}"
        )
        
        # Parse the bias analysis into structured form
        bias_result = CognitiveBiasChecker.parse_analysis_text(
            question=question,
            analyzed_rationale=dossier.initial_prediction_text,
            bias_analysis_text=bias_analysis_text
        )
        
        # Add to dossier
        dossier.bias_analysis = bias_result
        
        bias_time = time.time() - bias_start_time
        logger.info(f"Cognitive bias analysis completed in {bias_time:.2f}s for URL {question.page_url}. "
                   f"Detected: {len(bias_result.detected_biases)} biases, "
                   f"Severity: {bias_result.severity_assessment}")
        
        return dossier
    
    async def _add_contradiction_analysis(
        self, 
        dossier: EnhancedResearchDossier, 
        question: MetaculusQuestion
    ) -> EnhancedResearchDossier:
        """Add contradictory information analysis to the dossier."""
        logger.info(f"Starting contradictory information analysis for URL {question.page_url}")
        contradiction_start_time = time.time()
        
        # Prepare research materials for contradiction analysis
        research_materials = {
            'initial_research': dossier.initial_research,
            'initial_prediction': dossier.initial_prediction_text,
            'critique_text': dossier.critique_text,
            'targeted_research': dossier.targeted_research
        }
        
        # Add bias analysis context if available
        context = None
        if dossier.bias_analysis:
            context = f"Bias Analysis Context:\n{dossier.bias_analysis.bias_analysis_text}"
        
        # Perform contradiction analysis
        contradiction_analyzer = self._get_contradiction_analyzer()
        contradiction_result = await contradiction_analyzer.analyze_contradictory_information(
            question=question,
            research_materials=research_materials,
            context=context
        )
        
        # Add to dossier
        dossier.contradiction_analysis = contradiction_result
        
        contradiction_time = time.time() - contradiction_start_time
        logger.info(f"Contradictory information analysis completed in {contradiction_time:.2f}s for URL {question.page_url}. "
                   f"Detected: {len(contradiction_result.detected_contradictions)} contradictions, "
                   f"Irresolvable: {len(contradiction_result.irresolvable_conflicts)}")
        
        return dossier
    
    async def _add_volatility_analysis(
        self, 
        dossier: EnhancedResearchDossier, 
        question: MetaculusQuestion
    ) -> EnhancedResearchDossier:
        """Add information environment volatility analysis to the dossier."""
        logger.info(f"Starting volatility analysis for URL {question.page_url}")
        volatility_start_time = time.time()
        
        try:
            # Perform volatility analysis
            volatility_analyzer = self._get_volatility_analyzer()
            volatility_result = await volatility_analyzer.analyze_information_volatility(question)
            
            # Add to dossier (convert to data_models.VolatilityAnalysisResult if needed)
            from data_models import VolatilityAnalysisResult as DataModelVolatilityResult
            if not isinstance(volatility_result, DataModelVolatilityResult):
                # Convert to the expected type
                dossier.volatility_analysis = DataModelVolatilityResult(
                    question=volatility_result.question,
                    analyzed_keywords=volatility_result.analyzed_keywords,
                    news_volume=volatility_result.news_volume,
                    sentiment_volatility=volatility_result.sentiment_volatility,
                    conflicting_reports_score=volatility_result.conflicting_reports_score,
                    overall_volatility_score=volatility_result.overall_volatility_score,
                    volatility_level=volatility_result.volatility_level,
                    confidence_adjustment_factor=volatility_result.confidence_adjustment_factor,
                    midpoint_shrinkage_amount=volatility_result.midpoint_shrinkage_amount,
                    detailed_analysis=volatility_result.detailed_analysis
                )
            else:
                dossier.volatility_analysis = volatility_result
            
            volatility_time = time.time() - volatility_start_time
            logger.info(f"Volatility analysis completed in {volatility_time:.2f}s for URL {question.page_url}. "
                       f"Level: {volatility_result.volatility_level}, "
                       f"Score: {volatility_result.overall_volatility_score:.3f}")
        
        except Exception as e:
            logger.error(f"Error during volatility analysis for URL {question.page_url}: {e}")
            # Create a default low-volatility result as fallback
            from data_models import VolatilityAnalysisResult
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
    
    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """
        Orchestrate the configurable ensemble forecasting process.
        
        This method:
        1. Generates an enhanced research dossier with configured analysis
        2. Applies enhanced persona analyses using the dossier
        3. Synthesizes the final forecast with full analytical awareness
        4. Includes appropriate awareness summaries in the explanation
        """
        async with self._concurrency_limiter:
            overall_start_time = time.time()
            
            # --- ENHANCED RESEARCH PHASE ---
            logger.info(f"Starting enhanced research phase for URL {question.page_url}")
            research_start_time = time.time()
            enhanced_dossier = await self._generate_research_dossier(question)
            research_time = time.time() - research_start_time
            
            # --- ENHANCED PERSONA ANALYSIS PHASE ---
            logger.info(f"Starting enhanced persona analysis for URL {question.page_url}")
            persona_start_time = time.time()
            
            persona_reports = []
            for persona_name in self.PERSONAS:
                persona_prompt = self.PERSONAS[persona_name]
                persona_llm_key = self._get_persona_llm(persona_name)
                
                logger.info(f"Generating enhanced {persona_name} analysis for URL {question.page_url} using LLM '{persona_llm_key}'")
                
                # Generate enhanced persona prediction
                persona_step_start = time.time()
                enhanced_prediction = await self._generate_enhanced_persona_prediction(
                    enhanced_dossier, persona_name, persona_prompt
                )
                persona_step_time = time.time() - persona_step_start
                
                logger.info(f"Enhanced {persona_name} analysis completed in {persona_step_time:.2f}s")
                persona_reports.append((persona_name, enhanced_prediction))
            
            persona_total_time = time.time() - persona_start_time
            logger.info(f"All enhanced persona analyses completed in {persona_total_time:.2f}s")
            
            # --- ENHANCED SYNTHESIS PHASE ---
            logger.info(f"Starting enhanced synthesis for URL {question.page_url}")
            synthesis_start_time = time.time()
            
            reasoned_prediction = await self._synthesize_enhanced_ensemble_forecasts(
                question, persona_reports, enhanced_dossier
            )
            
            synthesis_time = time.time() - synthesis_start_time
            logger.info(f"Enhanced synthesis completed in {synthesis_time:.2f}s")
            
            # --- FORMAT FINAL REPORT ---
            awareness_summary = self._create_awareness_summary(enhanced_dossier)
            final_explanation = f"# Configurable Ensemble Forecast\n\n{reasoned_prediction.reasoning}{awareness_summary}"
            
            # Construct the final report object in a type-safe way
            if isinstance(question, BinaryQuestion):
                from forecasting_tools.data_models.binary_report import BinaryReport
                final_report = BinaryReport(
                    question=question,
                    prediction=reasoned_prediction.prediction_value,
                    explanation=final_explanation
                )
            elif isinstance(question, MultipleChoiceQuestion):
                from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
                final_report = MultipleChoiceReport(
                    question=question,
                    prediction=reasoned_prediction.prediction_value,
                    explanation=final_explanation
                )
            elif isinstance(question, NumericQuestion):
                from forecasting_tools.data_models.numeric_report import NumericReport
                final_report = NumericReport(
                    question=question,
                    prediction=reasoned_prediction.prediction_value,
                    explanation=final_explanation
                )
            else:
                raise TypeError(f"Unsupported question type: {type(question)}")
            
            # Apply volatility adjustment if configured and available
            if (self._config.enable_volatility_analysis and 
                enhanced_dossier.volatility_analysis and
                enhanced_dossier.volatility_analysis.confidence_adjustment_factor < 1.0):
                
                # Apply volatility adjustment to prediction
                adjusted_prediction = self._apply_volatility_adjustment(
                    reasoned_prediction.prediction_value,
                    enhanced_dossier.volatility_analysis
                )
                
                if adjusted_prediction != reasoned_prediction.prediction_value:
                    logger.info(f"Applied volatility adjustment: {reasoned_prediction.prediction_value} → {adjusted_prediction}")
                    final_report.prediction = adjusted_prediction
            
            # Log completion summary
            total_time = time.time() - overall_start_time
            enabled_features = self._config.get_enabled_features()
            logger.info(f"CONFIGURABLE FORECAST COMPLETED for URL {question.page_url}: "
                       f"Total={total_time:.2f}s (Research={research_time:.2f}s, "
                       f"Personas={persona_total_time:.2f}s, Synthesis={synthesis_time:.2f}s). "
                       f"Features: {', '.join(enabled_features)}")
            
            # Publish if required
            if self.publish_reports_to_metaculus:
                await final_report.publish_report_to_metaculus()
            
            return final_report
    
    async def _generate_enhanced_persona_prediction(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        persona_name: str,
        persona_prompt: str
    ) -> str:
        """Generate a persona prediction that incorporates all available analysis."""
        # Build enhanced prompt
        prompt = self._build_enhanced_persona_prompt(enhanced_dossier, persona_prompt)
        
        # Get persona-specific LLM
        persona_llm_key = self._get_persona_llm(persona_name)
        
        # Generate prediction
        prediction = await self.get_llm(persona_llm_key, "llm").invoke(prompt)
        
        return prediction
    
    def _build_enhanced_persona_prompt(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        persona_prompt: str
    ) -> str:
        """Build a persona prompt that incorporates all available analysis."""
        # Use the new consolidated PromptBuilder for dynamic prompt generation
        prompt_builder = PromptBuilder(enhanced_dossier.question)
        return prompt_builder.build_persona_prompt(enhanced_dossier, persona_prompt)
    
    async def _synthesize_enhanced_ensemble_forecasts(
        self,
        question: MetaculusQuestion,
        persona_reports: List[Tuple[str, str]],
        enhanced_dossier: EnhancedResearchDossier
    ) -> ReasonedPrediction:
        """Synthesize ensemble forecasts with awareness of all performed analysis."""
        asknews_client = self._create_asknews_client()
        strategy = CritiqueAndRefineStrategy(
            lambda name, kind: self.get_llm(name, "llm"),
            asknews_client=asknews_client,
            logger=logger
        )
        
        # Combine persona reports
        report_texts = []
        for name, report in persona_reports:
            report_texts.append(f"--- REPORT FROM {name.upper()} ---\n{report}\n--- END REPORT ---")
        
        combined_reports = "\n\n".join(report_texts)
        final_answer_format_instruction = strategy.get_final_answer_format_instruction(question)
        
        # Build enhanced synthesis prompt
        synthesis_prompt = self._build_enhanced_synthesis_prompt(
            question, combined_reports, enhanced_dossier, final_answer_format_instruction
        )
        
        # Generate final synthesis
        final_reasoning = await self.get_llm("refined_pred_llm", "llm").invoke(synthesis_prompt)
        prediction = self._parse_final_prediction(question, final_reasoning)
        
        return ReasonedPrediction(prediction_value=prediction, reasoning=final_reasoning)
    
    def _build_enhanced_synthesis_prompt(
        self,
        question: MetaculusQuestion,
        combined_reports: str,
        enhanced_dossier: EnhancedResearchDossier,
        final_answer_format_instruction: str
    ) -> str:
        """Build synthesis prompt that incorporates all available analysis."""
        # Use the new consolidated PromptBuilder for dynamic prompt generation
        prompt_builder = PromptBuilder(question)
        return prompt_builder.build_synthesis_prompt(
            enhanced_dossier, combined_reports, final_answer_format_instruction
        )
    
    def _apply_volatility_adjustment(
        self,
        original_prediction: Any,
        volatility_analysis: VolatilityAnalysisResult
    ) -> Any:
        """Apply volatility adjustment to prediction based on prediction type."""
        if volatility_analysis.confidence_adjustment_factor >= 1.0:
            return original_prediction  # No adjustment needed
        
        shrinkage_amount = volatility_analysis.midpoint_shrinkage_amount
        
        if isinstance(original_prediction, (int, float)):
            # Numeric prediction - shrink towards midpoint (0.5 for binary, appropriate midpoint for numeric)
            if 0 <= original_prediction <= 1:
                # Binary-style prediction
                midpoint = 0.5
                adjusted = original_prediction + (midpoint - original_prediction) * shrinkage_amount
                return max(0.01, min(0.99, adjusted))  # Clamp to valid range
            else:
                # Numeric prediction - would need question range information
                return original_prediction
        
        elif hasattr(original_prediction, '__iter__') and not isinstance(original_prediction, str):
            # List/array prediction (multiple choice) - shrink towards uniform distribution
            try:
                import numpy as np
                pred_array = np.array(original_prediction, dtype=float)
                uniform = np.ones_like(pred_array) / len(pred_array)
                adjusted = pred_array + (uniform - pred_array) * shrinkage_amount
                # Ensure it sums to 1
                adjusted = adjusted / adjusted.sum()
                return adjusted.tolist()
            except:
                return original_prediction
        
        return original_prediction  # Fallback - return unchanged
    
    def _create_awareness_summary(self, enhanced_dossier: EnhancedResearchDossier) -> str:
        """Create a summary of the analytical enhancements for the final explanation."""
        summary_parts = []
        
        if enhanced_dossier.bias_analysis and enhanced_dossier.bias_analysis.detected_biases:
            bias_summary = f"systematic cognitive bias analysis (detected: {', '.join(enhanced_dossier.bias_analysis.detected_biases)})"
            summary_parts.append(bias_summary)
        
        if enhanced_dossier.contradiction_analysis:
            contradiction_count = len(enhanced_dossier.contradiction_analysis.detected_contradictions)
            conflict_count = len(enhanced_dossier.contradiction_analysis.irresolvable_conflicts)
            if contradiction_count > 0:
                contradiction_summary = f"contradictory information analysis ({contradiction_count} contradictions detected, {conflict_count} irresolvable)"
                summary_parts.append(contradiction_summary)
        
        if enhanced_dossier.volatility_analysis:
            volatility_summary = f"information volatility assessment (level: {enhanced_dossier.volatility_analysis.volatility_level})"
            summary_parts.append(volatility_summary)
        
        if summary_parts:
            return f"\n\n## Enhanced Analysis Summary\n\nThis forecast incorporated: {', '.join(summary_parts)}."
        
        return ""


# Factory functions for common configurations
def create_configurable_ensemble_forecaster(**kwargs) -> ConfigurableEnsembleForecaster:
    """Create a standard ensemble forecaster without enhancements."""
    config = ForecasterConfig()  # No enhancements enabled
    return ConfigurableEnsembleForecaster(forecaster_config=config, **kwargs)


def create_configurable_bias_aware_forecaster(**kwargs) -> ConfigurableEnsembleForecaster:
    """Create a forecaster with cognitive bias analysis only."""
    config = ForecasterConfig(enable_bias_analysis=True)
    return ConfigurableEnsembleForecaster(forecaster_config=config, **kwargs)


def create_configurable_contradiction_aware_forecaster(**kwargs) -> ConfigurableEnsembleForecaster:
    """Create a forecaster with bias and contradiction analysis."""
    config = ForecasterConfig(
        enable_bias_analysis=True,
        enable_contradiction_analysis=True
    )
    return ConfigurableEnsembleForecaster(forecaster_config=config, **kwargs)


def create_configurable_volatility_aware_forecaster(**kwargs) -> ConfigurableEnsembleForecaster:
    """Create a forecaster with all analysis enhancements."""
    config = ForecasterConfig(
        enable_bias_analysis=True,
        enable_contradiction_analysis=True,
        enable_volatility_analysis=True
    )
    return ConfigurableEnsembleForecaster(forecaster_config=config, **kwargs)


def create_custom_forecaster(
    enable_bias_analysis: bool = False,
    enable_contradiction_analysis: bool = False,
    enable_volatility_analysis: bool = False,
    **kwargs
) -> ConfigurableEnsembleForecaster:
    """Create a forecaster with custom feature configuration."""
    config = ForecasterConfig(
        enable_bias_analysis=enable_bias_analysis,
        enable_contradiction_analysis=enable_contradiction_analysis,
        enable_volatility_analysis=enable_volatility_analysis
    )
    return ConfigurableEnsembleForecaster(forecaster_config=config, **kwargs)
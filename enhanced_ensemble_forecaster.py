"""
Example: Enhanced Forecaster Using DossierGenerator

This module demonstrates how to use the DossierGenerator pipeline instead of
inheritance-based approach for adding analysis enhancements to forecasters.

This shows how a BiasAwareEnsembleForecaster can be refactored to use composition
instead of inheritance for the research dossier generation.
"""

import logging
import time
from typing import Optional

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import (
    BinaryQuestion,
    ForecastReport,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    ReasonedPrediction,
)

from critique_strategy import CritiqueAndRefineStrategy
from data_models import EnhancedResearchDossier
from dossier_generator import DossierGenerator, DossierGenerationConfig
from forecasters.ensemble import EnsembleForecaster

logger = logging.getLogger(__name__)


class CompositionBasedEnsembleForecaster(EnsembleForecaster):
    """
    Ensemble forecaster that uses composition-based dossier generation
    instead of inheritance for adding analysis enhancements.
    
    This demonstrates how the DossierGenerator pipeline can replace
    the inheritance chain of BiasAware -> ContradictionAware -> VolatilityAware.
    """
    
    def __init__(
        self, 
        *args, 
        dossier_config: Optional[DossierGenerationConfig] = None,
        **kwargs
    ):
        """
        Initialize with optional dossier generation configuration.
        
        Args:
            dossier_config: Configuration for which analysis steps to include
        """
        super().__init__(*args, **kwargs)
        
        # Create dossier generator with the specified configuration
        self._dossier_config = dossier_config or DossierGenerationConfig()
        self._dossier_generator = None  # Lazy initialization
        
        logger.info(f"Initialized CompositionBasedEnsembleForecaster with config: "
                   f"bias={self._dossier_config.enable_bias_analysis}, "
                   f"contradiction={self._dossier_config.enable_contradiction_analysis}, "
                   f"volatility={self._dossier_config.enable_volatility_analysis}")
    
    def _get_dossier_generator(self) -> DossierGenerator:
        """Lazy initialization of dossier generator."""
        if self._dossier_generator is None:
            asknews_client = self._create_asknews_client()
            self._dossier_generator = DossierGenerator(
                get_llm=self.get_llm,
                asknews_client=asknews_client,
                config=self._dossier_config
            )
        return self._dossier_generator
    
    async def _generate_research_dossier(self, question: MetaculusQuestion) -> EnhancedResearchDossier:
        """
        Generate an enhanced research dossier using the configurable pipeline.
        
        This replaces the inheritance-based approach with a composition-based one.
        """
        logger.info(f"Starting composition-based dossier generation for URL {question.page_url}")
        
        # Generate the enhanced dossier using the pipeline
        dossier_generator = self._get_dossier_generator()
        
        # Use the parent's base dossier generation as the starting point
        enhanced_dossier = await dossier_generator.generate_research_dossier(
            question=question,
            base_dossier_generator=super()._generate_research_dossier
        )
        
        return enhanced_dossier
    
    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """
        Orchestrates the ensemble forecasting process using the configurable pipeline.
        
        This method generates an enhanced research dossier and applies it across
        all persona analyses, then synthesizes the results with awareness summaries.
        """
        async with self._concurrency_limiter:
            overall_start_time = time.time()
            
            # Generate enhanced research dossier using the pipeline
            logger.info(f"Starting enhanced research phase for URL {question.page_url}")
            research_start_time = time.time()
            enhanced_dossier = await self._generate_research_dossier(question)
            research_time = time.time() - research_start_time
            
            # Run persona analyses using the enhanced dossier
            logger.info(f"Starting persona analysis for URL {question.page_url}")
            persona_start_time = time.time()
            
            persona_reports = []
            for persona_name in self.PERSONAS:
                persona_prompt = self.PERSONAS[persona_name]
                
                # Generate persona prediction using enhanced dossier
                persona_prediction = await self._generate_enhanced_persona_prediction(
                    enhanced_dossier, persona_prompt
                )
                
                persona_reports.append((persona_name, persona_prediction))
            
            persona_total_time = time.time() - persona_start_time
            
            # Synthesize enhanced ensemble forecasts
            logger.info(f"Starting enhanced synthesis for URL {question.page_url}")
            synthesis_start_time = time.time()
            
            reasoned_prediction = await self._synthesize_enhanced_ensemble_forecasts(
                question, persona_reports, enhanced_dossier
            )
            
            synthesis_time = time.time() - synthesis_start_time
            
            # Create the final report with awareness summary
            dossier_generator = self._get_dossier_generator()
            awareness_summary = dossier_generator.get_analysis_summary(enhanced_dossier)
            final_explanation = f"# Enhanced Ensemble Forecast\n\n{reasoned_prediction.reasoning}{awareness_summary}"
            
            # Construct the final report object
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
                raise TypeError(f"Unsupported question type for final report construction: {type(question)}")
            
            # Log completion and performance
            overall_time = time.time() - overall_start_time
            logger.info(f"Completed enhanced ensemble forecasting for URL {question.page_url} in {overall_time:.2f}s "
                       f"(Research={research_time:.2f}s, Personas={persona_total_time:.2f}s, "
                       f"Synthesis={synthesis_time:.2f}s). Enhanced analysis applied to {len(self.PERSONAS)} personas.")
            
            # Publish if required
            if self.publish_reports_to_metaculus:
                await final_report.publish_report_to_metaculus()
            
            return final_report
    
    async def _generate_enhanced_persona_prediction(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        persona_prompt: str
    ) -> str:
        """
        Generate a persona prediction that incorporates all available analysis.
        
        This method builds an appropriate prompt based on what analysis was performed
        and applies it to generate persona-specific predictions.
        """
        # Build prompt based on available analysis
        prompt = self._build_enhanced_persona_prompt(enhanced_dossier, persona_prompt)
        
        # Generate prediction using the appropriate persona LLM
        persona_llm = self._get_persona_llm_for_prompt(persona_prompt)
        prediction = await self.get_llm(persona_llm, "llm").invoke(prompt)
        
        return prediction
    
    def _build_enhanced_persona_prompt(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        persona_prompt: str
    ) -> str:
        """
        Build a persona prompt that incorporates all available analysis enhancements.
        """
        # Start with base research information
        base_prompt = f"""
        You are a superforecaster producing a prediction with enhanced analytical awareness.

        {persona_prompt}

        Today is {time.strftime("%Y-%m-%d")}

        ## Question
        {enhanced_dossier.question.question_text}

        ## Background
        {enhanced_dossier.question.background_info}

        ## Resolution Criteria
        {enhanced_dossier.question.resolution_criteria}

        ## Research Analysis
        ### Initial Research
        {enhanced_dossier.initial_research}

        ### Initial Prediction
        {enhanced_dossier.initial_prediction_text}

        ### Adversarial Critique
        {enhanced_dossier.critique_text}

        ### Targeted Research
        {enhanced_dossier.targeted_research}
        """
        
        # Add bias analysis if available
        if enhanced_dossier.bias_analysis:
            base_prompt += f"""
        
        ### Cognitive Bias Analysis
        **Detected Biases:** {', '.join(enhanced_dossier.bias_analysis.detected_biases) if enhanced_dossier.bias_analysis.detected_biases else 'None significant'}
        **Severity:** {enhanced_dossier.bias_analysis.severity_assessment}
        **Priority Corrections:** {'; '.join(enhanced_dossier.bias_analysis.priority_corrections)}
        **Confidence Adjustment Recommended:** {enhanced_dossier.bias_analysis.confidence_adjustment_recommended}
        """
        
        # Add contradiction analysis if available
        if enhanced_dossier.contradiction_analysis:
            contradiction_info = enhanced_dossier.contradiction_analysis
            base_prompt += f"""
        
        ### Contradictory Information Analysis
        **Detected Contradictions:** {len(contradiction_info.detected_contradictions)}
        **Irresolvable Conflicts:** {len(contradiction_info.irresolvable_conflicts)}
        **Overall Coherence:** {contradiction_info.overall_coherence_assessment}
        """
        
        # Add volatility analysis if available
        if enhanced_dossier.volatility_analysis:
            volatility_info = enhanced_dossier.volatility_analysis
            base_prompt += f"""
        
        ### Information Volatility Analysis
        **Volatility Level:** {volatility_info.volatility_level}
        **Overall Score:** {volatility_info.overall_volatility_score:.2f}/1.0
        **Confidence Adjustment Factor:** {volatility_info.confidence_adjustment_factor:.2f}
        **Recommended Shrinkage:** {volatility_info.midpoint_shrinkage_amount:.0%} towards midpoint
        """
        
        # Add synthesis instructions
        base_prompt += """
        
        ## Your Enhanced Forecasting Task
        
        Integrate all the analysis above to produce a well-reasoned forecast that:
        1. Addresses any identified cognitive biases with appropriate corrections
        2. Accounts for contradictory information and unresolved conflicts
        3. Adjusts confidence based on information environment volatility
        4. Provides clear reasoning for your final prediction
        
        Focus on producing a robust forecast that acknowledges uncertainties and incorporates
        the systematic analysis performed above.
        """
        
        return base_prompt
    
    def _get_persona_llm_for_prompt(self, persona_prompt: str) -> str:
        """Get the appropriate LLM name for a persona prompt."""
        # This is a simple mapping - in practice, could be more sophisticated
        return "refined_pred_llm"
    
    async def _synthesize_enhanced_ensemble_forecasts(
        self,
        question: MetaculusQuestion,
        persona_reports: list[tuple[str, str]],
        enhanced_dossier: EnhancedResearchDossier
    ) -> ReasonedPrediction:
        """
        Synthesize ensemble forecasts with awareness of all performed analysis.
        """
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
        
        # Build synthesis prompt that incorporates all analysis
        synthesis_prompt = self._build_enhanced_synthesis_prompt(
            question, combined_reports, enhanced_dossier
        )
        
        # Generate final synthesis
        final_reasoning = await self.get_llm("refined_pred_llm", "llm").invoke(synthesis_prompt)
        prediction = self._parse_final_prediction(question, final_reasoning)
        
        return ReasonedPrediction(prediction_value=prediction, reasoning=final_reasoning)
    
    def _build_enhanced_synthesis_prompt(
        self,
        question: MetaculusQuestion,
        combined_reports: str,
        enhanced_dossier: EnhancedResearchDossier
    ) -> str:
        """Build synthesis prompt that incorporates all available analysis."""
        
        # Build analysis context
        analysis_context = ""
        
        if enhanced_dossier.bias_analysis:
            analysis_context += f"""
            
            ## Cognitive Bias Analysis Context
            Detected biases: {', '.join(enhanced_dossier.bias_analysis.detected_biases) if enhanced_dossier.bias_analysis.detected_biases else 'None significant'}
            Severity: {enhanced_dossier.bias_analysis.severity_assessment}
            Confidence adjustment recommended: {enhanced_dossier.bias_analysis.confidence_adjustment_recommended}
            """
        
        if enhanced_dossier.contradiction_analysis:
            contradiction_info = enhanced_dossier.contradiction_analysis
            analysis_context += f"""
            
            ## Contradiction Analysis Context
            Detected contradictions: {len(contradiction_info.detected_contradictions)}
            Irresolvable conflicts: {len(contradiction_info.irresolvable_conflicts)}
            Overall coherence: {contradiction_info.overall_coherence_assessment}
            """
        
        if enhanced_dossier.volatility_analysis:
            volatility_info = enhanced_dossier.volatility_analysis
            analysis_context += f"""
            
            ## Volatility Analysis Context
            Information volatility level: {volatility_info.volatility_level}
            Confidence adjustment factor: {volatility_info.confidence_adjustment_factor:.2f}
            Recommended shrinkage: {volatility_info.midpoint_shrinkage_amount:.0%} towards midpoint
            """
        
        from forecasting_tools import clean_indents
        
        return clean_indents(f"""
            You are a lead superforecaster responsible for producing a final, definitive forecast 
            from comprehensive enhanced analysis. You have received analyses from multiple expert 
            analysts that have been enhanced with systematic analysis steps.

            Your task is to synthesize their enhanced reports, account for all analytical 
            enhancements, and produce a single, coherent final rationale and prediction.

            ## The Question
            {question.question_text}

            ## Enhanced Analyst Reports
            {combined_reports}{analysis_context}

            ## Your Enhanced Synthesis Task

            Integrate all analysis components above into a final prediction that accounts for:
            1. Cognitive bias corrections and confidence adjustments
            2. Contradictory information and its impact on uncertainty  
            3. Information environment volatility and confidence shrinkage
            4. The diverse perspectives from the analyst team

            ### Synthesis Steps:
            1. **Evidence Integration**: Synthesize insights while addressing identified biases
            2. **Contradiction Resolution**: Acknowledge unresolved conflicts as key uncertainties
            3. **Volatility Adjustment**: Apply appropriate confidence adjustments for volatility
            4. **Final Prediction**: Provide a robust forecast with clear reasoning

            Produce your final forecast with explicit acknowledgment of the enhanced analytical 
            process and how it affected your reasoning and confidence levels.
        """)


# Factory functions for common configurations
def create_bias_aware_forecaster(**kwargs):
    """Create a forecaster with bias analysis only."""
    config = DossierGenerationConfig(enable_bias_analysis=True)
    return CompositionBasedEnsembleForecaster(dossier_config=config, **kwargs)


def create_contradiction_aware_forecaster(**kwargs):
    """Create a forecaster with bias and contradiction analysis."""
    config = DossierGenerationConfig(
        enable_bias_analysis=True,
        enable_contradiction_analysis=True
    )
    return CompositionBasedEnsembleForecaster(dossier_config=config, **kwargs)


def create_full_analysis_forecaster(**kwargs):
    """Create a forecaster with all analysis enhancements."""
    config = DossierGenerationConfig(
        enable_bias_analysis=True,
        enable_contradiction_analysis=True,
        enable_volatility_analysis=True
    )
    return CompositionBasedEnsembleForecaster(dossier_config=config, **kwargs)
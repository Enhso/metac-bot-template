"""
Bias-Aware Ensemble Forecaster

This module extends the ensemble forecasting approach to include systematic
cognitive bias analysis. It performs bias checking once during the research
phase and applies the corrections across all persona analyses.
"""

import asyncio
import logging
import time
from typing import overload

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import (
    BinaryQuestion,
    ForecastReport,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    Notepad
)

from critique_strategy import CritiqueAndRefineStrategy
from data_models import ResearchDossier, EnhancedResearchDossier
from forecasting_prompts import PERSONAS as ENSEMBLE_PERSONAS
from forecasters.ensemble import EnsembleForecaster

logger = logging.getLogger(__name__)


class BiasAwareEnsembleForecaster(EnsembleForecaster):
    """
    Enhanced ensemble forecaster that incorporates systematic cognitive bias
    analysis into the forecasting process.
    
    This forecaster extends the ensemble approach by:
    1. Performing cognitive bias analysis once during the research phase
    2. Applying bias corrections to all persona analyses
    3. Including bias-awareness in the final synthesis
    
    The workflow becomes:
    1. Generate research dossier with bias analysis (enhanced)
    2. Apply bias-aware persona analyses 
    3. Synthesize with bias-awareness context
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized BiasAwareEnsembleForecaster with bias-aware persona analysis")

    async def _generate_research_dossier(self, question: MetaculusQuestion) -> EnhancedResearchDossier:
        """
        Generate an enhanced research dossier that includes cognitive bias analysis.
        
        This performs the standard research once and adds cognitive bias analysis
        that will be applied across all persona analyses.
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

    def _parse_bias_analysis(self, question: MetaculusQuestion, analyzed_rationale: str, bias_analysis_text: str):
        """
        Parse the bias analysis text to extract structured information.
        
        This method delegates to the centralized parsing logic in CognitiveBiasChecker
        to ensure consistent parsing across all bias-aware forecasters.
        """
        from cognitive_bias_checker import CognitiveBiasChecker
        return CognitiveBiasChecker.parse_analysis_text(question, analyzed_rationale, bias_analysis_text)

    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """
        Orchestrates the bias-aware ensemble forecasting process for a single question.
        
        This enhanced version includes cognitive bias analysis in the research phase
        and applies bias-aware refinement to all persona analyses.
        """
        async with self._concurrency_limiter:
            overall_start_time = time.time()
            
            # --- ENHANCED RESEARCH PHASE (RUNS ONCE PER QUESTION WITH BIAS ANALYSIS) ---
            logger.info(f"Starting enhanced research phase with bias analysis for URL {question.page_url}")
            research_start_time = time.time()
            enhanced_dossier = await self._generate_research_dossier(question)
            research_time = time.time() - research_start_time
            
            # Log bias analysis summary
            if enhanced_dossier.bias_analysis:
                bias_info = enhanced_dossier.bias_analysis
                logger.info(f"Bias analysis for URL {question.page_url}: "
                           f"Detected {len(bias_info.detected_biases)} biases "
                           f"(Severity: {bias_info.severity_assessment})")
            
            # --- BIAS-AWARE PERSONA ANALYSIS PHASE (RUNS ONCE PER PERSONA) ---
            logger.info(f"Starting bias-aware persona analysis phase for URL {question.page_url}")
            persona_start_time = time.time()
            asknews_client = self._create_asknews_client()
            persona_reports = []
            
            for persona_name in self.PERSONAS:
                persona_prompt = self.PERSONAS[persona_name]
                persona_llm_key = self._get_persona_llm(persona_name)
                logger.info(f"Generating bias-aware {persona_name} analysis for URL {question.page_url} using LLM '{persona_llm_key}'")
                
                # Create a strategy instance with persona-specific LLM
                persona_strategy = CritiqueAndRefineStrategy(
                    lambda name, kind: self.get_llm(persona_llm_key, "llm") if name == "refined_pred_llm" else self.get_llm(name, "llm"), 
                    asknews_client=asknews_client,
                    logger=logger
                )
                
                # Use bias-aware refinement if bias analysis is available
                persona_step_start = time.time()
                if enhanced_dossier.bias_analysis is not None:
                    refined_prediction_text = await persona_strategy.generate_bias_aware_refined_prediction(
                        question=enhanced_dossier.question,
                        initial_research=enhanced_dossier.initial_research,
                        initial_prediction_text=enhanced_dossier.initial_prediction_text,
                        critique_text=enhanced_dossier.critique_text,
                        targeted_research=enhanced_dossier.targeted_research,
                        bias_analysis=enhanced_dossier.bias_analysis.bias_analysis_text,
                        persona_prompt=persona_prompt,
                    )
                else:
                    # Fallback to standard refinement if bias analysis is not available
                    refined_prediction_text = await persona_strategy.generate_refined_prediction(
                        enhanced_dossier.question,
                        enhanced_dossier.initial_research,
                        enhanced_dossier.initial_prediction_text,
                        enhanced_dossier.critique_text,
                        enhanced_dossier.targeted_research,
                        persona_prompt=persona_prompt,
                    )
                
                persona_step_time = time.time() - persona_step_start
                logger.info(f"Bias-aware {persona_name} analysis (LLM: {persona_llm_key}) completed in {persona_step_time:.2f}s for URL {question.page_url}")
                persona_reports.append((persona_name, refined_prediction_text))

            persona_total_time = time.time() - persona_start_time
            logger.info(f"All bias-aware persona analyses completed in {persona_total_time:.2f}s for URL {question.page_url}")

            # --- BIAS-AWARE SYNTHESIS PHASE (RUNS ONCE) ---
            logger.info(f"Starting bias-aware synthesis phase for URL {question.page_url}")
            synthesis_start_time = time.time()
            reasoned_prediction = await self._synthesize_bias_aware_ensemble_forecasts(
                question, persona_reports, enhanced_dossier.bias_analysis
            )
            synthesis_time = time.time() - synthesis_start_time
            logger.info(f"Bias-aware synthesis completed in {synthesis_time:.2f}s for URL {question.page_url}")

            # Log efficiency summary
            total_time = time.time() - overall_start_time
            logger.info(f"BIAS-AWARE EFFICIENCY SUMMARY for URL {question.page_url}: "
                       f"Total={total_time:.2f}s (Research+Bias={research_time:.2f}s, "
                       f"Personas={persona_total_time:.2f}s, Synthesis={synthesis_time:.2f}s). "
                       f"Bias analysis performed ONCE and applied to {len(self.PERSONAS)} persona analyses.")

            # Format the final explanation with bias awareness
            bias_summary = ""
            if enhanced_dossier.bias_analysis and enhanced_dossier.bias_analysis.detected_biases:
                bias_summary = f"\n\n## Cognitive Bias Mitigation\nThis ensemble forecast has been enhanced with systematic cognitive bias analysis. " \
                             f"Detected potential biases: {', '.join(enhanced_dossier.bias_analysis.detected_biases)}. " \
                             f"Bias corrections were applied across all {len(self.PERSONAS)} analytical perspectives."

            final_explanation = f"# Bias-Aware Ensemble Forecast\n\n{reasoned_prediction.reasoning}{bias_summary}"

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
                raise TypeError(f"Unsupported question type for final report construction: {type(question)}")

            # Publish if required
            if self.publish_reports_to_metaculus:
                await final_report.publish_report_to_metaculus()

            logger.info(f"Completed bias-aware ensemble forecasting for URL {question.page_url}")
            return final_report

    async def _synthesize_bias_aware_ensemble_forecasts(
        self, 
        question: MetaculusQuestion, 
        persona_reports: list[tuple[str, str]],
        bias_analysis = None
    ) -> ReasonedPrediction:
        """
        Synthesize ensemble forecasts with explicit bias awareness.
        
        This enhanced synthesis includes information about the cognitive bias
        analysis and corrections applied during the forecasting process.
        """
        asknews_client = self._create_asknews_client()
        strategy = CritiqueAndRefineStrategy(
            lambda name, kind: self.get_llm(name, "llm"), 
            asknews_client=asknews_client,
            logger=logger
        )
        
        report_texts = []
        for name, report in persona_reports:
            report_texts.append(f"--- REPORT FROM {name.upper()} ---\n{report}\n--- END REPORT ---")

        combined_reports = "\n\n".join(report_texts)
        final_answer_format_instruction = strategy.get_final_answer_format_instruction(question)

        # Include bias analysis context in synthesis
        bias_context = ""
        if bias_analysis:
            bias_context = f"""
            
            ## Cognitive Bias Analysis Context
            During the research phase, systematic cognitive bias analysis was performed on the initial reasoning.
            The following biases were identified and corrections were applied to all persona analyses:
            
            **Detected Biases:** {', '.join(bias_analysis.detected_biases) if bias_analysis.detected_biases else 'None significant'}
            **Severity Assessment:** {bias_analysis.severity_assessment}
            **Confidence Adjustment Recommended:** {bias_analysis.confidence_adjustment_recommended}
            
            All analyst reports above have been generated with these bias corrections in mind.
            """

        synthesis_prompt = clean_indents(
            f"""
            You are a lead superforecaster responsible for producing a final, definitive forecast from bias-aware ensemble analysis. You have received analyses from multiple expert analysts, each with a different cognitive style and approach to forecasting. Importantly, all analyses have been enhanced with systematic cognitive bias detection and correction.

            Your task is to synthesize their bias-corrected reports, weigh their arguments, resolve contradictions, and produce a single, coherent final rationale and prediction that accounts for the bias mitigation work performed.

            ## The Question
            {question.question_text}

            ## Bias-Corrected Analyst Reports
            {combined_reports}{bias_context}

            ## Your Enhanced Synthesis Task
            1.  **Synthesize Bias-Aware Arguments:** Briefly summarize the key arguments from each analyst, noting how bias corrections may have influenced their analyses.
            2.  **Weigh Evidence with Bias Awareness:** Critically evaluate the strength of each argument while considering the systematic bias corrections applied. How do the bias corrections affect the reliability of different viewpoints?
            3.  **Final Bias-Resistant Rationale:** Provide your final, synthesized rationale that explicitly accounts for the cognitive bias analysis and corrections performed across all perspectives.
            4.  **Calibrated Final Prediction:** State your final prediction, calibrated with awareness of the bias mitigation work.

            **Required Output Format:**
            **Step 1: Bias-Aware Synthesis of Analyst Views**
            - [Your summary and comparison of the bias-corrected analyst reports]
            **Step 2: Evidence Evaluation with Bias Context**
            - [Your evaluation considering bias corrections and their impact on argument strength]
            **Step 3: Final Bias-Resistant Rationale**
            - [Your comprehensive final rationale accounting for bias mitigation]
            **Step 4: Calibrated Final Prediction**
            - [Your final prediction in the required format:
              {final_answer_format_instruction}]
            """
        )

        final_reasoning = await self.get_llm("refined_pred_llm", "llm").invoke(synthesis_prompt)
        prediction = self._parse_final_prediction(question, final_reasoning)

        return ReasonedPrediction(prediction_value=prediction, reasoning=final_reasoning)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        """
        Returns default LLM configuration including the bias checker LLM.
        """
        defaults = super()._llm_config_defaults()
        
        # Add bias checker LLM (can use the same as critique_llm or a specialized model)
        defaults["bias_checker_llm"] = defaults.get("critique_llm", defaults["refined_pred_llm"])
        
        return defaults
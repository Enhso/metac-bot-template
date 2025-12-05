"""
Contradiction-Aware Ensemble Forecaster

This module extends the bias-aware ensemble forecasting approach to include
systematic contradictory information detection and resolution. It identifies
conflicting evidence during research and attempts to reconcile contradictions
or flag them as key uncertainties in the forecast.
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
from contradictory_information_analyzer import ContradictionAnalysisResult
from forecasting_prompts import PERSONAS as ENSEMBLE_PERSONAS
from forecasters.bias_aware_ensemble import BiasAwareEnsembleForecaster
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ContradictionAwareEnsembleForecaster(BiasAwareEnsembleForecaster):
    """
    Enhanced ensemble forecaster that incorporates both systematic cognitive bias
    analysis and contradictory information detection into the forecasting process.
    
    This forecaster extends the bias-aware ensemble approach by:
    1. Performing cognitive bias analysis during the research phase (inherited)
    2. Analyzing research materials for contradictory information (NEW)
    3. Attempting to resolve contradictions or flag them as uncertainties (NEW)
    4. Applying both bias and contradiction awareness to all persona analyses
    5. Including contradiction analysis in the final synthesis
    
    The workflow becomes:
    1. Generate research dossier with bias analysis (inherited)
    2. Analyze research materials for contradictory information (NEW)
    3. Apply bias-aware and contradiction-aware persona analyses 
    4. Synthesize with full awareness of biases and contradictions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized ContradictionAwareEnsembleForecaster with bias and contradiction analysis")

    async def _generate_research_dossier(self, question: MetaculusQuestion) -> EnhancedResearchDossier:
        """
        Generate an enhanced research dossier that includes both cognitive bias analysis
        and contradictory information analysis.
        
        This performs the standard research once and adds both bias and contradiction
        analysis that will be applied across all persona analyses.
        """
        # Generate the standard research dossier first with bias analysis (from parent)
        enhanced_dossier = await super()._generate_research_dossier(question)
        
        # Perform contradictory information analysis on the research materials
        asknews_client = self._create_asknews_client()
        strategy = CritiqueAndRefineStrategy(
            lambda name, kind: self.get_llm(name, "llm"), 
            asknews_client=asknews_client,
            logger=logger
        )
        
        contradiction_start_time = time.time()
        logger.info(f"Starting contradictory information analysis for URL {question.page_url}")
        
        # Prepare research materials for contradiction analysis
        research_materials = {
            'initial_research': enhanced_dossier.initial_research,
            'initial_prediction': enhanced_dossier.initial_prediction_text,
            'critique_text': enhanced_dossier.critique_text,
            'targeted_research': enhanced_dossier.targeted_research
        }
        
        # Analyze for contradictory information
        contradiction_analysis_result = await strategy._contradiction_analyzer.analyze_contradictory_information(
            question=question,
            research_materials=research_materials,
            context=f"Bias Analysis Context:\n{enhanced_dossier.bias_analysis.bias_analysis_text if enhanced_dossier.bias_analysis else 'No bias analysis available'}"
        )
        
        contradiction_time = time.time() - contradiction_start_time
        logger.info(f"Contradictory information analysis completed in {contradiction_time:.2f}s for URL {question.page_url}")
        
        # Add contradiction analysis to the enhanced dossier
        enhanced_dossier.contradiction_analysis = contradiction_analysis_result
        
        return enhanced_dossier

    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """
        Orchestrates the contradiction-aware ensemble forecasting process for a single question.
        
        This enhanced version includes both cognitive bias analysis and contradictory information
        analysis in the research phase and applies both types of awareness to all persona analyses.
        """
        async with self._concurrency_limiter:
            overall_start_time = time.time()
            
            # --- ENHANCED RESEARCH PHASE (RUNS ONCE PER QUESTION WITH BIAS AND CONTRADICTION ANALYSIS) ---
            logger.info(f"Starting enhanced research phase with bias and contradiction analysis for URL {question.page_url}")
            research_start_time = time.time()
            enhanced_dossier = await self._generate_research_dossier(question)
            research_time = time.time() - research_start_time
            
            # Log analysis summaries
            if enhanced_dossier.bias_analysis:
                bias_info = enhanced_dossier.bias_analysis
                logger.info(f"Bias analysis for URL {question.page_url}: "
                           f"Detected {len(bias_info.detected_biases)} biases "
                           f"(Severity: {bias_info.severity_assessment})")
            
            if enhanced_dossier.contradiction_analysis:
                contradiction_info = enhanced_dossier.contradiction_analysis
                logger.info(f"Contradiction analysis for URL {question.page_url}: "
                           f"Detected {len(contradiction_info.detected_contradictions)} contradictions, "
                           f"resolved {len(contradiction_info.resolution_attempts)}, "
                           f"identified {len(contradiction_info.irresolvable_conflicts)} irresolvable conflicts "
                           f"(Coherence: {contradiction_info.overall_coherence_assessment})")
            
            # --- BIAS AND CONTRADICTION-AWARE PERSONA ANALYSIS PHASE ---
            logger.info(f"Starting bias and contradiction-aware persona analysis phase for URL {question.page_url}")
            persona_start_time = time.time()
            asknews_client = self._create_asknews_client()
            persona_reports = []
            
            for persona_name in self.PERSONAS:
                persona_prompt = self.PERSONAS[persona_name]
                persona_llm_key = self._get_persona_llm(persona_name)
                logger.info(f"Generating bias and contradiction-aware {persona_name} analysis for URL {question.page_url} using LLM '{persona_llm_key}'")
                
                # Create a strategy instance with persona-specific LLM
                persona_strategy = CritiqueAndRefineStrategy(
                    lambda name, kind: self.get_llm(persona_llm_key, "llm") if name == "refined_pred_llm" else self.get_llm(name, "llm"), 
                    asknews_client=asknews_client,
                    logger=logger
                )
                
                # Use enhanced refinement with both bias and contradiction awareness
                persona_step_start = time.time()
                refined_prediction_text = await self._generate_contradiction_aware_prediction(
                    enhanced_dossier, persona_strategy, persona_prompt
                )
                
                persona_step_time = time.time() - persona_step_start
                logger.info(f"Bias and contradiction-aware {persona_name} analysis (LLM: {persona_llm_key}) completed in {persona_step_time:.2f}s for URL {question.page_url}")
                persona_reports.append((persona_name, refined_prediction_text))

            persona_total_time = time.time() - persona_start_time
            logger.info(f"All bias and contradiction-aware persona analyses completed in {persona_total_time:.2f}s for URL {question.page_url}")

            # --- ENHANCED SYNTHESIS PHASE ---
            logger.info(f"Starting enhanced synthesis phase for URL {question.page_url}")
            synthesis_start_time = time.time()
            reasoned_prediction = await self._synthesize_contradiction_aware_ensemble_forecasts(
                question, persona_reports, enhanced_dossier.bias_analysis, enhanced_dossier.contradiction_analysis
            )
            synthesis_time = time.time() - synthesis_start_time
            logger.info(f"Enhanced synthesis completed in {synthesis_time:.2f}s for URL {question.page_url}")

            # Log efficiency summary
            total_time = time.time() - overall_start_time
            logger.info(f"CONTRADICTION-AWARE EFFICIENCY SUMMARY for URL {question.page_url}: "
                       f"Total={total_time:.2f}s (Research+Analysis={research_time:.2f}s, "
                       f"Personas={persona_total_time:.2f}s, Synthesis={synthesis_time:.2f}s). "
                       f"Bias and contradiction analysis performed ONCE and applied to {len(self.PERSONAS)} persona analyses.")

            # Format the final explanation with full awareness
            awareness_summary = self._create_awareness_summary(enhanced_dossier)
            final_explanation = f"# Contradiction and Bias-Aware Ensemble Forecast\n\n{reasoned_prediction.reasoning}{awareness_summary}"

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

            logger.info(f"Completed contradiction-aware ensemble forecasting for URL {question.page_url}")
            return final_report

    async def _generate_contradiction_aware_prediction(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        strategy: CritiqueAndRefineStrategy,
        persona_prompt: str
    ) -> str:
        """
        Generate a prediction that is aware of both cognitive biases and contradictory information.
        """
        # Use consolidated PromptBuilder for prompt generation
        prompt_builder = PromptBuilder(enhanced_dossier.question)
        enhanced_prompt = prompt_builder.build_contradiction_aware_persona_prompt(
            enhanced_dossier, persona_prompt
        )
        
        # Use the refined prediction LLM with the enhanced prompt
        refined_prediction_text = await strategy._get_llm("refined_pred_llm", "llm").invoke(enhanced_prompt)
        
        return refined_prediction_text

    def _get_final_answer_format_instruction(self, question: MetaculusQuestion) -> str:
        """
        Get the final answer format instruction for a question.
        
        Delegates to PromptBuilder for centralized format instruction handling.
        """
        prompt_builder = PromptBuilder(question)
        return prompt_builder.get_detailed_final_answer_format_instruction()

    async def _synthesize_contradiction_aware_ensemble_forecasts(
        self, 
        question: MetaculusQuestion, 
        persona_reports: list[tuple[str, str]],
        bias_analysis = None,
        contradiction_analysis = None
    ) -> ReasonedPrediction:
        """
        Synthesize ensemble forecasts with explicit awareness of both biases and contradictions.
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

        # Include both bias and contradiction analysis context in synthesis
        analysis_context = ""
        
        if bias_analysis:
            analysis_context += f"""
            
            ## Cognitive Bias Analysis Context
            The research was analyzed for cognitive biases with the following findings:
            
            **Detected Biases:** {', '.join(bias_analysis.detected_biases) if bias_analysis.detected_biases else 'None significant'}
            **Severity Assessment:** {bias_analysis.severity_assessment}
            **Confidence Adjustment Recommended:** {bias_analysis.confidence_adjustment_recommended}
            """
        
        if contradiction_analysis:
            analysis_context += f"""
            
            ## Contradictory Information Analysis Context
            The research was analyzed for contradictory information with the following findings:
            
            **Overall Coherence Assessment:** {contradiction_analysis.overall_coherence_assessment}
            **Detected Contradictions:** {len(contradiction_analysis.detected_contradictions)}
            **Irresolvable Conflicts:** {len(contradiction_analysis.irresolvable_conflicts)}
            **Key Uncertainties:** {len(contradiction_analysis.key_uncertainties)}
            **Confidence Impact:** {contradiction_analysis.confidence_impact}
            """

        synthesis_prompt = clean_indents(
            f"""
            You are a lead superforecaster responsible for producing a final, definitive forecast from a comprehensive analysis that includes bias correction and contradiction resolution. You have received analyses from multiple expert analysts, each enhanced with systematic cognitive bias detection and contradictory information analysis.

            Your task is to synthesize their enhanced reports, weigh their arguments while accounting for bias corrections and contradiction resolutions, and produce a single, coherent final rationale and prediction.

            ## The Question
            {question.question_text}

            ## Enhanced Analyst Reports (Bias and Contradiction-Aware)
            {combined_reports}{analysis_context}

            ## Your Comprehensive Synthesis Task
            1.  **Synthesize Enhanced Arguments:** Summarize key arguments from each analyst, noting how bias corrections and contradiction analysis influenced their reasoning.
            2.  **Weigh Evidence with Full Awareness:** Evaluate argument strength considering both bias mitigation and contradiction resolution status.
            3.  **Final Enhanced Rationale:** Provide a rationale that explicitly accounts for both cognitive bias corrections and contradictory information analysis.
            4.  **Calibrated Final Prediction:** State your prediction with confidence adjusted for both types of analysis.

            **Required Output Format:**
            **Step 1: Enhanced Synthesis of Analyst Views**
            - [Your summary considering bias and contradiction awareness]
            **Step 2: Evidence Evaluation with Full Context**
            - [Your evaluation accounting for both bias corrections and contradiction analysis]
            **Step 3: Final Enhanced Rationale**
            - [Your comprehensive rationale with full analytical awareness]
            **Step 4: Calibrated Final Prediction**
            - [Your final prediction in the required format:
              {final_answer_format_instruction}]
            """
        )

        final_reasoning = await self.get_llm("refined_pred_llm", "llm").invoke(synthesis_prompt)
        prediction = self._parse_final_prediction(question, final_reasoning)

        return ReasonedPrediction(prediction_value=prediction, reasoning=final_reasoning)

    def _create_awareness_summary(self, enhanced_dossier: EnhancedResearchDossier) -> str:
        """Create a summary of the bias and contradiction analysis for the final explanation."""
        summary_parts = []
        
        if enhanced_dossier.bias_analysis and enhanced_dossier.bias_analysis.detected_biases:
            summary_parts.append(f"systematic cognitive bias analysis (detected: {', '.join(enhanced_dossier.bias_analysis.detected_biases)})")
        
        if enhanced_dossier.contradiction_analysis:
            contradiction_count = len(enhanced_dossier.contradiction_analysis.detected_contradictions)
            conflict_count = len(enhanced_dossier.contradiction_analysis.irresolvable_conflicts)
            if contradiction_count > 0:
                summary_parts.append(f"contradictory information analysis ({contradiction_count} contradictions detected, {conflict_count} irresolvable)")
        
        if summary_parts:
            return f"\n\n## Enhanced Analytical Process\nThis ensemble forecast has been enhanced with {' and '.join(summary_parts)}. These analyses were applied across all {len(self.PERSONAS)} analytical perspectives to improve forecast robustness and identify key uncertainties."
        
        return ""

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        """
        Returns default LLM configuration including the contradiction analyzer LLM.
        """
        defaults = super()._llm_config_defaults()
        
        # Add contradiction analyzer LLM (can use the same as critique_llm or a specialized model)
        defaults["contradiction_analyzer_llm"] = defaults.get("critique_llm", defaults["refined_pred_llm"])
        
        return defaults
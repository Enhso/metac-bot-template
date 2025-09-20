"""
Ensemble forecasting bot that uses multiple analytical personas to create robust forecasts.
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
from data_models import ResearchDossier
from forecasting_prompts import PERSONAS as ENSEMBLE_PERSONAS
from .self_critique import SelfCritiqueForecaster

logger = logging.getLogger(__name__)


class EnsembleForecaster(SelfCritiqueForecaster):
    """
    This bot implements an ensemble strategy by running the forecasting process
    multiple times, each guided by a different analytical persona. This approach
    aims to produce more robust forecasts by synthesizing diverse perspectives.
    
    Enhanced version supports per-persona LLM assignment for maximum cognitive diversity.
    """
    PERSONAS = ENSEMBLE_PERSONAS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create persona-to-LLM mapping for cognitive diversity
        self._persona_llm_mapping = self._create_persona_llm_mapping()
        logger.info(f"Initialized EnsembleForecaster with {len(self.PERSONAS)} personas and diverse LLM assignments")

    def _create_persona_llm_mapping(self) -> dict[str, str]:
        """
        Create a mapping from persona names to their assigned LLM keys.
        Falls back to refined_pred_llm if persona-specific LLM is not configured.
        """
        mapping = {}
        for persona_name in self.PERSONAS.keys():
            # Convert persona name to LLM key format
            llm_key = f"persona_{persona_name.lower().replace(' ', '_').replace('the_', '')}_llm"
            
            # Check if persona-specific LLM exists, fallback to default
            if self._llms.get(llm_key) is not None:
                mapping[persona_name] = llm_key
                logger.info(f"Persona '{persona_name}' assigned to LLM '{llm_key}'")
            else:
                mapping[persona_name] = "refined_pred_llm"
                logger.warning(f"Persona '{persona_name}' falling back to 'refined_pred_llm' (LLM key '{llm_key}' not found)")
        
        return mapping

    def _get_persona_llm(self, persona_name: str) -> str:
        """
        Get the LLM key assigned to a specific persona.
        
        Args:
            persona_name: The name of the persona
            
        Returns:
            The LLM key to use for this persona
        """
        return self._persona_llm_mapping.get(persona_name, "refined_pred_llm")

    async def _initialize_notepad(self, question: MetaculusQuestion) -> Notepad:
        notepad = await super()._initialize_notepad(question)
        notepad.note_entries["personas"] = list(self.PERSONAS.keys())
        return notepad

    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """
        Orchestrates the ensemble forecasting process for a single question.
        This refactored version separates research from persona-based analysis:
        1. Generate research dossier once (expensive operations)
        2. Apply each persona to the same research artifacts (cheap operations)
        3. Synthesize final forecast from all persona reports
        """
        async with self._concurrency_limiter:
            overall_start_time = time.time()
            
            # --- RESEARCH PHASE (RUNS ONCE PER QUESTION) ---
            logger.info(f"Starting research phase for URL {question.page_url}")
            research_start_time = time.time()
            research_dossier = await self._generate_research_dossier(question)
            research_time = time.time() - research_start_time
            
            # --- ANALYSIS PHASE (RUNS ONCE PER PERSONA) ---
            logger.info(f"Starting persona analysis phase for URL {question.page_url}")
            persona_start_time = time.time()
            asknews_client = self._create_asknews_client()
            strategy = CritiqueAndRefineStrategy(
                lambda name, kind: self.get_llm(name, "llm"), 
                asknews_client=asknews_client,
                logger=logger
            )
            persona_reports = []
            
            for persona_name in self.PERSONAS:
                persona_prompt = self.PERSONAS[persona_name]
                persona_llm_key = self._get_persona_llm(persona_name)
                logger.info(f"Generating {persona_name} analysis for URL {question.page_url} using LLM '{persona_llm_key}'")
                
                # Create a strategy instance with persona-specific LLM
                persona_strategy = CritiqueAndRefineStrategy(
                    lambda name, kind: self.get_llm(persona_llm_key, "llm") if name == "refined_pred_llm" else self.get_llm(name, "llm"), 
                    asknews_client=asknews_client,
                    logger=logger
                )
                
                # Use the pre-generated research dossier for persona analysis
                persona_step_start = time.time()
                refined_prediction_text = await persona_strategy.generate_refined_prediction(
                    research_dossier.question,
                    research_dossier.initial_research,
                    research_dossier.initial_prediction_text,
                    research_dossier.critique_text,
                    research_dossier.targeted_research,
                    persona_prompt=persona_prompt,
                )
                persona_step_time = time.time() - persona_step_start
                logger.info(f"{persona_name} analysis (LLM: {persona_llm_key}) completed in {persona_step_time:.2f}s for URL {question.page_url}")
                persona_reports.append((persona_name, refined_prediction_text))

            persona_total_time = time.time() - persona_start_time
            logger.info(f"All persona analyses completed in {persona_total_time:.2f}s for URL {question.page_url}")

            # --- SYNTHESIS PHASE (RUNS ONCE) ---
            logger.info(f"Starting synthesis phase for URL {question.page_url}")
            synthesis_start_time = time.time()
            reasoned_prediction = await self._synthesize_ensemble_forecasts(question, persona_reports)
            synthesis_time = time.time() - synthesis_start_time
            logger.info(f"Synthesis completed in {synthesis_time:.2f}s for URL {question.page_url}")

            # Log efficiency summary
            total_time = time.time() - overall_start_time
            logger.info(f"EFFICIENCY SUMMARY for URL {question.page_url}: "
                       f"Total={total_time:.2f}s (Research={research_time:.2f}s, "
                       f"Personas={persona_total_time:.2f}s, Synthesis={synthesis_time:.2f}s). "
                       f"Research was performed ONCE instead of {len(self.PERSONAS)} times, "
                       f"saving ~{research_time * (len(self.PERSONAS) - 1):.2f}s and "
                       f"~{((len(self.PERSONAS) - 1) / len(self.PERSONAS)) * 100:.0f}% of research API calls.")

            # Format the final explanation to meet the validation requirement
            final_explanation = f"# Final Synthesized Forecast\n\n{reasoned_prediction.reasoning}"

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

            logger.info(f"Completed ensemble forecasting for URL {question.page_url}")
            return final_report

    async def _synthesize_ensemble_forecasts(
        self, question: MetaculusQuestion, persona_reports: list[tuple[str, str]]
    ) -> ReasonedPrediction:
        """
        Takes the reports from all personas and synthesizes them into a final reasoned prediction.
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

        synthesis_prompt = clean_indents(
            f"""
            You are a lead superforecaster responsible for producing a final, definitive forecast. You have received analyses from multiple expert analysts, each with a different cognitive style and approach to forecasting.

            Your task is to synthesize their reports, weigh their arguments, resolve contradictions, and produce a single, coherent final rationale and prediction.

            ## The Question
            {question.question_text}

            ## Analyst Reports
            {combined_reports}

            ## Your Task
            1.  **Synthesize Arguments:** Briefly summarize the key arguments from each analyst. Identify points of agreement and disagreement across the different perspectives.
            2.  **Weigh the Evidence:** Critically evaluate the strength of each argument. Which analysts' cases are more compelling and why? How do you reconcile their different conclusions? Consider how cognitive diversity might affect the reliability of different viewpoints.
            3.  **Final Rationale:** Provide your final, synthesized rationale. It should reflect your judgment after considering all perspectives and leveraging the wisdom of crowds effect from diverse analytical approaches.
            4.  **Final Prediction:** State your final, calibrated prediction in the required format.

            **Required Output Format:**
            **Step 1: Synthesis of Analyst Views**
            - [Your summary and comparison of the analyst reports, noting areas of consensus and disagreement]
            **Step 2: Weighing the Evidence**
            - [Your evaluation of the competing arguments and how cognitive diversity influences your analysis]
            **Step 3: Final Rationale**
            - [Your comprehensive final rationale synthesizing all perspectives]
            **Step 4: Final Prediction**
            - [Your final prediction in the required format:
              {final_answer_format_instruction}]
            """
        )

        final_reasoning = await self.get_llm("refined_pred_llm", "llm").invoke(synthesis_prompt)
        prediction = self._parse_final_prediction(question, final_reasoning)

        return ReasonedPrediction(prediction_value=prediction, reasoning=final_reasoning)

    def _parse_final_prediction(self, question: MetaculusQuestion, reasoning: str):
        """
        Parse final prediction from synthesized reasoning using centralized logic.
        
        This method delegates to the centralized prediction extraction logic to ensure
        consistency across all prediction parsing in the application.
        """
        return self._extract_prediction_using_centralized_logic(question, reasoning)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        """
        Returns a dictionary of default llms for the ensemble bot, including persona-specific LLMs.
        """
        defaults = super()._llm_config_defaults()
        assert defaults.get("default") is not None
        
        # Add persona-specific LLM defaults (fallback to refined_pred_llm)
        persona_defaults = {}
        for persona_name in cls.PERSONAS.keys():
            llm_key = f"persona_{persona_name.lower().replace(' ', '_').replace('the_', '')}_llm"
            persona_defaults[llm_key] = defaults["refined_pred_llm"]
        
        defaults.update(persona_defaults)
        return defaults

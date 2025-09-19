"""
Self-critique forecasting bot that uses adversarial critique to refine predictions.
"""

import asyncio
import logging
import os
import time
from typing import Sequence, overload

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    ForecastReport,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
)

from critique_strategy import CritiqueAndRefineStrategy
from data_models import ResearchDossier
from report_logger import ReportLogger

logger = logging.getLogger(__name__)


class SelfCritiqueForecaster(ForecastBot):

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    @staticmethod
    def _create_asknews_client() -> AsyncAskNewsSDK | None:
        """
        Create an AsyncAskNewsSDK client if credentials are available.
        
        Returns:
            AsyncAskNewsSDK instance if credentials are available, None otherwise.
        """
        if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
            return AsyncAskNewsSDK(
                client_id=os.getenv("ASKNEWS_CLIENT_ID"),
                client_secret=os.getenv("ASKNEWS_SECRET"),
            )
        return None

    @classmethod
    def log_report_summary(
        cls,
        forecast_reports: Sequence[ForecastReport | BaseException],
        raise_errors: bool = True,
    ) -> None:
        """
        Log forecast reports using the standardized ReportLogger utility.
        
        This method delegates to ReportLogger to provide consistent logging
        across different ForecastBot implementations while handling the specific
        report structure used by this bot (raw explanation rather than structured sections).
        """
        ReportLogger.log_forecast_summary(
            forecast_reports=forecast_reports,
            bot_class_name=cls.__name__,
            raise_errors=raise_errors,
            use_structured_sections=False,  # Use raw explanation for self-critique reports
            max_explanation_length=10000,
        )

    async def _generate_research_dossier(self, question: MetaculusQuestion) -> ResearchDossier:
        """
        Generates a complete research dossier for a question containing all research artifacts.
        This method performs the expensive research operations only once per question.
        """
        start_time = time.time()
        logger.info(f"Starting research dossier generation for URL {question.page_url}")
        
        asknews_client = self._create_asknews_client()
        strategy = CritiqueAndRefineStrategy(
            lambda name, kind: self.get_llm(name, "llm"), 
            asknews_client=asknews_client,
            logger=logger
        )
        
        # Perform shared research pipeline with timing for each step
        step_start = time.time()
        initial_research = await strategy.initial_research(question)
        logger.info(f"Initial research completed in {time.time() - step_start:.2f}s for URL {question.page_url}")
        
        step_start = time.time()
        initial_prediction_text = await strategy.generate_initial_prediction(question, initial_research)
        logger.info(f"Initial prediction completed in {time.time() - step_start:.2f}s for URL {question.page_url}")
        
        step_start = time.time()
        critique_text = await strategy.generate_adversarial_critique(question, initial_prediction_text)
        logger.info(f"Adversarial critique completed in {time.time() - step_start:.2f}s for URL {question.page_url}")
        
        step_start = time.time()
        targeted_research = await strategy.perform_targeted_search(critique_text)
        logger.info(f"Targeted search completed in {time.time() - step_start:.2f}s for URL {question.page_url}")
        
        total_time = time.time() - start_time
        logger.info(f"Research dossier generation completed in {total_time:.2f}s for URL {question.page_url}")
        
        return ResearchDossier(
            question=question,
            initial_research=initial_research,
            initial_prediction_text=initial_prediction_text,
            critique_text=critique_text,
            targeted_research=targeted_research
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates the self-critique forecasting process by generating a research dossier
        and producing a final refined prediction using a neutral (no-persona) approach.
        """
        async with self._concurrency_limiter:
            # Generate the research dossier with all artifacts
            research_dossier = await self._generate_research_dossier(question)
            
            # Generate final refined prediction using the research dossier (no persona)
            asknews_client = self._create_asknews_client()
            strategy = CritiqueAndRefineStrategy(
                lambda name, kind: self.get_llm(name, "llm"), 
                asknews_client=asknews_client,
                logger=logger
            )
            
            refined_prediction_text = await strategy.generate_refined_prediction(
                research_dossier.question,
                research_dossier.initial_research,
                research_dossier.initial_prediction_text,
                research_dossier.critique_text,
                research_dossier.targeted_research,
                persona_prompt=None,  # No persona for SelfCritiqueForecaster
            )

            logger.info(f"Completed self-critique process for URL {question.page_url}")

            comment = f"""
## Initial Prediction
{research_dossier.initial_prediction_text}

## Adversarial Critique
{research_dossier.critique_text}

## Final Refined Prediction & Rationale
{refined_prediction_text}
"""

            return comment

    def _extract_prediction_using_centralized_logic(self, question: MetaculusQuestion, text: str):
        """
        Centralized prediction extraction logic using forecasting_tools.PredictionExtractor.
        
        This method consolidates all prediction parsing logic to eliminate duplication and
        ensure consistent behavior across the application.
        
        Args:
            question: The MetaculusQuestion object containing question metadata
            text: The raw LLM output text containing the prediction
            
        Returns:
            The parsed prediction value (type varies by question type)
        """
        if isinstance(question, BinaryQuestion):
            return PredictionExtractor.extract_last_percentage_value(
                text, max_prediction=1, min_prediction=0
            )
        elif isinstance(question, MultipleChoiceQuestion):
            return PredictionExtractor.extract_option_list_with_percentage_afterwards(
                text, question.options
            )
        elif isinstance(question, NumericQuestion):
            dist = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                text, question
            )
            # Apply normalization: sort percentiles to ensure proper ordering
            dist.declared_percentiles.sort(key=lambda p: p.percentile)
            return dist
        else:
            raise TypeError(f"Unsupported question type for prediction parsing: {type(question)}")

    @overload
    def _parse_and_normalize_prediction(self, question: BinaryQuestion, research: str) -> float:
        ...

    @overload
    def _parse_and_normalize_prediction(self, question: MultipleChoiceQuestion, research: str) -> PredictedOptionList:
        ...

    @overload
    def _parse_and_normalize_prediction(self, question: NumericQuestion, research: str) -> NumericDistribution:
        ...

    def _parse_and_normalize_prediction(self, question: MetaculusQuestion, research: str):
        """
        Parse and normalize predictions from raw LLM output using centralized PredictionExtractor logic.
        
        This method delegates to forecasting_tools.PredictionExtractor for consistent parsing
        across the application while preserving logging for debugging.
        
        Args:
            question: The MetaculusQuestion object containing question metadata
            research: The raw LLM output text containing the prediction
            
        Returns:
            The parsed and normalized prediction value (type varies by question type)
        """
        prediction = self._extract_prediction_using_centralized_logic(question, research)
        logger.info(f"Extracted final {type(question).__name__} prediction for URL {question.page_url}: {prediction}")
        return prediction

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prediction = self._parse_and_normalize_prediction(question, research)
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=research
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prediction = self._parse_and_normalize_prediction(question, research)
        return ReasonedPrediction(prediction_value=prediction, reasoning=research)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        prediction = self._parse_and_normalize_prediction(question, research)
        return ReasonedPrediction(prediction_value=prediction, reasoning=research)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        """
        Returns a dictionary of default llms for the bot.
        """
        defaults = super()._llm_config_defaults()
        assert defaults.get("default") is not None
        defaults.update({
            "initial_pred_llm": defaults["default"],
            "critique_llm": defaults["default"],
            "refined_pred_llm": defaults["default"],
            "keyword_extractor_llm": defaults["default"],
            "summarizer": defaults["default"],
            "parser": defaults["default"],
            "researcher": defaults["default"],
        })
        return defaults

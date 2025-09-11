import argparse
import asyncio
import logging
import traceback
import time
from pathlib import Path
from dataclasses import dataclass

from typing import Literal, Sequence, overload
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    ForecastReport,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    clean_indents,
    Notepad
)
from critique_strategy import CritiqueAndRefineStrategy

from config.loader import load_bot_config, default_config_path
from forecasting_prompts import PERSONAS as ENSEMBLE_PERSONAS

logger = logging.getLogger(__name__)


@dataclass
class ResearchDossier:
    """
    Contains all research artifacts generated during the shared research phase.
    This dossier is used as input for persona-based analysis, avoiding the need
    to repeat expensive research operations for each persona.
    """
    question: MetaculusQuestion
    initial_research: str
    initial_prediction_text: str
    critique_text: str
    targeted_research: str


class SelfCritiqueForecaster(ForecastBot):

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    @classmethod
    def log_report_summary(
        cls,
        forecast_reports: Sequence[ForecastReport | BaseException],
        raise_errors: bool = True,
    ) -> None:
        """
        A specialized logger for the EnsembleForecaster that understands the structure
        of its synthesized reports.
        """
        valid_reports = [
            report for report in forecast_reports if isinstance(report, ForecastReport)
        ]

        full_summary = "\n"
        full_summary += "-" * 100 + "\n"

        for report in valid_reports:
            # This is the key change: we don't try to parse sections like .summary
            # We just display the whole explanation.
            question_summary = clean_indents(
                f"""
                URL: {report.question.page_url}
                Errors: {report.errors}
                <<<<<<<<<<<<<<<<<<<< Synthesized Ensemble Report >>>>>>>>>>>>>>>>>>>>>
                {report.explanation[:10000]}
                -------------------------------------------------------------------------------------------
            """
            )
            full_summary += question_summary + "\n"

        full_summary += f"Bot: {cls.__name__}\n"
        for report in forecast_reports:
            if isinstance(report, ForecastReport):
                short_summary = f"✅ URL: {report.question.page_url} | Minor Errors: {len(report.errors)}"
            else:
                exception_message = (
                    str(report)
                    if len(str(report)) < 1000
                    else f"{str(report)[:500]}...{str(report)[-500:]}"
                )
                short_summary = f"❌ Exception: {report.__class__.__name__} | Message: {exception_message}"
            full_summary += short_summary + "\n"

        total_cost = sum(
            report.price_estimate if report.price_estimate else 0
            for report in valid_reports
        )
        average_minutes = (
            (
                sum(
                    report.minutes_taken if report.minutes_taken else 0
                    for report in valid_reports
                )
                / len(valid_reports)
            )
            if valid_reports
            else 0
        )
        average_cost = total_cost / len(valid_reports) if valid_reports else 0
        full_summary += "\nStats for passing reports:\n"
        full_summary += f"Total cost estimated: ${total_cost:.5f}\n"
        full_summary += f"Average cost per question: ${average_cost:.5f}\n"
        full_summary += (
            f"Average time spent per question: {average_minutes:.4f} minutes\n"
        )
        full_summary += "-" * 100 + "\n\n\n"
        logger.info(full_summary)

        exceptions = [
            report for report in forecast_reports if isinstance(report, BaseException)
        ]

        if exceptions and raise_errors:
            for exc in exceptions:
                logger.error(
                    "Exception occurred during forecasting:\n%s",
                    "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )
            raise RuntimeError(
                f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
            )



    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates the entire "self-critique" forecasting process.
        """
        async with self._concurrency_limiter:
            # Use the centralized CritiqueAndRefineStrategy for orchestration
            strategy = CritiqueAndRefineStrategy(lambda name, kind: self.get_llm(name, "llm"), logger)

            # STEP 1: Initial, broad research.
            initial_research = await strategy.initial_research(question)

            # STEP 2: Initial prediction
            initial_prediction_text = await strategy.generate_initial_prediction(question, initial_research)

            # STEP 3: Generate Adversarial Critique
            critique_text = await strategy.generate_adversarial_critique(question, initial_prediction_text)

            # STEP 4: Perform Targeted Search
            targeted_research = await strategy.perform_targeted_search(critique_text)

            # STEP 5: Generate Refined Prediction
            refined_prediction_text = await strategy.generate_refined_prediction(
                question,
                initial_research,
                initial_prediction_text,
                critique_text,
                targeted_research,
            )

            logger.info(
                f"Completed self-critique process for URL {question.page_url}"
            )

            comment = f"""
## Initial Prediction
{initial_prediction_text}

## Adversarial Critique
{critique_text}

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

class EnsembleForecaster(SelfCritiqueForecaster):
    """
    This bot implements an ensemble strategy by running the forecasting process
    multiple times, each guided by a different analytical persona. This approach
    aims to produce more robust forecasts by synthesizing diverse perspectives.
    """
    PERSONAS = ENSEMBLE_PERSONAS

    async def _initialize_notepad(self, question: MetaculusQuestion) -> Notepad:
        notepad = await super()._initialize_notepad(question)
        notepad.note_entries["personas"] = list(self.PERSONAS.keys())
        return notepad

    async def _generate_research_dossier(self, question: MetaculusQuestion) -> ResearchDossier:
        """
        Generates a complete research dossier for a question containing all research artifacts.
        This method performs the expensive research operations only once per question.
        """
        start_time = time.time()
        logger.info(f"Starting research dossier generation for URL {question.page_url}")
        
        strategy = CritiqueAndRefineStrategy(lambda name, kind: self.get_llm(name, "llm"), logger)
        
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
            strategy = CritiqueAndRefineStrategy(lambda name, kind: self.get_llm(name, "llm"), logger)
            persona_reports = []
            
            for persona_name in self.PERSONAS:
                persona_prompt = self.PERSONAS[persona_name]
                logger.info(f"Generating {persona_name} analysis for URL {question.page_url}")
                
                # Use the pre-generated research dossier for persona analysis
                persona_step_start = time.time()
                refined_prediction_text = await strategy.generate_refined_prediction(
                    research_dossier.question,
                    research_dossier.initial_research,
                    research_dossier.initial_prediction_text,
                    research_dossier.critique_text,
                    research_dossier.targeted_research,
                    persona_prompt=persona_prompt,
                )
                persona_step_time = time.time() - persona_step_start
                logger.info(f"{persona_name} analysis completed in {persona_step_time:.2f}s for URL {question.page_url}")
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


    async def _get_initial_research(self, question: MetaculusQuestion) -> str:
        """Helper to get initial research for a question."""
        strategy = CritiqueAndRefineStrategy(lambda name, kind: self.get_llm(name, "llm"), logger)
        return await strategy.initial_research(question)



    async def _synthesize_ensemble_forecasts(
        self, question: MetaculusQuestion, persona_reports: list[tuple[str, str]]
    ) -> ReasonedPrediction:
        """
        Takes the reports from all personas and synthesizes them into a final reasoned prediction.
        """
        strategy = CritiqueAndRefineStrategy(lambda name, kind: self.get_llm(name, "llm"), logger)
        
        report_texts = []
        for name, report in persona_reports:
            report_texts.append(f"--- REPORT FROM {name.upper()} ---\n{report}\n--- END REPORT ---")

        combined_reports = "\n\n".join(report_texts)
        final_answer_format_instruction = strategy.get_final_answer_format_instruction(question)

        synthesis_prompt = clean_indents(
            f"""
            You are a lead superforecaster responsible for producing a final, definitive forecast. You have received analyses from three of your expert analysts, each with a different cognitive style: a Skeptic, a Proponent, and a Quant.

            Your task is to synthesize their reports, weigh their arguments, resolve contradictions, and produce a single, coherent final rationale and prediction.

            ## The Question
            {question.question_text}

            ## Analyst Reports
            {combined_reports}

            ## Your Task
            1.  **Synthesize Arguments:** Briefly summarize the key arguments from each analyst. Identify points of agreement and disagreement.
            2.  **Weigh the Evidence:** Critically evaluate the strength of each argument. Which analyst's case is more compelling and why? How do you reconcile their different conclusions?
            3.  **Final Rationale:** Provide your final, synthesized rationale. It should reflect your judgment after considering all three perspectives.
            4.  **Final Prediction:** State your final, calibrated prediction in the required format.

            **Required Output Format:**
            **Step 1: Synthesis of Analyst Views**
            - [Your summary and comparison of the analyst reports]
            **Step 2: Weighing the Evidence**
            - [Your evaluation of the competing arguments]
            **Step 3: Final Rationale**
            - [Your comprehensive final rationale]
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

def create_ensemble_forecaster(config_path: str | Path | None = None) -> EnsembleForecaster:
    """
    Create an EnsembleForecaster instance with configuration loaded from YAML.
    
    This is the centralized factory function that ensures consistent configuration
    loading across all entry points. It provides a single source of truth for
    bot configuration and prevents misconfiguration errors.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default path.
        
    Returns:
        Configured EnsembleForecaster instance ready for use.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if config_path is None:
        config_path = str(default_config_path())
    
    logger.info(f"Loading bot configuration from: {config_path}")
    
    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(config_path)
    
    # Log key configuration values for verification
    logger.info(f"Configuration loaded successfully:")
    logger.info(f"  - research_reports_per_question: {bot_cfg.get('research_reports_per_question')}")
    logger.info(f"  - publish_reports_to_metaculus: {bot_cfg.get('publish_reports_to_metaculus')}")
    logger.info(f"  - skip_previously_forecasted_questions: {bot_cfg.get('skip_previously_forecasted_questions')}")
    logger.info(f"  - folder_to_save_reports_to: {bot_cfg.get('folder_to_save_reports_to')}")
    logger.info(f"  - Number of LLMs configured: {len(llms)}")
    
    # Validate that essential configuration is present
    if 'research_reports_per_question' not in bot_cfg:
        raise ValueError("Missing required configuration: research_reports_per_question")
    
    # Construct the forecaster from externalized configuration
    forecaster = EnsembleForecaster(llms=llms, **bot_cfg)
    
    logger.info("EnsembleForecaster created successfully with loaded configuration")
    return forecaster


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the EnsembleForecaster forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions", "minibench"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path()),
        help="Path to YAML config file (default: config/bot_config.yaml)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions", "minibench"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
        "minibench",
    ], "Invalid run mode"

    # Create forecaster using the centralized configuration factory
    bot_one = create_ensemble_forecaster(args.config)

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            bot_one.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        bot_one.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot_one.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029        ]
        ]
        bot_one.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            bot_one.forecast_questions(questions, return_exceptions=True)
        )
    elif run_mode == "minibench":
        forecast_reports = asyncio.run(
            bot_one.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )

    EnsembleForecaster.log_report_summary(forecast_reports)

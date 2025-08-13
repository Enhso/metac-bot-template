import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
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
)

from asknews_sdk import AsyncAskNewsSDK

logger = logging.getLogger(__name__)

class SelfCritiqueForecaster(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = 1  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def _generate_initial_prediction(
        self, question: MetaculusQuestion, initial_research: str
    ) -> str:
        """
        Generates an initial prediction based on the broad, initial research.
        """
        # WHY: This prompt is designed to get a quick, baseline forecast.
        # It's a simplified version of the original prompts, asking for a rationale
        # and a clearly formatted prediction to kickstart the critique process.
        prompt = clean_indents(
            f"""
            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}

            Available Research:
            {initial_research}

            Based *only* on the information above, provide a brief, initial forecast.
            State your reasoning and conclude with your prediction in the format required by the question type (e.g., "Probability: ZZ%", a list of option probabilities, or a percentile distribution).
            """
        )

        # WHY: We use the 'initial_pred_llm' here, which can be a faster, cheaper model
        # since this is just a first draft that we expect to improve upon.
        initial_prediction_text = await self.get_llm("initial_pred_llm", "llm").invoke(
            prompt
        )
        logger.info(f"Generated initial prediction for URL {question.page_url}")
        return initial_prediction_text

    async def _generate_adversarial_critique(self):
        prompt = clean_indents(
            f"""
            """
        )

    async def _perform_targeted_search(self):
        prompt = clean_indents(
            f"""
            """
        )

    async def _generate_refined_prediction(self):
        prompt = clean_indents(
            f"""
            """
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates the entire "self-critique" forecasting process.
        """
        # WHY: We use the concurrency limiter here to manage costs and avoid rate limit errors,
        # as this entire block constitutes one "unit of work" for a single question.
        async with self._concurrency_limiter:
            # === STEP 1: Initial, broad research. ===
            logger.info(f"Starting self-critique process for: {question.page_url}")
            initial_research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                initial_research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            else:
                logger.warning(
                    f"No research provider found. Proceeding without initial research."
                )

            # WHY: This is a simple but effective way to handle rate limiting.
            # We pause for a second after a burst of API calls to avoid being blocked.
            await asyncio.sleep(1)

            # === STEP 2: Generate Initial Prediction ===
            initial_prediction_text = await self._generate_initial_prediction(question, initial_research)

            # === STEP 3: Generate Adversarial Critique ===
            critique_text = await self._generate_adversarial_critique(question, initial_prediction_text)
            logger.info("Critique generated. Now performing targeted search.")

            # === STEP 4: Perform Targeted Search ===
            targeted_research = await self._perform_targeted_search(critique_text)

            # WHY: Pause again after our second set of API calls.
            await asyncio.sleep(1)

            # === STEP 5: Generate Refined Prediction ===
            refined_prediction_text = await self._generate_refined_prediction(
                question, initial_research, initial_prediction_text, critique_text, targeted_research
            )

            # WHY: We combine all the steps into a single, comprehensive report.
            # This provides full transparency of the bot's reasoning process.
            full_report = f"""
# ============== FORECASTING PROCESS REPORT ==============
## 1. Initial Broad Research
{initial_research}

## 2. Initial Prediction & Rationale
{initial_prediction_text}

## 3. Adversarial Critique
{critique_text}

## 4. Targeted Research Based on Critique
{targeted_research}

## 5. Final Refined Prediction & Rationale
{refined_prediction_text}
# ================= END OF REPORT =================
"""
            logger.info(
                f"Completed self-critique process for URL {question.page_url}"
            )
            return full_report

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # WHY: The 'research' variable now contains the entire report from our
        # self-critique loop. The final, refined prediction and rationale are
        # at the end of this string. The hard work of reasoning is already done.
        # We simply extract the final answer. The full report becomes the reasoning.
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            research, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Extracted final prediction for URL {question.page_url}: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=research # The whole report is the reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        # WHY: Same logic as the binary method. We rely on the refined prediction step
        # to have formatted the answer correctly for the extractor to find it.
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                research, question.options
            )
        )
        logger.info(
            f"Extracted final prediction for URL {question.page_url}: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=research
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        # WHY: And again for numeric. The 'research' string from the critique loop
        # is passed directly to the extractor.
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                research, question
            )
        )
        logger.info(
            f"Extracted final prediction for URL {question.page_url}: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=research
        )

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
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        """
        Returns a dictionary of default llms for the bot.
        """
        # WHY: We are extending the base class's defaults with our new,
        # purpose-specific LLMs. This follows the library's design pattern
        # and silences the warnings at startup.
        defaults = super()._llm_config_defaults()
        defaults.update({
            "initial_pred_llm": defaults["default"],
            "critique_llm": defaults["default"],
            "refined_pred_llm": defaults["default"],
        })
        return defaults

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
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    bot_one = SelfCritiqueForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="metaculus/openai/gpt-4.1",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
                max_tokens=1024,
            ),
            "initial_pred_llm": GeneralLlm(
                model="metaculus/openai/gpt-4.1",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
                max_tokens=2048,
            ),
            "critique_llm": GeneralLlm(
                model="metaculdefaultus/openai/gpt-4.1",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
                max_tokens=2048,
            ),
            "refined_pred_llm": GeneralLlm(
                model="metaculus/anthropic/claude-3-7-sonnet-latest",
                temperature=0.3,
                timeout=80,
                allowed_tries=2,
                max_tokens=6144,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 4096,
                },
            ),
            "summarizer": GeneralLlm(
                model="metaculus/openai/gpt-4.1",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
                max_tokens=2048,
            ),
        },
    )

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
        ]
        bot_one.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            bot_one.forecast_questions(questions, return_exceptions=True)
        )
    SelfCritiqueForecaster.log_report_summary(forecast_reports)

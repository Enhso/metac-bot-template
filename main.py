import argparse
import asyncio
import logging
import os
import httpx

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

    async def _generate_adversarial_critique(
        self, question: MetaculusQuestion, initial_prediction_text: str
    ) -> str:
        """
        Challenges the initial prediction to find weaknesses.
        """
        # WHY: This prompt uses a "red team" or "devil's advocate" persona. This is critical
        # for identifying biases (like confirmation bias) and uncovering hidden assumptions
        # in the initial forecast. It explicitly asks for actionable questions.
        prompt = clean_indents(
            f"""
            You are a skeptical analyst assigned to critique a colleague's forecast.
            Your goal is to find flaws and weaknesses. Do not be agreeable.

            The original question is: {question.question_text}

            Here is your colleague's initial forecast and rationale:
            ---
            {initial_prediction_text}
            ---

            Critique this forecast. Point out potential biases, flawed logic, and key unstated assumptions.
            Most importantly, conclude with a list of 2-3 specific, researchable questions that, if answered, would most significantly challenge or confirm this initial forecast.
            """
        )

        # WHY: The 'critique_llm' can be a model optimized for critical reasoning.
        # Its job is not to be creative but to be analytical and find faults.
        critique_text = await self.get_llm("critique_llm", "llm").invoke(prompt)
        logger.info(f"Generated critique for URL {question.page_url}")
        return critique_text

    async def _perform_targeted_search(self, critique_text: str) -> str:
        """
        Performs a targeted search based on the questions raised in the critique.
        """
        # WHY: This is a simple but powerful step. We use an LLM to extract ONLY the
        # questions from the critique, ensuring our subsequent search is highly focused
        # and not diluted by the rest of the critique text.
        extraction_prompt = clean_indents(
            f"""
            Extract the specific, researchable questions from the following text.
            List only the questions, each on a new line. If there are no questions, return an empty string.

            Text:
            ---
            {critique_text}
            ---
            """
        )
        # We can use a fast model for this simple extraction task.
        questions_text = await self.get_llm("summarizer", "llm").invoke(extraction_prompt)

        if not questions_text.strip():
            logger.warning("No new research questions were generated from the critique.")
            return "No targeted search was performed as no new questions were identified."

        # WHY: We now use the extracted questions as direct queries for our news searcher.
        # This directly links the identified weakness to a data-gathering step,
        # forming the core of the refinement loop.
        logger.info(f"Performing targeted search with queries: {questions_text.splitlines()}")
        # We assume AskNewsSearcher can take the raw text block of questions.
        targeted_research = await AskNewsSearcher().get_formatted_news_async(
            questions_text
        )
        return targeted_research

    async def _generate_refined_prediction(
        self,
        question: MetaculusQuestion,
        initial_research: str,
        initial_prediction_text: str,
        critique_text: str,
        targeted_research: str,
    ) -> str:
        """
        Synthesizes all information into a final, refined prediction.
        """
        # WHY: This is the most important prompt. It synthesizes all previous steps.
        # By providing the full context (initial thought, critique, new data), we enable
        # the LLM to produce a more robust and well-reasoned final forecast.
        # It explicitly asks the model to address the critique.
        prompt = clean_indents(
            f"""
            You are a senior forecaster responsible for producing a final, high-quality prediction.
            You have been provided with a full dossier on the question.

            ## Original Question
            {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated"}
            Today is {datetime.now().strftime("%Y-%m-%d")}.

            ## Dossier

            ### 1. Initial Broad Research
            {initial_research}

            ### 2. Initial Prediction and Rationale
            {initial_prediction_text}

            ### 3. Adversarial Critique of Initial Prediction
            {critique_text}

            ### 4. New, Targeted Research Based on Critique
            {targeted_research}

            ## Your Task
            Your task is to synthesize all of the above information into a single, final forecast.
            1.  Start by explicitly stating how the critique and targeted research have changed your initial view.
            2.  Provide a comprehensive final rationale for your prediction.
            3.  Conclude with your final prediction, ensuring it is in the precise format required for parsing. For example:
                - For a binary question: "Probability: ZZ%"
                - For a multiple choice question, list each option with its probability: "Option_A: P_A%\\nOption_B: P_B%..."
                - For a numeric question, provide the requested percentiles: "Percentile 10: XX\\nPercentile 20: XX..."
            """
        )

        # WHY: We use our most powerful and context-aware LLM here ('refined_pred_llm').
        # This is where we want to spend our token budget, as it's the step that
        # produces the final, user-facing output.
        refined_prediction_text = await self.get_llm("refined_pred_llm", "llm").invoke(
            prompt
        )
        logger.info(f"Generated refined prediction for URL {question.page_url}")
        return refined_prediction_text


    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates the entire "self-critique" forecasting process.
        """
        # WHY: We use the concurrency limiter here to manage costs and avoid rate limit errors,
        # as this entire block constitutes one "unit of work" for a single question.
        async with self._concurrency_limiter:
            # STEP 1: Initial, broad research.
            initial_research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                initial_research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            else:
                logger.warning(
                    f"No research provider found. Proceeding without initial research for URL {question.page_url}."
                )

            # STEP 2: Generate Initial Prediction
            initial_prediction_text = await self._generate_initial_prediction(question, initial_research)

            # STEP 3: Generate Adversarial Critique
            critique_text = await self._generate_adversarial_critique(question, initial_prediction_text)

            # WHY: We add a short delay here to avoid hitting the AskNews API rate limit.
            # This politely spaces out our initial research calls from our targeted research calls.
            await asyncio.sleep(1) # Add this line

            # STEP 4: Perform Targeted Search
            targeted_research = await self._perform_targeted_search(critique_text)

            # STEP 5: Generate Refined Prediction
            refined_prediction_text = await self._generate_refined_prediction(
                question, initial_research, initial_prediction_text, critique_text, targeted_research
            )

            # WHY: We combine all the steps into a single, comprehensive report.
            # This report becomes the `research` input for the next stage and is
            # ultimately saved in the `explanation` field of the ForecastReport,
            # providing full transparency of the bot's reasoning process.
            full_report = f"""
# ============== FORECASTING PROCESS REPORT ==============
## 1. Initial Research
{initial_research}

## 2. Initial Prediction
{initial_prediction_text}

## 3. Adversarial Critique
{critique_text}

## 4. Targeted Research
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
                model="metaculus/openai/gpt-4.1",
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

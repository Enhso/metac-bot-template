import argparse
import asyncio
import logging
import os

from datetime import datetime
from typing import Literal
from forecasting_tools import (
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

    def _normalize_probabilities(self, predictions: PredictedOptionList) -> PredictedOptionList:
        """
        Clamps probabilities to the Metaculus API limits [0.001, 0.999] and re-normalizes.
        """
        min_prob = 0.001
        max_prob = 0.999

        # Clamp values that are too low or too high
        for p in predictions.predicted_options:
            if p.probability < min_prob:
                p.probability = min_prob
            if p.probability > max_prob:
                p.probability = max_prob

        # Re-normalize to ensure the sum is 1.0
        total = sum(p.probability for p in predictions.predicted_options)
        for p in predictions.predicted_options:
            p.probability /= total

        return predictions

    async def _generate_initial_prediction(
        self, question: MetaculusQuestion, initial_research: str
    ) -> str:
        """
        Generates an initial prediction based on the broad, initial research.
        """
        prompt = clean_indents(
            f"""
            You are a world-class superforecaster tasked with generating a baseline prediction. Your methodology is grounded in empirical evidence and a deep awareness of cognitive biases.

            **Question:** {question.question_text}
            **Background:** {question.background_info}
            **Resolution Criteria:** {question.resolution_criteria}
            **Date of Forecast:** {datetime.now().strftime("%Y-%m-%d")}

            **Initial Research Dossier:**
            {initial_research}

            **Your Task:**
            Based *only* on the information provided above, produce an initial forecast.

            1.  **Triage & Deconstruct:** Is this question knowable? Break it down into its core, tractable components (a "Fermi-ization"). Distinguish between what is known, what is unknown, and your key assumptions.
            2.  **Establish the Outside View:** What is the base rate for events like this? Identify a relevant reference class and state the historical probability. This will be the initial anchor for your forecast.
            3.  **Incorporate the Inside View:** How do the specific details of this case alter the base rate? Analyze the provided research, weighing the evidence.
            4.  **Initial Synthesis & Forecast:** Integrate the outside and inside views to produce a precise, probabilistic forecast. Clearly state your reasoning, referencing specific points from the research.

            Conclude with your prediction in the required format.

            *   **For Binary Questions:** "Initial Probability: XX%"
            *   **For Numeric Questions:** Provide a five-point percentile distribution (10th, 25th, 50th, 75th, 90th).
            *   **For Multiple Choice Questions:** List each option with its assigned probability, ensuring they sum to 100%.
            """
        )

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
        prompt = clean_indents(
            f"""
            You are a skeptical "red team" analyst. Your sole purpose is to identify the flaws in a colleague's forecast to prevent catastrophic errors. Do not be agreeable. Your critique must be constructive and lead to actionable lines of inquiry.

            **Question:** {question.question_text}

            **Colleague's Initial Forecast & Rationale:**
            {initial_prediction_text}

            **Your Task:**
            Critique the forecast with extreme skepticism.

            1.  **Challenge the Premise:** What if the core assumption is wrong? Actively search for alternative interpretations of the evidence.
            2.  **Identify Biases:** Did the forecaster fall prey to confirmation bias, anchoring, or wishful thinking? Is the "inside view" overwhelming a more reliable "outside view"?
            3.  **Expose Knowledge Gaps:** What crucial information is missing? What remains uncertain?
            4.  **Generate Research Questions:** Conclude with a list of 2-4 specific, high-impact, and researchable questions. These questions should be designed to directly probe the weakest points of the initial forecast. They will form the basis for the next stage of intelligence gathering.
            """
        )

        critique_text = await self.get_llm("critique_llm", "llm").invoke(prompt)
        logger.info(f"Generated critique for URL {question.page_url}")
        return critique_text

    async def _perform_targeted_search(self, critique_text: str) -> str:
        """
        Performs a targeted search based on the questions raised in the critique.
        """
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
        questions_text = await self.get_llm("summarizer", "llm").invoke(extraction_prompt)

        if not questions_text.strip():
            logger.warning("No new research questions were generated from the critique.")
            return "No targeted search was performed as no new questions were identified."

        logger.info(f"Performing targeted search with queries: {questions_text.splitlines()}")

        sdk = AsyncAskNewsSDK(
          client_id=os.getenv("ASKNEWS_CLIENT_ID"),
          client_secret=os.getenv("ASKNEWS_SECRET"),)
        try:
            results = await sdk.news.search_news(query=questions_text, n_articles=5, strategy="news knowledge")
            return results.as_string if results.as_string is not None else "No results found."
        except Exception as e:
            logger.error(f"Targeted search failed with an error: {e}")
            return "An error occurred during the targeted search."

    async def _generate_refined_prediction(
        self,
        question: MetaculusQuestion,
        initial_research: str,
        initial_prediction_text: str,
        critique_text: str,
        targeted_research: str,
    ) -> str:
        if isinstance(question, MultipleChoiceQuestion):
            final_answer_format_instruction = f"""
                - For the multiple choice question, list each option with its probability. You MUST use the exact option text provided.
                  Example:
                  "0 or 1": 10%
                  "2 or 3": 70%
                  "4 or more": 20%
                  The options for this question are: {question.options}
            """
        elif isinstance(question, NumericQuestion):
            final_answer_format_instruction = """
                - For the numeric question, provide the requested percentiles. You MUST ensure the values are in strictly increasing order (10th percentile < 20th < 40th, etc.).
                  Example:
                  Percentile 10: 115
                  Percentile 20: 118
                  Percentile 40: 122
                  Percentile 60: 126
                  Percentile 80: 130
                  Percentile 90: 135
            """
        else: # BinaryQuestion
            final_answer_format_instruction = """
                - For the binary question: "Probability: ZZ%"
            """

        prompt = clean_indents(
            f"""
            You are a lead superforecaster producing the final, definitive forecast for your team. You must weigh all perspectives and evidence to arrive at the most accurate and well-calibrated prediction possible.

            **Question:** {question.question_text}
            **Background:** {question.background_info}
            **Resolution Criteria:** {question.resolution_criteria}
            **Date of Forecast:** {datetime.now().strftime("%Y-%m-%d")}

            **Complete Dossier:**
            1.  **Initial Research:** {initial_research}
            2.  **Initial Forecast:** {initial_prediction_text}
            3.  **Adversarial Critique:** {critique_text}
            4.  **Targeted Research Report:** {targeted_research}

            **Your Task:**
            Produce a final, comprehensive forecast by synthesizing the entire dossier.

            1.  **Update Your Beliefs:** Explicitly state how the critique and the targeted research have shifted your perspective from the initial forecast. Did the new information confirm, challenge, or refine your original view? By how much did you update your probabilities, and why? This demonstrates "perpetual beta."
            2.  **Final Rationale:** Provide a complete, final analysis that integrates the outside view, the inside view, and all new evidence. Acknowledge any remaining uncertainties and justify the level of confidence in your final prediction.
            3.  **Final Forecast:** Conclude with your final prediction in the precise format required: {final_answer_format_instruction}
            """
        )
        refined_prediction_text = await self.get_llm("refined_pred_llm", "llm").invoke(prompt)
        logger.info(f"Generated refined prediction for URL {question.page_url}")
        return refined_prediction_text

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Orchestrates the entire "self-critique" forecasting process.
        """
        async with self._concurrency_limiter:
            # STEP 1: Initial, broad research.
            initial_research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                sdk = AsyncAskNewsSDK(
                    client_id=os.getenv("ASKNEWS_CLIENT_ID"),
                    client_secret=os.getenv("ASKNEWS_SECRET"),
                )
                try:
                    logger.info(f"Performing comprehensive initial search for {question.page_url}")
                    results = await sdk.news.search_news(
                        query=question.question_text, n_articles=10, strategy="news knowledge"
                    )
                    initial_research = results.as_string if results.as_string is not None else "No results found."
                except Exception as e:
                    logger.error(f"Initial research for {question.page_url} failed: {e}", exc_info=True)
                    initial_research = f"An error occurred during initial research: {e}"
            else:
                logger.warning(f"No research provider found for URL {question.page_url}.")
            # STEP 2: Generate Initial Prediction
            initial_prediction_text = await self._generate_initial_prediction(question, initial_research)

            # STEP 3: Generate Adversarial Critique
            critique_text = await self._generate_adversarial_critique(question, initial_prediction_text)

            # STEP 4: Perform Targeted Search
            targeted_research = await self._perform_targeted_search(critique_text)

            # STEP 5: Generate Refined Prediction
            refined_prediction_text = await self._generate_refined_prediction(
                question, initial_research, initial_prediction_text, critique_text, targeted_research
            )

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
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            research, question.options
        )
        normalized_prediction = self._normalize_probabilities(prediction)
        logger.info(f"Extracted and normalized final prediction for URL {question.page_url}: {normalized_prediction}")
        return ReasonedPrediction(prediction_value=normalized_prediction, reasoning=research)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        dist = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            research, question
        )
        dist.declared_percentiles.sort(key=lambda p: p.percentile)
        logger.info(f"Extracted and sorted final prediction for URL {question.page_url}: {dist.declared_percentiles}")
        return ReasonedPrediction(prediction_value=dist, reasoning=research)

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
                timeout=80,
                allowed_tries=2,
                max_tokens=1024,
            ),
            "initial_pred_llm": GeneralLlm(
                model="metaculus/openai/gpt-4.1",
                temperature=0.3,
                timeout=80,
                allowed_tries=2,
                max_tokens=2048,
            ),
            "critique_llm": GeneralLlm(
                model="metaculus/anthropic/claude-3-7-sonnet-latest",
                temperature=1.0,
                timeout=80,
                allowed_tries=2,
                max_tokens=6144,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 4096,
                },
            ),
            "refined_pred_llm": GeneralLlm(
                model="metaculus/anthropic/claude-3-7-sonnet-latest",
                temperature=1.0,
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
                timeout=80,
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
  #          "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029        ]
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

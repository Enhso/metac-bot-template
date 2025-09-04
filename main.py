import argparse
import asyncio
import logging
import os
import traceback

from datetime import datetime
from typing import Literal, Sequence
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

from asknews_sdk import AsyncAskNewsSDK
from config.loader import load_bot_config, default_config_path

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

    async def _extract_keywords_for_search(self, text: str) -> str:
        """
        Uses a lightweight LLM to extract key entities and concepts for a targeted news search.
        """

        await asyncio.sleep(1)

        logger.info(f"Attempting to extract keywords from the following text: '{text}'")
        if not text or not text.strip():
            logger.warning("Keyword extraction was called with empty text. Skipping LLM call.")
            return "" # Return an empty string to be handled by the calling function

        prompt = clean_indents(
            f"""
            Your task is to extract critical search keywords from the user's text.
            Do not add any explanation, preamble, or formatting.
            Your entire response must be a single line of space-separated keywords.

            Here is an example:
            Text: "Will the US Federal Reserve raise interest rates in the next quarter of 2025 due to inflation concerns?"
            Keywords: US Federal Reserve interest rates 2025 inflation

            Now, perform this task on the following text:
            Text: "{text}"
            Keywords:
            """
        )
        try:
            keywords = await self.get_llm("keyword_extractor_llm", "llm").invoke(prompt)

            if not keywords or not keywords.strip():
                logger.warning(
                    f"Keyword extraction with model '{self.get_llm('keyword_extractor_llm', 'llm').model}' returned an empty string. Falling back to original text."
                )
                return text

            logger.info(f"Extracted keywords: {keywords.strip()}")
            return keywords.strip()

        except Exception as e:
            logger.error(f"Failed to extract keywords due to an API error: {e}. Falling back to original text.")
            return text
    async def _generate_initial_prediction(
        self, question: MetaculusQuestion, initial_research: str
    ) -> str:
        """
        Generates an initial prediction based on the broad, initial research.
        """
        prompt = clean_indents(
            f"""
            You are a superforecaster following the principles of Philip Tetlock. Your reasoning must be transparent, self-critical, and grounded in evidence. Your goal is to produce an initial, rigorously derived forecast.

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            ## Question:
            {question.question_text}

            ## Background:
            {question.background_info}

            ## Resolution Criteria:
            {question.resolution_criteria}

            ## Available Research:
            {initial_research}

            ## Your Task:
            Generate an initial forecast by following these four steps precisely.

            ### Step 1: Triage and Deconstruction (Fermi-ize)
            First, assess the question's tractability. Is it a "clock-like" or "cloud-like" problem? Then, break the core question down into smaller, more manageable, and quantifiable components. List these sub-questions.

            ### Step 2: Establish the Outside View
            Identify a suitable reference class for this event. What is the base rate of outcomes for similar situations? State the reference class and the resulting base rate probability. This will be your initial anchor.

            ### Step 3: Integrate the Inside View
            Now, analyze the unique, case-specific evidence from the provided research. How does this new information adjust your initial anchor from the outside view? Systematically discuss the evidence for and against, adjusting your probability estimate up or down. Mention any potential biases (e.g., availability bias, confirmation bias) that might be influencing your interpretation of the inside view.

            ### Step 4: Initial Forecast and Rationale Synthesis
            Synthesize your findings from the steps above into a coherent rationale. Clearly state your key assumptions and where your uncertainty lies. Conclude with your initial prediction in the precise format required.

            **Required Output Format:**
            **Step 1: Triage and Deconstruction**
            - [Your analysis of the question's tractability and your list of sub-questions]
            **Step 2: Outside View**
            - Reference Class: [Your identified reference class]
            - Base Rate/Anchor: [Your calculated base rate]
            **Step 3: Inside View**
            - [Your analysis adjusting the anchor based on case-specific evidence]
            **Step 4: Initial Forecast and Rationale Synthesis**
            - [Your synthesized rationale]
            - [Your final prediction in the required format]
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
            You are an intelligence analyst assigned to conduct a "red team" exercise. Your sole purpose is to challenge a colleague's forecast with constructive, aggressive skepticism. Do not be agreeable. Your goal is to expose every potential weakness.

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            ## The Original Question:
            {question.question_text}

            ## Colleague's Initial Forecast and Rationale:
            ---
            {initial_prediction_text}
            ---

            ## Your Task:
            Critique this forecast by addressing the following points:

            1.  **Challenge Core Assumptions:** Identify the 2-3 most critical stated or unstated assumptions in the initial forecast. Why might they be wrong?
            2.  **Propose an Alternative Perspective:** Actively consider the opposite conclusion. What key evidence or alternative interpretation was downplayed or missed entirely?
            3.  **Stress-Test the Outside View:** Was the chosen reference class appropriate? Propose at least one alternative reference class and explain how it might change the forecast.
            4.  **Generate High-Value Questions:** Conclude with a list of 2-3 specific, researchable questions. Each question should be self-contained and not refer to any external information. These questions should be designed to resolve the greatest points of uncertainty you've identified and have the highest potential to falsify the initial forecast.
            """
        )

        critique_text = await self.get_llm("critique_llm", "llm").invoke(prompt)
        logger.info(f"Generated critique for URL {question.page_url}")
        return critique_text

    async def _perform_targeted_search(self, critique_text: str) -> str:
        """
        Performs separate, targeted searches for each question raised in the critique
        and combines the results into a structured report.
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
        questions = [q for q in questions_text.splitlines() if q.strip()]

        if not questions:
            logger.warning("No new research questions were generated from the critique.")
            return "No targeted search was performed as no new questions were identified."

        sdk = AsyncAskNewsSDK(
            client_id=os.getenv("ASKNEWS_CLIENT_ID"),
            client_secret=os.getenv("ASKNEWS_SECRET"),
        )

        # This inner function will handle the search for a single question
        async def search_for_question(question: str) -> str:
            keywords = await self._extract_keywords_for_search(question)
            log_query = keywords if keywords != question else f"(fallback) {question}"
            logger.info(f"Performing targeted search for: {log_query}")

            try:
                # Perform a focused search with fewer articles
                results = await sdk.news.search_news(query=keywords, n_articles=3, strategy="news knowledge")
                search_results_string = results.as_string if results.as_string is not None else "No results found."

                # Format the output clearly, associating results with the original question
                return f"### Research for question: \"{question}\"\n\n{search_results_string}"
            except Exception as e:
                logger.error(f"Targeted search for '{question}' failed with an error: {e}")
                return f"### Research for question: \"{question}\"\n\nAn error occurred during the targeted search."

        individual_reports = []
        for question in questions:
            # 1. Perform a single search request and wait for it to complete.
            report = await search_for_question(question)
            individual_reports.append(report)

            # 2. Pause for 10 seconds before starting the next loop iteration.
            logger.info(f"AskNews rate limit: Pausing for 10 seconds after targeted search for question: \"{question[:50]}...\"")
            await asyncio.sleep(10)

        # Join the individual, formatted reports into a single string
        # This provides a clean, structured input for the final synthesis step
        combined_report = "\n\n---\n\n".join(individual_reports)

        logger.info("Completed all targeted searches.")
        return combined_report

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
                - For the multiple choice question, list each option with its probability. You MUST use the exact option text provided. All probabilities must be between 0.1% and 99.9% and sum to 1.0.
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
                - For the binary question: "Probability: ZZ%". All probabilities must be between 0.1% and 99.9%.
            """

        prompt = clean_indents(
            f"""
            You are a superforecaster producing a final, synthesized prediction. You have reviewed an initial analysis, a skeptical critique, and new targeted research. Your task is to integrate all evidence into a refined forecast, demonstrating intellectual humility and a commitment to "perpetual beta" (continuous improvement).

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            ## Question
            {question.question_text}

            ## Background
            {question.background_info}

            ## Resolution Criteria
            {question.resolution_criteria}

            ## Dossier
            ### 1. Initial Research
            {initial_research}
            ### 2. Initial Prediction (Thesis)
            {initial_prediction_text}
            ### 3. Adversarial Critique (Antithesis)
            {critique_text}
            ### 4. New, Targeted Research
            {targeted_research}

            ## Your Task:
            Follow this three-step process to generate your final analysis.

            ### Step 1: Synthesize Thesis, Antithesis, and New Evidence
            Adopt a "dragonfly eye" perspective. Explicitly discuss how the critique and the targeted research have altered your initial view. Which arguments from the initial forecast still hold, and which have been weakened or overturned? Weigh the conflicting points and synthesize them. Don't just discard one view for another; integrate them.

            ### Step 2: Final Rationale and Probabilistic Thinking
            Provide your final, comprehensive rationale. Explain how you are balancing the competing causal forces. Your reasoning should be granular, distinguishing between multiple degrees of uncertainty. Acknowledge what you still don't know and what key indicators could change your mind in the future.

            ### Step 3: Final Calibrated Prediction
            Conclude with your final prediction. Update your numerical forecast with precision, reflecting the synthesis above. Ensure it is in the precise format required.

            **Required Output Format:**
            **Step 1: Synthesis**
            - [Your discussion on how the critique and new data changed the forecast]
            **Step 2: Final Rationale**
            - [Your comprehensive final rationale and remaining uncertainties]
            **Step 3: Final Prediction**
            - [Your final prediction in the required format:
              {final_answer_format_instruction}]
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
            # Use the centralized CritiqueAndRefineStrategy for orchestration
            strategy = CritiqueAndRefineStrategy(self.get_llm, logger)

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
            prediction_value=prediction, reasoning=research
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            research, question.options
        )
        logger.info(f"Extracted final prediction for URL {question.page_url}: {prediction}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=research)

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
    PERSONAS = {
        "The Skeptic": clean_indents(
            """
            ## Your Persona: The Skeptic
            You are playing the role of a cautious, skeptical analyst. Your primary goal is to identify and prioritize risks, potential failure points, and reasons why the event will *not* happen. Challenge assumptions and focus on the downside.
            """
        ),
        "The Proponent": clean_indents(
            """
            ## Your Persona: The Proponent
            You are playing the role of a forward-looking, optimistic analyst. Your primary goal is to identify catalysts, opportunities, and the strongest arguments for why the event *will* happen. Focus on the upside potential and the driving forces for success.
            """
        ),
        "The Quant": clean_indents(
            """
            ## Your Persona: The Quant
            You are playing the role of a data-driven quantitative analyst. Your reasoning must be strictly grounded in the provided data, base rates, and statistical evidence. Ignore narrative, anecdotal evidence, and qualitative arguments. Focus only on the numbers.
            """
        ),
    }

    async def _initialize_notepad(self, question: MetaculusQuestion) -> Notepad:
        notepad = await super()._initialize_notepad(question)
        notepad.note_entries["personas"] = list(self.PERSONAS.keys())
        return notepad

    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """
        Orchestrates the ensemble forecasting process for a single question.
        This corrected version performs research once, then analyzes with each persona,
        and correctly uses the concurrency limiter.
        """
        async with self._concurrency_limiter:
            # --- RESEARCH PHASE (RUNS ONCE PER QUESTION) ---
            strategy = CritiqueAndRefineStrategy(self.get_llm, logger)
            initial_research = await strategy.initial_research(question)
            initial_prediction_text = await strategy.generate_initial_prediction(question, initial_research)
            critique_text = await strategy.generate_adversarial_critique(question, initial_prediction_text)
            targeted_research = await strategy.perform_targeted_search(critique_text)

            # --- ANALYSIS PHASE (RUNS ONCE PER PERSONA) ---
            persona_reports = []
            for persona_name in self.PERSONAS:
                persona_prompt = self.PERSONAS[persona_name]
                refined_prediction_text = await strategy.generate_refined_prediction(
                    question,
                    initial_research,
                    initial_prediction_text,
                    critique_text,
                    targeted_research,
                    persona_prompt=persona_prompt,
                )
                persona_reports.append((persona_name, refined_prediction_text))

            # --- SYNTHESIS PHASE (RUNS ONCE) ---
            reasoned_prediction = await self._synthesize_ensemble_forecasts(question, persona_reports)

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

            return final_report


    async def _get_initial_research(self, question: MetaculusQuestion) -> str:
        """Helper to consolidate initial research logic."""
        initial_research = ""
        if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
            sdk = AsyncAskNewsSDK(
                client_id=os.getenv("ASKNEWS_CLIENT_ID"),
                client_secret=os.getenv("ASKNEWS_SECRET"),
            )
            try:
                search_query = await self._extract_keywords_for_search(question.question_text)
                results = await sdk.news.search_news(query=search_query, n_articles=10, strategy="news knowledge")
                initial_research = results.as_string if results.as_string is not None else "No results found."
            except Exception as e:
                logger.error(f"Initial research for {question.page_url} failed: {e}", exc_info=True)
                initial_research = f"An error occurred during initial research: {e}"
        return initial_research

    def _get_final_answer_format_instruction(self, question: MetaculusQuestion) -> str:
        """Helper to get the correct final answer format instruction based on question type."""
        if isinstance(question, MultipleChoiceQuestion):
            return f"""
                - For the multiple choice question, list each option with its probability. You MUST use the exact option text provided. All probabilities must be between 0.1% and 99.9% and sum to 1.0.
                  Example:
                  "0 or 1": 10%
                  "2 or 3": 70%
                  "4 or more": 20%
                  The options for this question are: {question.options}
            """
        elif isinstance(question, NumericQuestion):
            return """
                - For the numeric question, provide the requested percentiles. You MUST ensure the values are in strictly increasing order (10th percentile < 20th < 40th, etc.).
                  Example:
                  Percentile 10: 115
                  Percentile 20: 118
                  Percentile 40: 122
                  Percentile 60: 126
                  Percentile 80: 130
                  Percentile 90: 135
            """
        else:  # BinaryQuestion
            return """
                - For the binary question: "Probability: ZZ%". All probabilities must be between 0.1% and 99.9%.
            """

    async def _generate_refined_prediction_with_persona(
        self,
        question: MetaculusQuestion,
        initial_research: str,
        initial_prediction_text: str,
        critique_text: str,
        targeted_research: str,
        persona_prompt: str,
    ) -> str:
        """
        Builds the prompt for a refined prediction and injects the persona.
        """
        final_answer_format_instruction = self._get_final_answer_format_instruction(question)

        prompt = clean_indents(
            f"""
            You are a superforecaster producing a final, synthesized prediction.

            {persona_prompt}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            ## Question
            {question.question_text}

            ## Background
            {question.background_info}

            ## Resolution Criteria
            {question.resolution_criteria}

            ## Dossier
            ### 1. Initial Research
            {initial_research}https://www.metaculus.com/questions/38880
            ### 2. Initial Prediction (Thesis)
            {initial_prediction_text}
            ### 3. Adversarial Critique (Antithesis)
            {critique_text}
            ### 4. New, Targeted Research
            {targeted_research}

            ## Your Task:
            Follow this three-step process to generate your final analysis.

            ### Step 1: Synthesize Thesis, Antithesis, and New Evidence
            Adopt a "dragonfly eye" perspective. Explicitly discuss how the critique and the targeted research have altered your initial view. Which arguments from the initial forecast still hold, and which have been weakened or overturned? Weigh the conflicting points and synthesize them. Don't just discard one view for another; integrate them.

            ### Step 2: Final Rationale and Probabilistic Thinking
            Provide your final, comprehensive rationale. Explain how you are balancing the competing causal forces. Your reasoning should be granular, distinguishing between multiple degrees of uncertainty. Acknowledge what you still don't know and what key indicators could change your mind in the future.

            ### Step 3: Final Calibrated Prediction
            Conclude with your final prediction. Update your numerical forecast with precision, reflecting the synthesis above. Ensure it is in the precise format required.

            **Required Output Format:**
            **Step 1: Synthesis**
            - [Your discussion on how the critique and new data changed the forecast]
            **Step 2: Final Rationale**
            - [Your comprehensive final rationale and remaining uncertainties]
            **Step 3: Final Prediction**
            - [Your final prediction in the required format:
              {final_answer_format_instruction}]
            """
        )

        return await self.get_llm("refined_pred_llm", "llm").invoke(prompt)

    async def _synthesize_ensemble_forecasts(
        self, question: MetaculusQuestion, persona_reports: list[tuple[str, str]]
    ) -> ReasonedPrediction:
        """
        Takes the reports from all personas and synthesizes them into a final reasoned prediction.
        """
        report_texts = []
        for name, report in persona_reports:
            report_texts.append(f"--- REPORT FROM {name.upper()} ---\n{report}\n--- END REPORT ---")

        combined_reports = "\n\n".join(report_texts)
        final_answer_format_instruction = self._get_final_answer_format_instruction(question)

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
        """Parses the final prediction from the synthesized reasoning in a type-safe way."""
        if isinstance(question, BinaryQuestion):
            return PredictionExtractor.extract_last_percentage_value(reasoning, max_prediction=1, min_prediction=0)
        elif isinstance(question, MultipleChoiceQuestion):
            return PredictionExtractor.extract_option_list_with_percentage_afterwards(reasoning, question.options)
        elif isinstance(question, NumericQuestion):
            dist = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(reasoning, question)
            dist.declared_percentiles.sort(key=lambda p: p.percentile)
            return dist
        raise TypeError(f"Unsupported question type for parsing: {type(question)}")

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

    # Load bot configuration and llms from YAML
    bot_cfg, llms = load_bot_config(args.config)

    # Construct the forecaster from externalized configuration
    bot_one = EnsembleForecaster(llms=llms, **bot_cfg)

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

import asyncio
import logging
import os
from datetime import datetime
from typing import Callable, Optional

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import (
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    clean_indents,
)


class CritiqueAndRefineStrategy:
    """
    A reusable strategy that encapsulates the forecasting flow:
    1) Initial research (broad)
    2) Initial prediction
    3) Adversarial critique
    4) Targeted search
    5) Refined prediction (optionally persona-guided)

    This class is intentionally framework-agnostic and depends only on:
    - a get_llm(name: str, kind: str) -> LLM interface with .invoke(prompt) coroutine
    - asyncio for pacing
    - AskNews SDK credentials from environment
    - a logger (optional)
    """

    def __init__(
        self,
        get_llm: Callable[[str, str], any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._get_llm = get_llm
        self._logger = logger or logging.getLogger(__name__)

    # ----------------------------- Public Orchestration -----------------------------
    async def initial_research(self, question: MetaculusQuestion) -> str:
        """Performs initial broad research using AskNews (if credentials available)."""
        initial_research = ""
        if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
            sdk = AsyncAskNewsSDK(
                client_id=os.getenv("ASKNEWS_CLIENT_ID"),
                client_secret=os.getenv("ASKNEWS_SECRET"),
            )
            try:
                search_query = await self.extract_keywords_for_search(question.question_text)
                self._logger.info(
                    f"Performing comprehensive initial search for '{question.page_url}' with query: '{search_query}'"
                )
                results = await sdk.news.search_news(
                    query=search_query, n_articles=10, strategy="news knowledge"
                )
                initial_research = (
                    results.as_string if results.as_string is not None else "No results found."
                )
                self._logger.info(
                    "AskNews rate limit: Pausing for 10 seconds after initial research call."
                )
                await asyncio.sleep(10)
            except Exception as e:
                self._logger.error(
                    f"Initial research for {question.page_url} failed: {e}", exc_info=True
                )
                initial_research = f"An error occurred during initial research: {e}"
        else:
            self._logger.warning(
                f"No research provider found for URL {question.page_url}."
            )
        return initial_research

    async def generate_initial_prediction(
        self, question: MetaculusQuestion, initial_research: str
    ) -> str:
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
        initial_prediction_text = await self._get_llm("initial_pred_llm", "llm").invoke(
            prompt
        )
        self._logger.info(
            f"Generated initial prediction for URL {question.page_url}"
        )
        return initial_prediction_text

    async def generate_adversarial_critique(
        self, question: MetaculusQuestion, initial_prediction_text: str
    ) -> str:
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
        critique_text = await self._get_llm("critique_llm", "llm").invoke(prompt)
        self._logger.info(f"Generated critique for URL {question.page_url}")
        return critique_text

    async def perform_targeted_search(self, critique_text: str) -> str:
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
        questions_text = await self._get_llm("summarizer", "llm").invoke(
            extraction_prompt
        )
        questions = [q for q in questions_text.splitlines() if q.strip()]

        if not questions:
            self._logger.warning(
                "No new research questions were generated from the critique."
            )
            return (
                "No targeted search was performed as no new questions were identified."
            )

        sdk = AsyncAskNewsSDK(
            client_id=os.getenv("ASKNEWS_CLIENT_ID"),
            client_secret=os.getenv("ASKNEWS_SECRET"),
        )

        async def search_for_question(question: str) -> str:
            keywords = await self.extract_keywords_for_search(question)
            log_query = keywords if keywords != question else f"(fallback) {question}"
            self._logger.info(f"Performing targeted search for: {log_query}")

            try:
                results = await sdk.news.search_news(
                    query=keywords, n_articles=3, strategy="news knowledge"
                )
                search_results_string = (
                    results.as_string if results.as_string is not None else "No results found."
                )
                return (
                    f'### Research for question: "{question}"\n\n{search_results_string}'
                )
            except Exception as e:
                self._logger.error(
                    f"Targeted search for '{question}' failed with an error: {e}"
                )
                return (
                    f'### Research for question: "{question}"\n\nAn error occurred during the targeted search.'
                )

        individual_reports = []
        for q in questions:
            report = await search_for_question(q)
            individual_reports.append(report)
            self._logger.info(
                "AskNews rate limit: Pausing for 10 seconds after targeted search."
            )
            await asyncio.sleep(10)

        combined_report = "\n\n---\n\n".join(individual_reports)
        self._logger.info("Completed all targeted searches.")
        return combined_report

    async def generate_refined_prediction(
        self,
        question: MetaculusQuestion,
        initial_research: str,
        initial_prediction_text: str,
        critique_text: str,
        targeted_research: str,
        persona_prompt: Optional[str] = None,
    ) -> str:
        final_answer_format_instruction = self._final_answer_format_instruction(question)

        persona_section = f"\n\n{persona_prompt}\n\n" if persona_prompt else "\n\n"

        prompt = clean_indents(
            f"""
            You are a superforecaster producing a final, synthesized prediction.{persona_section}

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

        refined_prediction_text = await self._get_llm("refined_pred_llm", "llm").invoke(
            prompt
        )
        self._logger.info(
            f"Generated refined prediction for URL {question.page_url}"
        )
        return refined_prediction_text

    # ----------------------------- Utilities -----------------------------
    async def extract_keywords_for_search(self, text: str) -> str:
        await asyncio.sleep(1)
        self._logger.info(
            f"Attempting to extract keywords from the following text: '{text}'"
        )
        if not text or not text.strip():
            self._logger.warning(
                "Keyword extraction was called with empty text. Skipping LLM call."
            )
            return ""

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
            keywords = await self._get_llm("keyword_extractor_llm", "llm").invoke(prompt)
            if not keywords or not keywords.strip():
                self._logger.warning(
                    "Keyword extraction model returned an empty string. Falling back to original text."
                )
                return text
            self._logger.info(f"Extracted keywords: {keywords.strip()}")
            return keywords.strip()
        except Exception as e:
            self._logger.error(
                f"Failed to extract keywords due to an API error: {e}. Falling back to original text."
            )
            return text

    def _final_answer_format_instruction(self, question: MetaculusQuestion) -> str:
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

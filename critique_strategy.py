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
)
from forecasting_prompts import (
    build_keyword_extractor_prompt,
    build_initial_prediction_prompt,
    build_adversarial_critique_prompt,
    build_extract_questions_from_critique_prompt,
    build_refined_prediction_prompt,
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
        prompt = build_initial_prediction_prompt(question, initial_research)
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
        prompt = build_adversarial_critique_prompt(question, initial_prediction_text)
        critique_text = await self._get_llm("critique_llm", "llm").invoke(prompt)
        self._logger.info(f"Generated critique for URL {question.page_url}")
        return critique_text

    async def perform_targeted_search(self, critique_text: str) -> str:
        extraction_prompt = build_extract_questions_from_critique_prompt(critique_text)
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

        prompt = build_refined_prediction_prompt(
            question=question,
            initial_research=initial_research,
            initial_prediction_text=initial_prediction_text,
            critique_text=critique_text,
            targeted_research=targeted_research,
            final_answer_format_instruction=final_answer_format_instruction,
            persona_prompt=persona_prompt,
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

        prompt = build_keyword_extractor_prompt(text)
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

    def get_final_answer_format_instruction(self, question: MetaculusQuestion) -> str:
        """Public method to get the final answer format instruction for a question."""
        return self._final_answer_format_instruction(question)

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

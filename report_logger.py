"""
Generic report logging utility for ForecastBot implementations.

This module provides a standardized way to log forecast reports with consistent
formatting while being flexible enough to handle different report structures.
"""

import logging
import traceback
from typing import Sequence

from forecasting_tools import ForecastReport, clean_indents

logger = logging.getLogger(__name__)


class ReportLogger:
    """
    A generic utility class for logging forecast reports with consistent formatting.
    
    This class can handle different report structures and provides a standardized
    logging format that can be used by any ForecastBot implementation.
    """
    
    @classmethod
    def log_forecast_summary(
        cls,
        forecast_reports: Sequence[ForecastReport | BaseException],
        bot_class_name: str,
        raise_errors: bool = True,
        use_structured_sections: bool = False,
        max_explanation_length: int = 10000,
    ) -> None:
        """
        Log a summary of forecast reports with consistent formatting.
        
        Args:
            forecast_reports: Sequence of ForecastReport objects or exceptions
            bot_class_name: Name of the bot class for identification
            raise_errors: Whether to raise errors if exceptions occurred
            use_structured_sections: If True, attempt to use structured sections
                                   (summary, first_rationale). If False, use raw explanation.
            max_explanation_length: Maximum length of explanation text to display
        """
        valid_reports = [
            report for report in forecast_reports if isinstance(report, ForecastReport)
        ]

        full_summary = "\n"
        full_summary += "-" * 100 + "\n"

        for report in valid_reports:
            if use_structured_sections:
                content = cls._get_structured_content(report, max_explanation_length)
            else:
                content = cls._get_raw_content(report, max_explanation_length)
            
            question_summary = clean_indents(
                f"""
                URL: {report.question.page_url}
                Errors: {report.errors}
                <<<<<<<<<<<<<<<<<<<< Report Content >>>>>>>>>>>>>>>>>>>>>
                {content}
                -------------------------------------------------------------------------------------------
            """
            )
            full_summary += question_summary + "\n"

        # Add bot identification and short summaries
        full_summary += f"Bot: {bot_class_name}\n"
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

        # Add statistics
        full_summary += cls._generate_statistics(valid_reports)
        full_summary += "-" * 100 + "\n\n\n"
        
        logger.info(full_summary)

        # Handle exceptions
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

    @classmethod
    def _get_structured_content(cls, report: ForecastReport, max_length: int) -> str:
        """
        Attempt to get structured content from report (summary and first_rationale).
        Falls back to raw explanation if structured sections are not available.
        """
        try:
            summary = report.summary
            first_rationale = report.first_rationale
            content = f"{summary}\n\n<<<<<<<<<<<<<<<<<<<< First Rationale >>>>>>>>>>>>>>>>>>>>>\n{first_rationale[:max_length]}"
        except Exception as e:
            # Fall back to raw explanation if structured sections fail
            content = f"Failed to get structured sections ({e}). Using raw explanation:\n{report.explanation[:max_length]}"
        return content

    @classmethod
    def _get_raw_content(cls, report: ForecastReport, max_length: int) -> str:
        """Get raw explanation content from report."""
        return report.explanation[:max_length]

    @classmethod
    def _generate_statistics(cls, valid_reports: list[ForecastReport]) -> str:
        """Generate statistics summary for the valid reports."""
        if not valid_reports:
            return "\nStats: No valid reports to analyze.\n"
        
        total_cost = sum(
            report.price_estimate if report.price_estimate else 0
            for report in valid_reports
        )
        average_minutes = (
            sum(
                report.minutes_taken if report.minutes_taken else 0
                for report in valid_reports
            )
            / len(valid_reports)
        )
        average_cost = total_cost / len(valid_reports)
        
        stats = "\nStats for passing reports:\n"
        stats += f"Total cost estimated: ${total_cost:.5f}\n"
        stats += f"Average cost per question: ${average_cost:.5f}\n"
        stats += f"Average time spent per question: {average_minutes:.4f} minutes\n"
        
        return stats

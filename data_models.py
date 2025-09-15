"""
Data models for the forecasting system.
"""

from dataclasses import dataclass
from forecasting_tools import MetaculusQuestion


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

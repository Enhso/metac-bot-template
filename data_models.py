"""
Data models for the forecasting system.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from forecasting_tools import MetaculusQuestion

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from contradictory_information_analyzer import ContradictionAnalysisResult


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


@dataclass
class BiasAnalysisResult:
    """
    Contains the results of cognitive bias analysis for a forecasting rationale.
    
    This structure captures the systematic bias detection and correction
    recommendations from the Cognitive Bias Red Team analysis.
    """
    question: MetaculusQuestion
    analyzed_rationale: str
    bias_analysis_text: str
    detected_biases: list[str]
    severity_assessment: str  # Low/Medium/High
    priority_corrections: list[str]
    confidence_adjustment_recommended: bool
    
    
@dataclass
class EnhancedResearchDossier(ResearchDossier):
    """
    Extended research dossier that includes cognitive bias analysis and contradiction analysis.
    
    This enhanced version supports the full bias-aware and contradiction-aware forecasting pipeline
    by including systematic bias detection, correction recommendations, and contradiction resolution.
    """
    bias_analysis: Optional[BiasAnalysisResult] = None
    contradiction_analysis: Optional["ContradictionAnalysisResult"] = None

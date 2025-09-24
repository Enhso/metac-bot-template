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
class VolatilityAnalysisResult:
    """
    Contains the results of volatility analysis for a forecasting question.
    
    This structure captures information volatility metrics and recommended 
    confidence adjustments based on news sentiment and volume analysis.
    """
    question: MetaculusQuestion
    analyzed_keywords: list[str]
    news_volume: int  # Number of relevant news articles found
    sentiment_volatility: float  # 0.0 (stable) to 1.0 (highly volatile)
    conflicting_reports_score: float  # 0.0 (no conflicts) to 1.0 (high conflicts)
    overall_volatility_score: float  # 0.0 (stable) to 1.0 (highly volatile)
    volatility_level: str  # "Low", "Medium", "High"
    confidence_adjustment_factor: float  # Multiplier for shrinking to midpoint (0.0 to 1.0)
    midpoint_shrinkage_amount: float  # How much to shrink towards 50%
    detailed_analysis: str  # LLM's detailed analysis


@dataclass
class EnhancedResearchDossier(ResearchDossier):
    """
    Extended research dossier that includes cognitive bias analysis, contradiction analysis,
    and volatility analysis.
    
    This enhanced version supports the full bias-aware, contradiction-aware, and volatility-adjusted
    forecasting pipeline by including systematic bias detection, correction recommendations,
    contradiction resolution, and information environment volatility assessment.
    """
    bias_analysis: Optional[BiasAnalysisResult] = None
    contradiction_analysis: Optional["ContradictionAnalysisResult"] = None
    volatility_analysis: Optional[VolatilityAnalysisResult] = None

"""
Configurable Dossier Generation Pipeline

This module implements a flexible, composition-based approach for generating research dossiers
with various analysis enhancements. Instead of using inheritance to layer analysis steps,
this system uses a pipeline of configurable analysis steps that can be composed dynamically.

The DossierGenerator class replaces the rigid inheritance chain of:
SelfCritiqueForecaster -> BiasAware -> ContradictionAware -> VolatilityAware

With a flexible pipeline where analysis steps can be configured and combined as needed.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any, Awaitable
from dataclasses import dataclass

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import MetaculusQuestion

from cognitive_bias_checker import CognitiveBiasChecker
from volatility_analyzer import VolatilityAnalyzer
from critique_strategy import CritiqueAndRefineStrategy
from data_models import EnhancedResearchDossier, BiasAnalysisResult, VolatilityAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class DossierGenerationConfig:
    """Configuration for dossier generation pipeline."""
    enable_bias_analysis: bool = False
    enable_contradiction_analysis: bool = False
    enable_volatility_analysis: bool = False


class AnalysisStep(ABC):
    """Base class for analysis steps in the dossier generation pipeline."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def analyze(
        self, 
        dossier: EnhancedResearchDossier,
        question: MetaculusQuestion,
        strategy: CritiqueAndRefineStrategy
    ) -> EnhancedResearchDossier:
        """
        Perform analysis and return an enhanced dossier with the analysis results.
        
        Args:
            dossier: The current research dossier (may already have other analysis)
            question: The MetaculusQuestion being analyzed
            strategy: The CritiqueAndRefineStrategy instance for LLM access
            
        Returns:
            Enhanced dossier with this analysis step's results added
        """
        pass


class BiasAnalysisStep(AnalysisStep):
    """Analysis step that adds cognitive bias detection and correction."""
    
    def __init__(self):
        super().__init__("BiasAnalysis")
    
    async def analyze(
        self, 
        dossier: EnhancedResearchDossier,
        question: MetaculusQuestion,
        strategy: CritiqueAndRefineStrategy
    ) -> EnhancedResearchDossier:
        """Add cognitive bias analysis to the dossier."""
        logger.info(f"Starting cognitive bias analysis for URL {question.page_url}")
        bias_start_time = time.time()
        
        # Analyze the initial prediction for cognitive biases
        bias_analysis_text = await strategy.perform_cognitive_bias_analysis(
            question=question,
            rationale_text=dossier.initial_prediction_text,
            reasoning_context=f"Initial Research Context:\n{dossier.initial_research}"
        )
        
        # Parse the bias analysis to extract structured information
        bias_result = CognitiveBiasChecker.parse_analysis_text(
            question=question,
            analyzed_rationale=dossier.initial_prediction_text,
            bias_analysis_text=bias_analysis_text
        )
        
        bias_time = time.time() - bias_start_time
        logger.info(f"Cognitive bias analysis completed in {bias_time:.2f}s for URL {question.page_url}")
        
        # Add bias analysis to the dossier
        dossier.bias_analysis = bias_result
        return dossier


class ContradictionAnalysisStep(AnalysisStep):
    """Analysis step that adds contradictory information detection and resolution."""
    
    def __init__(self):
        super().__init__("ContradictionAnalysis")
    
    async def analyze(
        self, 
        dossier: EnhancedResearchDossier,
        question: MetaculusQuestion,
        strategy: CritiqueAndRefineStrategy
    ) -> EnhancedResearchDossier:
        """Add contradictory information analysis to the dossier."""
        logger.info(f"Starting contradictory information analysis for URL {question.page_url}")
        contradiction_start_time = time.time()
        
        # Prepare research materials for contradiction analysis
        research_materials = {
            'initial_research': dossier.initial_research,
            'initial_prediction': dossier.initial_prediction_text,
            'critique_text': dossier.critique_text,
            'targeted_research': dossier.targeted_research
        }
        
        # Analyze for contradictory information using strategy's analyzer
        contradiction_analysis_result = await strategy._contradiction_analyzer.analyze_contradictory_information(
            question=question,
            research_materials=research_materials,
            context=f"Bias Analysis Context:\n{dossier.bias_analysis.bias_analysis_text if dossier.bias_analysis else 'No bias analysis available'}"
        )
        
        contradiction_time = time.time() - contradiction_start_time
        logger.info(f"Contradictory information analysis completed in {contradiction_time:.2f}s for URL {question.page_url}")
        
        # Add contradiction analysis to the dossier
        dossier.contradiction_analysis = contradiction_analysis_result
        return dossier


class VolatilityAnalysisStep(AnalysisStep):
    """Analysis step that adds information environment volatility analysis."""
    
    def __init__(self, get_llm: Callable[[str, str], Any], asknews_client: AsyncAskNewsSDK):
        super().__init__("VolatilityAnalysis")
        self._volatility_analyzer = VolatilityAnalyzer(
            get_llm=get_llm,
            asknews_client=asknews_client,
            logger=logger
        )
    
    async def analyze(
        self, 
        dossier: EnhancedResearchDossier,
        question: MetaculusQuestion,
        strategy: CritiqueAndRefineStrategy
    ) -> EnhancedResearchDossier:
        """Add volatility analysis to the dossier."""
        logger.info(f"Starting volatility analysis for URL {question.page_url}")
        volatility_start_time = time.time()
        
        try:
            volatility_analysis = await self._volatility_analyzer.analyze_information_volatility(question)
            dossier.volatility_analysis = volatility_analysis
            
            volatility_time = time.time() - volatility_start_time
            logger.info(f"Volatility analysis completed in {volatility_time:.2f}s for URL {question.page_url}. "
                       f"Level: {volatility_analysis.volatility_level}, "
                       f"Score: {volatility_analysis.overall_volatility_score:.3f}")
        
        except Exception as e:
            logger.error(f"Error during volatility analysis for URL {question.page_url}: {e}")
            # Create a default low-volatility result as fallback
            from data_models import VolatilityAnalysisResult
            dossier.volatility_analysis = VolatilityAnalysisResult(
                question=question,
                analyzed_keywords=[],
                news_volume=0,
                sentiment_volatility=0.0,
                conflicting_reports_score=0.0,
                overall_volatility_score=0.0,
                volatility_level="Low",
                confidence_adjustment_factor=1.0,
                midpoint_shrinkage_amount=0.0,
                detailed_analysis="Volatility analysis failed; assuming low volatility."
            )
        
        return dossier


class DossierGenerator:
    """
    Configurable pipeline for generating enhanced research dossiers.
    
    This class replaces the inheritance-based approach with a composition-based
    pipeline that can be configured to include different analysis steps.
    """
    
    def __init__(
        self,
        get_llm: Callable[[str, str], Any],
        asknews_client: AsyncAskNewsSDK,
        config: Optional[DossierGenerationConfig] = None
    ):
        """
        Initialize the dossier generator with the specified configuration.
        
        Args:
            get_llm: Function to get LLM instances
            asknews_client: AskNews client for news analysis
            config: Configuration specifying which analysis steps to include
        """
        self._get_llm = get_llm
        self._asknews_client = asknews_client
        self._config = config or DossierGenerationConfig()
        self._pipeline = self._build_analysis_pipeline()
    
    def _build_analysis_pipeline(self) -> List[AnalysisStep]:
        """Build the analysis pipeline based on configuration."""
        pipeline = []
        
        if self._config.enable_bias_analysis:
            pipeline.append(BiasAnalysisStep())
        
        if self._config.enable_contradiction_analysis:
            pipeline.append(ContradictionAnalysisStep())
        
        if self._config.enable_volatility_analysis:
            pipeline.append(VolatilityAnalysisStep(
                get_llm=self._get_llm,
                asknews_client=self._asknews_client
            ))
        
        return pipeline
    
    async def generate_research_dossier(
        self,
        question: MetaculusQuestion,
        base_dossier_generator: Callable[[MetaculusQuestion], Any]
    ) -> EnhancedResearchDossier:
        """
        Generate an enhanced research dossier using the configured analysis pipeline.
        
        Args:
            question: The MetaculusQuestion to analyze
            base_dossier_generator: Function that generates the base research dossier
            
        Returns:
            Enhanced research dossier with all configured analysis steps applied
        """
        logger.info(f"Starting dossier generation for URL {question.page_url} with {len(self._pipeline)} analysis steps")
        overall_start_time = time.time()
        
        # Generate the base research dossier
        base_dossier = await base_dossier_generator(question)
        
        # Convert to enhanced dossier format
        enhanced_dossier = EnhancedResearchDossier(
            question=base_dossier.question,
            initial_research=base_dossier.initial_research,
            initial_prediction_text=base_dossier.initial_prediction_text,
            critique_text=base_dossier.critique_text,
            targeted_research=base_dossier.targeted_research
        )
        
        # Create strategy for LLM access
        strategy = CritiqueAndRefineStrategy(
            lambda name, kind: self._get_llm(name, "llm"), 
            asknews_client=self._asknews_client,
            logger=logger
        )
        
        # Apply each analysis step in sequence
        for step in self._pipeline:
            step_start_time = time.time()
            enhanced_dossier = await step.analyze(enhanced_dossier, question, strategy)
            step_time = time.time() - step_start_time
            logger.info(f"{step.name} completed in {step_time:.2f}s for URL {question.page_url}")
        
        total_time = time.time() - overall_start_time
        logger.info(f"Enhanced dossier generation completed in {total_time:.2f}s for URL {question.page_url}")
        
        return enhanced_dossier
    
    def get_analysis_summary(self, enhanced_dossier: EnhancedResearchDossier) -> str:
        """
        Generate a summary of all analysis performed on the dossier.
        
        This method consolidates the analysis results into a human-readable summary
        that can be included in forecast explanations.
        """
        summary_parts = []
        
        if enhanced_dossier.bias_analysis and enhanced_dossier.bias_analysis.detected_biases:
            summary_parts.append(f"systematic cognitive bias analysis (detected: {', '.join(enhanced_dossier.bias_analysis.detected_biases)})")
        
        if enhanced_dossier.contradiction_analysis:
            contradiction_count = len(enhanced_dossier.contradiction_analysis.detected_contradictions)
            conflict_count = len(enhanced_dossier.contradiction_analysis.irresolvable_conflicts)
            if contradiction_count > 0:
                summary_parts.append(f"contradictory information analysis ({contradiction_count} contradictions detected, {conflict_count} irresolvable)")
        
        if enhanced_dossier.volatility_analysis:
            volatility_info = enhanced_dossier.volatility_analysis
            summary_parts.append(f"information volatility analysis ({volatility_info.volatility_level} volatility detected)")
        
        if summary_parts:
            return f"\n\n## Enhanced Analytical Process\nThis forecast has been enhanced with {' and '.join(summary_parts)}. These analyses improve forecast robustness and identify key uncertainties."
        
        return ""
    
    @classmethod
    def create_bias_aware_config(cls) -> DossierGenerationConfig:
        """Create configuration for bias-aware analysis only."""
        return DossierGenerationConfig(enable_bias_analysis=True)
    
    @classmethod
    def create_contradiction_aware_config(cls) -> DossierGenerationConfig:
        """Create configuration for bias and contradiction analysis."""
        return DossierGenerationConfig(
            enable_bias_analysis=True,
            enable_contradiction_analysis=True
        )
    
    @classmethod
    def create_full_analysis_config(cls) -> DossierGenerationConfig:
        """Create configuration for all available analysis steps."""
        return DossierGenerationConfig(
            enable_bias_analysis=True,
            enable_contradiction_analysis=True,
            enable_volatility_analysis=True
        )
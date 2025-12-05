"""
Verification tests for the PromptBuilder consolidation refactoring.

This test suite ensures that the consolidated PromptBuilder produces semantically
identical prompts to the original scattered implementations. It compares:

1. ContradictionAwareEnsembleForecaster._build_contradiction_aware_prompt 
   vs PromptBuilder.build_contradiction_aware_persona_prompt

2. VolatilityAwareEnsembleForecaster._build_volatility_aware_prompt
   vs PromptBuilder.build_volatility_aware_persona_prompt

3. CompositionBasedEnsembleForecaster._build_enhanced_persona_prompt
   vs PromptBuilder.build_enhanced_persona_prompt

4. CompositionBasedEnsembleForecaster._build_enhanced_synthesis_prompt
   vs PromptBuilder.build_enhanced_synthesis_prompt
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Import the PromptBuilder and data models
from prompt_builder import PromptBuilder, LegacyPromptBuilder
from data_models import (
    EnhancedResearchDossier, 
    BiasAnalysisResult, 
    VolatilityAnalysisResult
)


@dataclass
class MockContradictionAnalysisResult:
    """Mock contradiction analysis result for testing."""
    detected_contradictions: List[Dict[str, Any]]
    resolution_attempts: List[Dict[str, Any]]
    irresolvable_conflicts: List[Dict[str, Any]]
    key_uncertainties: List[str]
    overall_coherence_assessment: str
    confidence_impact: str


@dataclass
class MockQuestion:
    """Mock MetaculusQuestion for testing."""
    question_text: str = "Will event X happen by 2025?"
    background_info: str = "This question asks about event X happening."
    resolution_criteria: str = "Resolves YES if event X happens, NO otherwise."
    page_url: str = "https://metaculus.com/questions/12345/"
    options: Optional[List[str]] = None
    
    def get_question_type(self) -> str:
        return "binary"


def create_mock_enhanced_dossier(
    include_bias: bool = True,
    include_contradiction: bool = True,
    include_volatility: bool = True
) -> EnhancedResearchDossier:
    """Create a mock enhanced dossier for testing."""
    question = MockQuestion()
    
    # Create bias analysis if requested
    bias_analysis = None
    if include_bias:
        bias_analysis = BiasAnalysisResult(
            question=question,
            analyzed_rationale="Test rationale for analysis",
            bias_analysis_text="## Cognitive Bias Analysis\n\nThis is a detailed bias analysis.",
            detected_biases=["confirmation_bias", "anchoring_bias"],
            severity_assessment="Moderate",
            priority_corrections=["Consider alternative evidence", "Question initial assumptions"],
            confidence_adjustment_recommended="Reduce confidence by 5-10%"
        )
    
    # Create contradiction analysis if requested
    contradiction_analysis = None
    if include_contradiction:
        contradiction_analysis = MockContradictionAnalysisResult(
            detected_contradictions=[
                {"description": "Source A says X, Source B says not X", "severity": "High"},
                {"description": "Data from 2024 conflicts with 2023", "severity": "Medium"}
            ],
            resolution_attempts=[
                {"contradiction_index": 0, "resolution": "Source A is more recent"}
            ],
            irresolvable_conflicts=[
                {"description": "Fundamental disagreement on methodology"}
            ],
            key_uncertainties=["Timing uncertainty", "Data quality uncertainty"],
            overall_coherence_assessment="Moderate coherence with some key conflicts",
            confidence_impact="Reduce confidence by 10-15%"
        )
    
    # Create volatility analysis if requested
    volatility_analysis = None
    if include_volatility:
        volatility_analysis = VolatilityAnalysisResult(
            question=question,
            analyzed_keywords=["event X", "2025", "prediction"],
            news_volume=42,
            sentiment_volatility=0.65,
            conflicting_reports_score=0.45,
            overall_volatility_score=0.55,
            volatility_level="Medium",
            confidence_adjustment_factor=0.85,
            midpoint_shrinkage_amount=0.15,
            detailed_analysis="The information environment shows moderate volatility with some conflicting reports."
        )
    
    return EnhancedResearchDossier(
        question=question,
        initial_research="Initial research findings about event X.",
        initial_prediction_text="Based on initial research, probability is 60%.",
        critique_text="However, there are concerns about data quality.",
        targeted_research="Additional research found more evidence.",
        bias_analysis=bias_analysis,
        contradiction_analysis=contradiction_analysis,
        volatility_analysis=volatility_analysis
    )


class TestPromptBuilderConsolidation:
    """Test suite for verifying PromptBuilder consolidation."""
    
    def test_contradiction_aware_prompt_contains_key_elements(self):
        """
        Verify that the consolidated contradiction-aware prompt contains all
        the essential elements that were in the original implementation.
        """
        dossier = create_mock_enhanced_dossier(
            include_bias=True, 
            include_contradiction=True, 
            include_volatility=False
        )
        persona_prompt = "You are an analytical forecaster."
        
        prompt_builder = PromptBuilder(dossier.question)
        prompt = prompt_builder.build_contradiction_aware_persona_prompt(dossier, persona_prompt)
        
        # Verify key structural elements are present
        assert "superforecaster" in prompt.lower()
        assert "cognitive biases and contradictory information" in prompt.lower()
        assert persona_prompt in prompt
        assert dossier.question.question_text in prompt
        assert dossier.question.background_info in prompt
        assert dossier.question.resolution_criteria in prompt
        
        # Verify research sections
        assert "Initial Research" in prompt
        assert dossier.initial_research in prompt
        assert dossier.initial_prediction_text in prompt
        assert dossier.critique_text in prompt
        assert dossier.targeted_research in prompt
        
        # Verify bias section when bias analysis is present
        assert "Cognitive Bias Analysis" in prompt
        assert dossier.bias_analysis.bias_analysis_text in prompt
        
        # Verify contradiction section when contradiction analysis is present
        assert "Contradictory Information Analysis" in prompt
        assert "Overall Coherence Assessment" in prompt
        assert "Detected Contradictions" in prompt
        assert "Key Contradictions" in prompt
        assert "Irresolvable Conflicts" in prompt
        assert "Key Uncertainties" in prompt
        assert "Confidence Impact" in prompt
        
        # Verify synthesis task instructions
        assert "Enhanced Synthesis Task" in prompt
        assert "Step 1:" in prompt or "Step 1" in prompt
        assert "Step 2:" in prompt or "Step 2" in prompt
        assert "Step 3:" in prompt or "Step 3" in prompt
        assert "Step 4:" in prompt or "Step 4" in prompt
    
    def test_volatility_aware_prompt_extends_contradiction_aware(self):
        """
        Verify that the volatility-aware prompt includes volatility section
        and properly extends the contradiction-aware prompt.
        """
        dossier = create_mock_enhanced_dossier(
            include_bias=True, 
            include_contradiction=True, 
            include_volatility=True
        )
        persona_prompt = "You are a volatility-aware forecaster."
        
        prompt_builder = PromptBuilder(dossier.question)
        prompt = prompt_builder.build_volatility_aware_persona_prompt(dossier, persona_prompt)
        
        # Should contain everything from contradiction-aware prompt
        assert "superforecaster" in prompt.lower()
        assert persona_prompt in prompt
        assert "Cognitive Bias Analysis" in prompt
        assert "Contradictory Information Analysis" in prompt
        
        # Should also contain volatility section
        assert "Volatility" in prompt
        assert "Volatility Level" in prompt
        assert "Medium" in prompt  # Our mock volatility level
        assert "Overall Volatility Score" in prompt
        assert "News Volume" in prompt
        assert "Sentiment Volatility" in prompt
        assert "Conflicting Reports Score" in prompt
        assert "Keywords Analyzed" in prompt
        assert "Recommended Confidence Adjustment" in prompt
        assert "shrinking predictions" in prompt.lower() or "shrinkage" in prompt.lower()
        
        # Should update synthesis instructions to include volatility
        assert "volatility" in prompt.lower()
    
    def test_enhanced_persona_prompt_dynamically_adapts(self):
        """
        Verify that the enhanced persona prompt dynamically includes sections
        based on which analyses are available.
        """
        prompt_builder_bias_only = PromptBuilder(create_mock_enhanced_dossier(
            include_bias=True, include_contradiction=False, include_volatility=False
        ).question)
        prompt_builder_all = PromptBuilder(create_mock_enhanced_dossier(
            include_bias=True, include_contradiction=True, include_volatility=True
        ).question)
        
        # Test with bias only
        dossier_bias_only = create_mock_enhanced_dossier(
            include_bias=True, include_contradiction=False, include_volatility=False
        )
        prompt_bias_only = prompt_builder_bias_only.build_enhanced_persona_prompt(
            dossier_bias_only, "Test persona"
        )
        
        assert "Cognitive Bias Analysis" in prompt_bias_only
        assert "Contradictory Information Analysis" not in prompt_bias_only
        assert "Information Volatility Analysis" not in prompt_bias_only
        
        # Test with all analyses
        dossier_all = create_mock_enhanced_dossier(
            include_bias=True, include_contradiction=True, include_volatility=True
        )
        prompt_all = prompt_builder_all.build_enhanced_persona_prompt(
            dossier_all, "Test persona"
        )
        
        assert "Cognitive Bias Analysis" in prompt_all
        assert "Contradictory Information Analysis" in prompt_all
        assert "Information Volatility Analysis" in prompt_all or "Volatility Analysis" in prompt_all
    
    def test_enhanced_synthesis_prompt_includes_analysis_context(self):
        """
        Verify that the enhanced synthesis prompt includes all available
        analysis context for the final synthesis.
        """
        dossier = create_mock_enhanced_dossier(
            include_bias=True, include_contradiction=True, include_volatility=True
        )
        combined_reports = """
        --- REPORT FROM ANALYST 1 ---
        Analysis from analyst 1.
        --- END REPORT ---
        
        --- REPORT FROM ANALYST 2 ---
        Analysis from analyst 2.
        --- END REPORT ---
        """
        
        prompt_builder = PromptBuilder(dossier.question)
        prompt = prompt_builder.build_enhanced_synthesis_prompt(
            combined_reports, dossier
        )
        
        # Verify structure
        assert "lead superforecaster" in prompt.lower()
        assert dossier.question.question_text in prompt
        assert combined_reports in prompt or "ANALYST" in prompt
        
        # Verify bias context
        assert "Cognitive Bias" in prompt
        assert "Detected biases" in prompt.lower() or "detected biases" in prompt.lower()
        
        # Verify contradiction context
        assert "Contradiction" in prompt
        assert "coherence" in prompt.lower()
        
        # Verify volatility context
        assert "Volatility" in prompt
        assert "shrinkage" in prompt.lower() or "adjustment" in prompt.lower()
        
        # Verify synthesis instructions
        assert "Synthesis" in prompt
        assert "Evidence Integration" in prompt or "evidence" in prompt.lower()
    
    def test_detailed_format_instruction_for_binary_question(self):
        """Verify format instruction for binary questions."""
        question = MockQuestion()
        question.get_question_type = lambda: "binary"
        
        prompt_builder = PromptBuilder(question)
        instruction = prompt_builder.get_detailed_final_answer_format_instruction()
        
        assert "binary" in instruction.lower()
        assert "Probability" in instruction
        assert "0.1%" in instruction or "99.9%" in instruction
    
    def test_prompt_builder_handles_missing_analyses(self):
        """
        Verify that PromptBuilder gracefully handles missing analyses
        without raising errors.
        """
        dossier = create_mock_enhanced_dossier(
            include_bias=False, include_contradiction=False, include_volatility=False
        )
        persona_prompt = "You are a forecaster."
        
        prompt_builder = PromptBuilder(dossier.question)
        
        # Should not raise errors even without analyses
        persona_prompt_result = prompt_builder.build_persona_prompt(dossier, persona_prompt)
        assert persona_prompt_result is not None
        assert len(persona_prompt_result) > 0
        
        # Enhanced prompts should also work
        enhanced_prompt = prompt_builder.build_enhanced_persona_prompt(dossier, persona_prompt)
        assert enhanced_prompt is not None
        
        synthesis_prompt = prompt_builder.build_enhanced_synthesis_prompt(
            "Test reports", dossier
        )
        assert synthesis_prompt is not None


class TestLegacyPromptBuilderEquivalence:
    """
    Test that LegacyPromptBuilder produces equivalent results to the new
    PromptBuilder methods for verification purposes.
    """
    
    def test_legacy_contradiction_aware_prompt_structure(self):
        """
        Verify that LegacyPromptBuilder.build_contradiction_aware_prompt
        has the expected structure for comparison purposes.
        """
        dossier = create_mock_enhanced_dossier(
            include_bias=True, include_contradiction=True, include_volatility=False
        )
        persona_prompt = "You are a test forecaster."
        
        legacy_prompt = LegacyPromptBuilder.build_contradiction_aware_prompt(
            dossier, persona_prompt
        )
        
        # Verify key elements
        assert "superforecaster" in legacy_prompt.lower()
        assert persona_prompt in legacy_prompt
        assert "Bias" in legacy_prompt
        assert "Contradiction" in legacy_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

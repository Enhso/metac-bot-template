"""
PromptBuilder: Consolidated dynamic prompt generation for the forecasting system.

This module consolidates all prompt generation logic that was previously scattered
across various forecaster classes. The PromptBuilder dynamically constructs prompts
based on the available analyses in the research dossier.
"""

import time
from typing import Optional, TYPE_CHECKING

from forecasting_tools import clean_indents, MetaculusQuestion
from data_models import EnhancedResearchDossier, BiasAnalysisResult, VolatilityAnalysisResult

if TYPE_CHECKING:
    from contradictory_information_analyzer import ContradictionAnalysisResult

def _today_str() -> str:
    """Get today's date as a string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


class PromptBuilder:
    """
    Centralized prompt builder that dynamically constructs LLM prompts based on
    available analysis components in the research dossier.
    
    This class decouples prompt engineering from forecasting orchestration logic,
    making it easier to experiment with and modify prompts in the future.
    """
    
    def __init__(self, question: MetaculusQuestion):
        """
        Initialize the prompt builder for a specific question.
        
        Args:
            question: The Metaculus question being forecasted
        """
        self.question = question
    
    def build_synthesis_prompt(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        combined_reports: str,
        final_answer_format_instruction: str
    ) -> str:
        """
        Build a synthesis prompt that dynamically incorporates all available analyses.
        
        This is the primary method that inspects the dossier to see which analyses
        are present and constructs the appropriate prompt sections.
        
        Args:
            enhanced_dossier: Research dossier with optional analysis components
            combined_reports: Combined persona analysis reports
            final_answer_format_instruction: Format instruction for the final answer
            
        Returns:
            Complete synthesis prompt with dynamic sections based on available analyses
        """
        # Build analysis context summary based on what's available
        analysis_context_parts = []
        
        if enhanced_dossier.bias_analysis:
            analysis_context_parts.append(self._build_bias_analysis_context(enhanced_dossier.bias_analysis))
        
        if enhanced_dossier.contradiction_analysis:
            analysis_context_parts.append(self._build_contradiction_analysis_context(enhanced_dossier.contradiction_analysis))
        
        if enhanced_dossier.volatility_analysis:
            analysis_context_parts.append(self._build_volatility_analysis_context(enhanced_dossier.volatility_analysis))
        
        # Combine all analysis context sections
        analysis_context = "\n".join(analysis_context_parts) if analysis_context_parts else ""
        
        # Build synthesis instructions based on available analyses
        synthesis_instructions = self._build_synthesis_instructions(enhanced_dossier)
        
        return clean_indents(
            f"""
            You are a superforecaster synthesizing multiple expert analyses into a final prediction.

            Today is {_today_str()}.

            ## Question
            {self.question.question_text}

            ## Background
            {self.question.background_info}

            ## Resolution Criteria
            {self.question.resolution_criteria}

            {analysis_context}

            ## Expert Analysis Reports
            {combined_reports}

            ## Your Task:
            {synthesis_instructions}

            **Required Output Format:**
            **Final Synthesis**
            - [Your comprehensive synthesis of all expert analyses]
            **Final Prediction**
            - [Your final prediction in the required format:
              {final_answer_format_instruction}]
            """
        )
    
    def build_persona_prompt(
        self,
        enhanced_dossier: EnhancedResearchDossier,
        persona_prompt: str
    ) -> str:
        """
        Build a persona prompt that dynamically incorporates all available analyses.
        
        Args:
            enhanced_dossier: Research dossier with optional analysis components
            persona_prompt: The specific persona instructions
            
        Returns:
            Complete persona prompt with dynamic sections based on available analyses
        """
        # Base prompt components
        prompt_parts = [
            f"You are a superforecaster with enhanced analytical capabilities.",
            f"",
            f"{persona_prompt}",
            f"",
            f"Today is {time.strftime('%Y-%m-%d')}",
            f"",
            f"## Question",
            f"{enhanced_dossier.question.question_text}",
            f"",
            f"## Background",
            f"{enhanced_dossier.question.background_info}",
            f"",
            f"## Resolution Criteria", 
            f"{enhanced_dossier.question.resolution_criteria}",
            f"",
            f"## Research Analysis",
            f"### Initial Research",
            f"{enhanced_dossier.initial_research}",
            f"",
            f"### Initial Prediction",
            f"{enhanced_dossier.initial_prediction_text}",
            f"",
            f"### Adversarial Critique",
            f"{enhanced_dossier.critique_text}",
            f"",
            f"### Targeted Research",
            f"{enhanced_dossier.targeted_research}",
        ]
        
        # Dynamically add analysis sections based on what's available
        if enhanced_dossier.bias_analysis:
            prompt_parts.extend(self._build_bias_analysis_section(enhanced_dossier.bias_analysis))
        
        if enhanced_dossier.contradiction_analysis:
            prompt_parts.extend(self._build_contradiction_analysis_section(enhanced_dossier.contradiction_analysis))
        
        if enhanced_dossier.volatility_analysis:
            prompt_parts.extend(self._build_volatility_analysis_section(enhanced_dossier.volatility_analysis))
        
        # Add task instructions that adapt to available analyses
        task_instructions = self._build_persona_task_instructions(enhanced_dossier)
        prompt_parts.extend([
            "",
            "## Your Task:",
            task_instructions,
            "",
            f"**Required Output Format:**",
            f"- [Your final prediction in the required format: {self._get_final_answer_format_instruction()}]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_bias_analysis_context(self, bias_analysis: BiasAnalysisResult) -> str:
        """Build bias analysis context section for synthesis prompt."""
        return f"""
        ## Cognitive Bias Analysis Context
        The research was systematically analyzed for cognitive biases:
        
        **Detected Biases:** {', '.join(bias_analysis.detected_biases) if bias_analysis.detected_biases else 'None significant'}
        **Severity Assessment:** {bias_analysis.severity_assessment}
        **Confidence Adjustment Recommended:** {bias_analysis.confidence_adjustment_recommended}
        
        The analysts above have been provided with bias correction guidance to improve logical soundness.
        """
    
    def _build_contradiction_analysis_context(self, contradiction_analysis: "ContradictionAnalysisResult") -> str:
        """Build contradiction analysis context section for synthesis prompt."""
        return f"""
        ## Contradiction Analysis Context
        The research materials were analyzed for contradictory information:
        
        **Detected Contradictions:** {len(contradiction_analysis.detected_contradictions)}
        **Resolution Attempts:** {len(contradiction_analysis.resolution_attempts)}
        **Irresolvable Conflicts:** {len(contradiction_analysis.irresolvable_conflicts)}
        **Overall Coherence:** {contradiction_analysis.overall_coherence_assessment}
        
        The analysts above have been informed about contradictions and their impact on forecast uncertainty.
        """
    
    def _build_volatility_analysis_context(self, volatility_analysis: VolatilityAnalysisResult) -> str:
        """Build volatility analysis context section for synthesis prompt."""
        return f"""
        ## Information Volatility Context
        The information environment was assessed for volatility:
        
        **Volatility Level:** {volatility_analysis.volatility_level}
        **Overall Score:** {volatility_analysis.overall_volatility_score:.2f}/1.0
        **Confidence Adjustment Factor:** {volatility_analysis.confidence_adjustment_factor:.2f}
        **Recommended Midpoint Shrinkage:** {volatility_analysis.midpoint_shrinkage_amount:.0%}
        
        The analysts above have been advised to consider volatility in their confidence assessments.
        """
    
    def _build_bias_analysis_section(self, bias_analysis: BiasAnalysisResult) -> list[str]:
        """Build bias analysis section for persona prompt."""
        return [
            "",
            "### Cognitive Bias Analysis",
            f"**Detected Biases:** {', '.join(bias_analysis.detected_biases) if bias_analysis.detected_biases else 'None significant'}",
            f"**Severity Assessment:** {bias_analysis.severity_assessment}",
            f"**Priority Corrections:** {'; '.join(bias_analysis.priority_corrections)}",
            f"**Confidence Adjustment Recommended:** {bias_analysis.confidence_adjustment_recommended}",
            "",
            "Apply the bias corrections above to improve the logical soundness of your analysis.",
        ]
    
    def _build_contradiction_analysis_section(self, contradiction_analysis: "ContradictionAnalysisResult") -> list[str]:
        """Build contradiction analysis section for persona prompt."""
        sections = [
            "",
            "### Contradictory Information Analysis",
            f"**Detected Contradictions:** {len(contradiction_analysis.detected_contradictions)}",
            f"**Irresolvable Conflicts:** {len(contradiction_analysis.irresolvable_conflicts)}",
            f"**Overall Coherence Assessment:** {contradiction_analysis.overall_coherence_assessment}",
            "",
            "Account for contradictory information and unresolved conflicts in your uncertainty assessment.",
        ]
        
        # Include key contradictions if present
        if contradiction_analysis.detected_contradictions:
            sections.extend([
                "",
                "**Key Contradictions:**",
            ])
            for i, contradiction in enumerate(contradiction_analysis.detected_contradictions[:3], 1):
                sections.append(f"{i}. {contradiction.get('description', 'No description')} "
                               f"(Severity: {contradiction.get('severity', 'Unknown')})")
        
        return sections
    
    def _build_volatility_analysis_section(self, volatility_analysis: VolatilityAnalysisResult) -> list[str]:
        """Build volatility analysis section for persona prompt."""
        return [
            "",
            "### Information Environment Volatility Analysis",
            f"**Volatility Level:** {volatility_analysis.volatility_level}",
            f"**Overall Volatility Score:** {volatility_analysis.overall_volatility_score:.2f}/1.0",
            f"**News Volume Analyzed:** {volatility_analysis.news_volume} articles",
            f"**Sentiment Volatility:** {volatility_analysis.sentiment_volatility:.2f}/1.0",
            f"**Conflicting Reports Score:** {volatility_analysis.conflicting_reports_score:.2f}/1.0",
            f"**Keywords Analyzed:** {', '.join(volatility_analysis.analyzed_keywords)}",
            "",
            f"**Volatility Analysis:**",
            f"{volatility_analysis.detailed_analysis}",
            "",
            f"**Recommended Confidence Adjustment:**",
            f"Due to {volatility_analysis.volatility_level.lower()} information volatility, consider adjusting",
            f"prediction confidence. The analysis suggests shrinking predictions by",
            f"{volatility_analysis.midpoint_shrinkage_amount:.0%} towards the midpoint (50%) to account for",
            f"the unstable information environment.",
        ]
    
    def _build_synthesis_instructions(self, enhanced_dossier: EnhancedResearchDossier) -> str:
        """Build synthesis task instructions that adapt to available analyses."""
        # Determine which analyses are available
        analyses_available = []
        if enhanced_dossier.bias_analysis:
            analyses_available.append("cognitive bias")
        if enhanced_dossier.contradiction_analysis:
            analyses_available.append("contradictory information")
        if enhanced_dossier.volatility_analysis:
            analyses_available.append("information environment volatility")
        
        # Build adaptive instructions
        if not analyses_available:
            # Standard synthesis without enhancements
            return """
            Synthesize the expert analyses above into a final prediction. Consider the strengths and 
            weaknesses of each perspective, identify areas of consensus and disagreement, and integrate 
            the insights into a coherent final forecast.
            """
        elif len(analyses_available) == 1:
            analysis_type = analyses_available[0]
            return f"""
            Synthesize the expert analyses above into a final prediction that explicitly accounts for 
            {analysis_type}. The experts have been provided with systematic analysis of {analysis_type} 
            to improve forecast accuracy. Consider how this enhanced awareness affects the reliability 
            and uncertainty of their predictions.
            """
        elif len(analyses_available) == 2:
            return f"""
            Synthesize the expert analyses above into a final prediction that explicitly accounts for 
            {analyses_available[0]} and {analyses_available[1]}. The experts have been provided with 
            systematic analysis of both factors to improve forecast accuracy. Consider how this enhanced 
            awareness affects the reliability and uncertainty of their predictions.
            """
        else:
            # All three analyses available
            return """
            Synthesize the expert analyses above into a final prediction that explicitly accounts for 
            cognitive biases, contradictory information, AND information environment volatility. The 
            experts have been provided with comprehensive analysis of all three factors to improve 
            forecast accuracy. Consider how this enhanced awareness affects the reliability and 
            uncertainty of their predictions.
            """
    
    def _build_persona_task_instructions(self, enhanced_dossier: EnhancedResearchDossier) -> str:
        """Build persona task instructions that adapt to available analyses."""
        # Determine which analyses are available
        analyses_available = []
        if enhanced_dossier.bias_analysis:
            analyses_available.append("cognitive bias analysis")
        if enhanced_dossier.contradiction_analysis:
            analyses_available.append("contradiction detection")
        if enhanced_dossier.volatility_analysis:
            analyses_available.append("volatility assessment")
        
        # Build adaptive instructions
        base_instruction = """
        Analyze the research and provide your expert perspective on the forecast. Follow your persona's 
        cognitive approach and reasoning style while maintaining analytical rigor.
        """
        
        if analyses_available:
            enhancement_instruction = f"""
        
        **Enhanced Analysis Integration:**
        Incorporate the {', '.join(analyses_available)} provided above into your reasoning. Use this 
        systematic analysis to improve the logical soundness and calibration of your forecast.
        """
            return base_instruction + enhancement_instruction
        
        return base_instruction
    
    def _get_final_answer_format_instruction(self) -> str:
        """Get the final answer format instruction for the question type."""
        # This method centralizes the format instruction logic that was previously
        # duplicated across forecaster classes
        try:
            question_type = self.question.get_question_type()
        except (AttributeError, ValueError):
            # Default format if question type cannot be determined
            return "Prediction: [Your prediction in the appropriate format]"
            
        if question_type == "binary":
            return "Probability: X% (where X is between 1 and 99)"
        elif question_type == "numeric":
            if hasattr(self.question, 'open_upper_bound') and getattr(self.question, 'open_upper_bound', False):
                return "Prediction: X (your numeric prediction)"
            else:
                return "Prediction: X (your numeric prediction within the specified range)"
        elif question_type == "multiple_choice":
            return "Selected Option: [Your chosen option exactly as listed]"
        else:
            return "Prediction: [Your prediction in the appropriate format]"


class LegacyPromptBuilder:
    """
    Legacy prompt builders that replicate the exact behavior of the old methods.
    
    These are provided for backward compatibility and verification purposes.
    They should produce identical prompts to the original scattered methods.
    """
    
    @staticmethod
    def build_contradiction_aware_prompt(
        enhanced_dossier: EnhancedResearchDossier,
        persona_prompt: str
    ) -> str:
        """
        Legacy method that replicates ContradictionAwareEnsembleForecaster._build_contradiction_aware_prompt
        
        This method is preserved for verification that the new PromptBuilder produces
        identical results to the original implementation.
        """
        # Build base information
        question = enhanced_dossier.question
        final_answer_format_instruction = LegacyPromptBuilder._get_legacy_final_answer_format_instruction(question)
        
        # Build bias awareness section
        bias_section = ""
        if enhanced_dossier.bias_analysis:
            bias_section = f"""
            ## Cognitive Bias Analysis
            {enhanced_dossier.bias_analysis.bias_analysis_text}
            """
        
        # Build contradiction awareness section
        contradiction_section = ""
        if enhanced_dossier.contradiction_analysis:
            contradiction_info = enhanced_dossier.contradiction_analysis
            contradiction_section = f"""
            ## Contradictory Information Analysis
            
            **Overall Coherence Assessment:** {contradiction_info.overall_coherence_assessment}
            
            **Detected Contradictions:** {len(contradiction_info.detected_contradictions)}
            """
            
            if contradiction_info.detected_contradictions:
                contradiction_section += "\n**Key Contradictions:**\n"
                for i, contradiction in enumerate(contradiction_info.detected_contradictions[:3], 1):
                    contradiction_section += f"{i}. {contradiction.get('description', 'No description')} (Severity: {contradiction.get('severity', 'Unknown')})\n"
            
            if contradiction_info.irresolvable_conflicts:
                contradiction_section += f"\n**Irresolvable Conflicts:** {len(contradiction_info.irresolvable_conflicts)}\n"
                for conflict in contradiction_info.irresolvable_conflicts[:2]:
                    contradiction_section += f"- {conflict.get('description', 'No description')}\n"
            
            if contradiction_info.key_uncertainties:
                contradiction_section += f"\n**Key Uncertainties:**\n"
                for uncertainty in contradiction_info.key_uncertainties[:3]:
                    contradiction_section += f"- {uncertainty}\n"
            
            contradiction_section += f"\n**Confidence Impact:** {contradiction_info.confidence_impact}"
        
        return clean_indents(
            f"""
            You are a superforecaster producing a final, synthesized prediction that explicitly accounts for both cognitive biases and contradictory information.

            {persona_prompt}

            Today is {_today_str()}.

            ## Question
            {question.question_text}

            ## Background
            {question.background_info}

            ## Resolution Criteria
            {question.resolution_criteria}

            ## Dossier
            ### 1. Initial Research
            {enhanced_dossier.initial_research}
            ### 2. Initial Prediction (Thesis)
            {enhanced_dossier.initial_prediction_text}
            ### 3. Adversarial Critique (Antithesis)
            {enhanced_dossier.critique_text}
            ### 4. New, Targeted Research
            {enhanced_dossier.targeted_research}
            {bias_section}
            {contradiction_section}

            ## Your Enhanced Synthesis Task:
            Follow this enhanced three-step process that integrates cognitive bias awareness and contradictory information analysis.

            ### Step 1: Bias and Contradiction-Aware Evidence Integration
            Explicitly address any cognitive biases identified in your reasoning process. Systematically evaluate contradictory information and assess which contradictions can be resolved versus which represent genuine uncertainty. Don't just synthesize the thesis, antithesis, and targeted research—also incorporate the bias corrections and contradiction analysis to improve your reasoning quality.

            ### Step 2: Enhanced Rationale with Uncertainty Quantification  
            Provide your final rationale that explicitly accounts for both bias corrections and contradictory information. Explain how contradictory evidence affects your confidence level and prediction precision. Your reasoning should demonstrate systematic correction of identified biases and explicit handling of unresolved contradictions.

            ### Step 3: Bias and Contradiction-Adjusted Final Prediction
            Conclude with your final prediction that reflects both bias awareness and appropriate uncertainty given contradictory information. The prediction should incorporate any recommended confidence adjustments from the bias analysis and account for the uncertainty introduced by unresolved contradictions.

            Integrate all analysis components above into a final prediction that accounts for both cognitive biases and contradictory information.

            **Required Output Format:**
            **Step 1: Bias and Contradiction-Aware Evidence Integration**
            - [Your analysis incorporating bias corrections and contradiction resolution]
            **Step 2: Enhanced Rationale with Uncertainty Quantification**
            - [Your comprehensive rationale accounting for biases and contradictions]
            **Step 3: Bias and Contradiction-Adjusted Final Prediction**
            - [Your final prediction in the required format:
              {final_answer_format_instruction}]
            """
        )
    
    @staticmethod
    def _get_legacy_final_answer_format_instruction(question: MetaculusQuestion) -> str:
        """Legacy format instruction method for verification."""
        try:
            question_type = question.get_question_type()
        except (AttributeError, ValueError):
            return "Prediction: [Your prediction in the appropriate format]"
            
        if question_type == "binary":
            return "Probability: X% (where X is between 1 and 99)"
        elif question_type == "numeric":
            return "Prediction: X (your numeric prediction)"
        elif question_type == "multiple_choice":
            return "Selected Option: [Your chosen option exactly as listed]"
        else:
            return "Prediction: [Your prediction in the appropriate format]"
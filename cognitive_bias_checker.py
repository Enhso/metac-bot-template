"""
Cognitive Bias Self-Correction Module

This module implements a specialized "Cognitive Bias Red Team" agent that reviews
forecasting rationales to identify potential cognitive biases and suggest specific
corrections. This enhances the self-critique process by making it more targeted
and systematic.
"""

import logging
from typing import Optional, Callable
from datetime import datetime

from forecasting_tools import clean_indents, MetaculusQuestion
from data_models import BiasAnalysisResult


class CognitiveBiasChecker:
    """
    A specialized agent that identifies cognitive biases in forecasting rationales
    and suggests specific corrections to improve logical soundness.
    
    This class implements a parallel LLM call with a "Cognitive Bias Red Team" persona
    that specifically focuses on bias detection and correction.
    """
    
    def __init__(
        self,
        get_llm: Callable[[str, str], any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the Cognitive Bias Checker.
        
        Args:
            get_llm: Function to retrieve LLM instances (name, kind) -> LLM
            logger: Optional logger instance
        """
        self._get_llm = get_llm
        self._logger = logger or logging.getLogger(__name__)
    
    async def analyze_for_cognitive_biases(
        self,
        question: MetaculusQuestion,
        rationale_text: str,
        reasoning_context: Optional[str] = None
    ) -> str:
        """
        Analyze a forecasting rationale for cognitive biases and suggest corrections.
        
        Args:
            question: The forecasting question being analyzed
            rationale_text: The reasoning/rationale to analyze for biases
            reasoning_context: Optional additional context about the reasoning process
            
        Returns:
            A detailed bias analysis with specific correction suggestions
        """
        self._logger.info(f"Starting cognitive bias analysis for question: {question.page_url}")
        
        prompt = self._build_bias_analysis_prompt(question, rationale_text, reasoning_context)
        
        bias_analysis = await self._get_llm("bias_checker_llm", "llm").invoke(prompt)
        
        self._logger.info(f"Completed cognitive bias analysis for question: {question.page_url}")
        
        return bias_analysis
    
    def _build_bias_analysis_prompt(
        self,
        question: MetaculusQuestion,
        rationale_text: str,
        reasoning_context: Optional[str] = None
    ) -> str:
        """
        Build the prompt for cognitive bias analysis.
        
        This prompt implements a specialized "Cognitive Bias Red Team" persona
        that systematically checks for common forecasting biases.
        """
        context_section = f"\n\n## Additional Context\n{reasoning_context}" if reasoning_context else ""
        
        return clean_indents(
            f"""
            You are a Cognitive Bias Red Team specialist with expertise in behavioral psychology and forecasting accuracy. Your sole mission is to identify cognitive biases that may be compromising the logical soundness of forecasting rationales and provide specific, actionable corrections.

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            ## The Forecasting Question
            {question.question_text}

            ## The Rationale Under Analysis
            ---
            {rationale_text}
            ---{context_section}

            ## Your Task: Systematic Bias Detection and Correction

            Analyze the provided rationale for the presence of cognitive biases that commonly affect forecasting accuracy. For each bias you identify, provide specific evidence from the text and concrete correction suggestions.

            ### Step 1: Systematic Bias Scan
            Review the rationale for the following common forecasting biases:

            **1. Anchoring Bias:** Is the reasoning overly influenced by initial information or reference points without adequate adjustment?

            **2. Availability Bias:** Does the reasoning over-weight easily recalled examples or recent events while ignoring less memorable but relevant data?

            **3. Confirmation Bias:** Does the analysis cherry-pick evidence that supports a predetermined conclusion while dismissing contradictory information?

            **4. Overconfidence Bias:** Does the reasoning display excessive certainty given the available evidence and inherent uncertainty?

            **5. Base Rate Neglect:** Does the analysis ignore or insufficiently weight relevant historical frequencies or prior probabilities?

            **6. Representativeness Bias:** Does the reasoning rely on superficial similarities without considering underlying statistical relationships?

            **7. Recency Bias:** Is there disproportionate weight given to recent events or trends without considering longer-term patterns?

            **8. Survivorship Bias:** Does the analysis focus only on visible successes/failures while ignoring unseen cases?

            **9. Planning Fallacy:** For temporal predictions, is there systematic underestimation of time, costs, or complexity?

            **10. Attribution Bias:** Are outcomes incorrectly attributed to specific causes when alternative explanations exist?

            ### Step 2: Evidence-Based Bias Identification
            For each bias you detect:
            - Quote the specific text that demonstrates the bias
            - Explain how this constitutes evidence of the bias
            - Assess the severity of the bias impact (Low/Medium/High)

            ### Step 3: Specific Correction Recommendations
            For each identified bias, provide:
            - A specific, actionable correction strategy
            - Alternative framings or evidence to consider
            - Suggested probability adjustments if applicable

            ### Step 4: Overall Assessment and Priority Corrections
            - Summarize the most critical biases affecting this rationale
            - Rank the corrections by potential impact on forecast accuracy
            - Provide an overall bias risk assessment (Low/Medium/High)

            **Required Output Format:**
            **Step 1: Bias Detection Summary**
            [List of biases detected with brief rationale for each]

            **Step 2: Detailed Bias Analysis**
            For each detected bias:
            **Bias: [Bias Name]**
            - Evidence: [Specific quote and explanation]
            - Severity: [Low/Medium/High]

            **Step 3: Correction Recommendations**
            For each detected bias:
            **Correction for [Bias Name]:**
            - Strategy: [Specific correction approach]
            - Alternative Considerations: [What should be considered instead/additionally]
            - Probability Adjustment: [If applicable, suggested direction of adjustment]

            **Step 4: Priority Assessment**
            - Critical Biases: [Top 1-2 biases to address first]
            - Impact Ranking: [Rank corrections by importance]
            - Overall Bias Risk: [Low/Medium/High with explanation]

            **Note:** If no significant biases are detected, explain why the reasoning appears well-calibrated and resistant to common biases. Focus on providing constructive analysis that enhances forecasting accuracy.
            """
        )
    
    def get_bias_correction_integration_prompt(
        self,
        original_rationale: str,
        bias_analysis: str
    ) -> str:
        """
        Generate a prompt for integrating bias corrections into the final rationale.
        
        This can be used to help the main forecasting agent incorporate the
        bias analysis findings into their final reasoning.
        """
        return clean_indents(
            f"""
            You have received a cognitive bias analysis of your reasoning. Your task is to integrate these insights to produce a more calibrated and bias-resistant final rationale.

            ## Your Original Rationale
            ---
            {original_rationale}
            ---

            ## Cognitive Bias Analysis and Corrections
            ---
            {bias_analysis}
            ---

            ## Integration Task
            1. **Acknowledge Key Biases:** Address the most critical biases identified in the analysis
            2. **Apply Corrections:** Implement the specific correction strategies suggested
            3. **Adjust Reasoning:** Modify your reasoning to account for previously overlooked considerations
            4. **Recalibrate Confidence:** Adjust your confidence levels based on the bias analysis
            5. **Maintain Coherence:** Ensure your updated rationale remains logically coherent and well-structured

            Produce an updated rationale that demonstrates you have systematically addressed the identified cognitive biases while maintaining the strength of your core arguments.
            """
        )

    @staticmethod
    def parse_analysis_text(
        question: MetaculusQuestion,
        analyzed_rationale: str,
        bias_analysis_text: str
    ) -> BiasAnalysisResult:
        """
        Parse the bias analysis text to extract structured information.
        
        This centralized method extracts key information from the bias analysis to create
        a structured result that can be used for further processing. This method is used
        by both BiasAwareEnsembleForecaster and BiasAwareSelfCritiqueForecaster to ensure
        consistent parsing logic.
        
        Args:
            question: The MetaculusQuestion being analyzed
            analyzed_rationale: The original rationale that was analyzed
            bias_analysis_text: The raw bias analysis text to parse
            
        Returns:
            BiasAnalysisResult with structured information extracted from the text
        """
        # Simple parsing - in production, this could be more sophisticated
        detected_biases = []
        priority_corrections = []
        severity_assessment = "Medium"  # Default
        confidence_adjustment_recommended = False
        
        # Extract detected biases (look for bias names in the text)
        bias_keywords = [
            "anchoring", "availability", "confirmation", "overconfidence",
            "base rate neglect", "representativeness", "recency", "survivorship",
            "planning fallacy", "attribution"
        ]
        
        text_lower = bias_analysis_text.lower()
        for bias in bias_keywords:
            if bias in text_lower:
                detected_biases.append(bias.title() + " Bias")
        
        # Check for severity indicators
        if "high" in text_lower and "risk" in text_lower:
            severity_assessment = "High"
        elif "low" in text_lower and "risk" in text_lower:
            severity_assessment = "Low"
        
        # Check for confidence adjustment recommendations
        if any(phrase in text_lower for phrase in ["adjust confidence", "recalibrate", "less certain", "more uncertain"]):
            confidence_adjustment_recommended = True
        
        # Extract priority corrections (simplified - look for correction section)
        if "critical biases" in text_lower or "priority" in text_lower:
            priority_corrections = detected_biases[:2]  # Take top 2 as priority
        
        return BiasAnalysisResult(
            question=question,
            analyzed_rationale=analyzed_rationale,
            bias_analysis_text=bias_analysis_text,
            detected_biases=detected_biases,
            severity_assessment=severity_assessment,
            priority_corrections=priority_corrections,
            confidence_adjustment_recommended=confidence_adjustment_recommended
        )


def build_bias_aware_refinement_prompt(
    question: MetaculusQuestion,
    initial_research: str,
    initial_prediction_text: str,
    critique_text: str,
    targeted_research: str,
    bias_analysis: str,
    final_answer_format_instruction: str,
    persona_prompt: Optional[str] = None,
) -> str:
    """
    Enhanced version of the refinement prompt that includes cognitive bias analysis.
    
    This integrates the cognitive bias corrections into the standard refinement process,
    making the final prediction more resistant to systematic biases.
    """
    persona_section = f"\n\n{persona_prompt}\n\n" if persona_prompt else "\n\n"
    
    return clean_indents(
        f"""
        You are a superforecaster producing a final, synthesized prediction that explicitly accounts for cognitive biases.{persona_section}

        Today is {datetime.now().strftime("%Y-%m-%d")}.

        ## Question
        {question.question_text}

        ## Background
        {question.background_info}

        ## Resolution Criteria
        {question.resolution_criteria}

        ## Comprehensive Analysis Dossier
        ### 1. Initial Research
        {initial_research}

        ### 2. Initial Prediction (Thesis)
        {initial_prediction_text}

        ### 3. Adversarial Critique (Antithesis)
        {critique_text}

        ### 4. Targeted Research (Additional Evidence)
        {targeted_research}

        ### 5. Cognitive Bias Analysis and Corrections
        {bias_analysis}

        ## Your Enhanced Synthesis Task
        
        Integrate all five sources of analysis above into a final, bias-resistant prediction. Pay special attention to the cognitive bias analysis, which identifies systematic errors that may be affecting your reasoning.

        ### Step 1: Bias-Aware Evidence Integration
        - Synthesize the research, predictions, and critiques while explicitly addressing the cognitive biases identified
        - Apply the specific correction strategies recommended in the bias analysis
        - Recalibrate your confidence based on the bias assessment

        ### Step 2: Systematic Bias Mitigation
        - Demonstrate how you are avoiding or correcting each identified bias
        - Show alternative perspectives or evidence you are now considering
        - Explain any adjustments to your reasoning process

        ### Step 3: Final Bias-Resistant Rationale
        - Present your comprehensive final reasoning that accounts for all analysis components
        - Explicitly note areas of uncertainty and potential remaining biases
        - Justify your final probability assessment with reference to bias mitigation

        ### Step 4: Calibrated Final Prediction
        - State your final prediction in the required format
        - Include a brief confidence assessment that reflects your bias analysis

        **Required Output Format:**
        **Step 1: Bias-Aware Evidence Integration**
        [Your synthesis accounting for bias corrections]

        **Step 2: Systematic Bias Mitigation**
        [How you addressed each identified bias]

        **Step 3: Final Bias-Resistant Rationale**
        [Your comprehensive final reasoning]

        **Step 4: Calibrated Final Prediction**
        {final_answer_format_instruction}
        [Brief confidence note about bias mitigation]
        """
    )
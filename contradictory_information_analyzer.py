"""
Contradictory Information Detection and Analysis Module

This module implements a specialized analyzer that identifies and addresses
contradictory evidence found during research, which is a key skill for superforecasting.
The analyzer systematically examines all gathered information to detect conflicts,
flag contradictions, and attempt to synthesize resolutions or flag irresolvable
conflicts as key uncertainties.
"""

import logging
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from forecasting_tools import clean_indents, MetaculusQuestion


@dataclass
class ContradictionAnalysisResult:
    """
    Contains the results of contradictory information analysis.
    
    This structure captures identified contradictions, attempted resolutions,
    and flags key uncertainties for the final forecast.
    """
    question: MetaculusQuestion
    analyzed_text: str
    detected_contradictions: List[Dict[str, Any]]  # Each dict contains: sources, description, severity
    resolution_attempts: List[Dict[str, Any]]  # Each dict contains: contradiction_id, strategy, outcome
    irresolvable_conflicts: List[Dict[str, Any]]  # Each dict contains: description, impact_on_forecast
    key_uncertainties: List[str]
    overall_coherence_assessment: str  # High/Medium/Low
    confidence_impact: str  # Description of how contradictions affect forecast confidence


class ContradictoryInformationAnalyzer:
    """
    A specialized analyzer that identifies contradictory evidence in research materials
    and attempts to reconcile conflicts or flag them as key uncertainties.
    
    This implements a systematic approach to handling conflicting information:
    1. Identify contradictory statements or evidence
    2. Categorize contradictions by type and severity
    3. Attempt to resolve conflicts through additional context or synthesis
    4. Flag irresolvable conflicts as key forecast uncertainties
    """
    
    def __init__(
        self,
        get_llm: Callable[[str, str], Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the Contradictory Information Analyzer.
        
        Args:
            get_llm: Function to retrieve LLM instances (name, kind) -> LLM
            logger: Optional logger instance
        """
        self._get_llm = get_llm
        self._logger = logger or logging.getLogger(__name__)
    
    async def analyze_contradictory_information(
        self,
        question: MetaculusQuestion,
        research_materials: Dict[str, str],
        context: Optional[str] = None
    ) -> ContradictionAnalysisResult:
        """
        Analyze research materials for contradictory information and attempt resolution.
        
        Args:
            question: The forecasting question being analyzed
            research_materials: Dict containing different research sources
                Expected keys: 'initial_research', 'targeted_research', 'initial_prediction'
            context: Optional additional context about the research process
            
        Returns:
            A comprehensive contradiction analysis with resolution attempts
        """
        self._logger.info(f"Starting contradictory information analysis for question: {question.page_url}")
        
        # Combine all research materials into a single text for analysis
        combined_text = self._combine_research_materials(research_materials)
        
        # Step 1: Detect contradictions
        contradictions = await self._detect_contradictions(question, combined_text, context)
        
        # Step 2: Attempt to resolve contradictions
        resolutions = await self._attempt_contradiction_resolution(question, contradictions, combined_text)
        
        # Step 3: Identify irresolvable conflicts and key uncertainties
        irresolvable, uncertainties = await self._identify_key_uncertainties(question, contradictions, resolutions)
        
        # Step 4: Assess overall coherence and confidence impact
        coherence_assessment = await self._assess_overall_coherence(question, contradictions, resolutions)
        
        result = ContradictionAnalysisResult(
            question=question,
            analyzed_text=combined_text,
            detected_contradictions=contradictions,
            resolution_attempts=resolutions,
            irresolvable_conflicts=irresolvable,
            key_uncertainties=uncertainties,
            overall_coherence_assessment=coherence_assessment['coherence'],
            confidence_impact=coherence_assessment['confidence_impact']
        )
        
        self._logger.info(f"Completed contradictory information analysis for question: {question.page_url} - "
                         f"Found {len(contradictions)} contradictions, resolved {len(resolutions)}, "
                         f"identified {len(irresolvable)} irresolvable conflicts")
        
        return result
    
    def _combine_research_materials(self, research_materials: Dict[str, str]) -> str:
        """Combine different research sources into a single text for analysis."""
        sections = []
        
        if 'initial_research' in research_materials:
            sections.append(f"## Initial Research\n{research_materials['initial_research']}")
        
        if 'targeted_research' in research_materials:
            sections.append(f"## Targeted Research\n{research_materials['targeted_research']}")
        
        if 'initial_prediction' in research_materials:
            sections.append(f"## Initial Analysis\n{research_materials['initial_prediction']}")
        
        if 'critique_text' in research_materials:
            sections.append(f"## Critical Analysis\n{research_materials['critique_text']}")
        
        return "\n\n".join(sections)
    
    async def _detect_contradictions(
        self,
        question: MetaculusQuestion,
        combined_text: str,
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictory statements or evidence in the research materials.
        """
        prompt = self._build_contradiction_detection_prompt(question, combined_text, context)
        
        detection_result = await self._get_llm("contradiction_analyzer_llm", "llm").invoke(prompt)
        
        # Parse the detection result into structured contradictions
        contradictions = self._parse_contradiction_detection_result(detection_result)
        
        return contradictions
    
    async def _attempt_contradiction_resolution(
        self,
        question: MetaculusQuestion,
        contradictions: List[Dict[str, Any]],
        combined_text: str
    ) -> List[Dict[str, Any]]:
        """
        Attempt to resolve identified contradictions through synthesis or additional context.
        """
        if not contradictions:
            return []
        
        prompt = self._build_resolution_prompt(question, contradictions, combined_text)
        
        resolution_result = await self._get_llm("contradiction_analyzer_llm", "llm").invoke(prompt)
        
        # Parse the resolution attempts
        resolutions = self._parse_resolution_result(resolution_result, contradictions)
        
        return resolutions
    
    async def _identify_key_uncertainties(
        self,
        question: MetaculusQuestion,
        contradictions: List[Dict[str, Any]],
        resolutions: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Identify irresolvable conflicts and derive key uncertainties for forecasting.
        """
        prompt = self._build_uncertainty_identification_prompt(question, contradictions, resolutions)
        
        uncertainty_result = await self._get_llm("contradiction_analyzer_llm", "llm").invoke(prompt)
        
        # Parse irresolvable conflicts and key uncertainties
        irresolvable, uncertainties = self._parse_uncertainty_result(uncertainty_result)
        
        return irresolvable, uncertainties
    
    async def _assess_overall_coherence(
        self,
        question: MetaculusQuestion,
        contradictions: List[Dict[str, Any]],
        resolutions: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Assess the overall coherence of the research and impact on forecast confidence.
        """
        prompt = self._build_coherence_assessment_prompt(question, contradictions, resolutions)
        
        coherence_result = await self._get_llm("contradiction_analyzer_llm", "llm").invoke(prompt)
        
        # Parse the coherence assessment
        assessment = self._parse_coherence_result(coherence_result)
        
        return assessment
    
    def _build_contradiction_detection_prompt(
        self,
        question: MetaculusQuestion,
        combined_text: str,
        context: Optional[str] = None
    ) -> str:
        """
        Build the prompt for detecting contradictory information.
        """
        context_section = f"\n\n## Additional Context\n{context}" if context else ""
        
        return clean_indents(
            f"""
            You are a specialized Contradiction Detection Agent with expertise in identifying conflicting evidence and statements in research materials. Your mission is to systematically identify contradictory information that could affect forecasting accuracy.

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            ## The Forecasting Question
            {question.question_text}

            ## Research Materials to Analyze
            ---
            {combined_text}
            ---{context_section}

            ## Your Task: Systematic Contradiction Detection

            Analyze the provided research materials to identify contradictory statements, conflicting evidence, or incompatible conclusions. Look for both explicit contradictions (direct conflicts) and implicit contradictions (statements that cannot both be true).

            ### Types of Contradictions to Identify:

            **1. Factual Contradictions:** Different sources providing conflicting factual claims or data points.

            **2. Temporal Contradictions:** Conflicting information about timing, sequences, or deadlines.

            **3. Causal Contradictions:** Different explanations for the same phenomenon or conflicting cause-effect relationships.

            **4. Quantitative Contradictions:** Conflicting numbers, percentages, or statistical claims.

            **5. Qualitative Contradictions:** Conflicting assessments, opinions, or interpretations of the same evidence.

            **6. Scope Contradictions:** Different definitions or boundaries for the same concept or event.

            **7. Methodological Contradictions:** Conflicting approaches or assumptions in analysis.

            ### Detection Process:

            **Step 1: Identify Potential Contradictions**
            - Scan for statements that directly conflict with each other
            - Look for implicit conflicts where multiple claims cannot simultaneously be true
            - Note any inconsistencies in data, timelines, or logical reasoning

            **Step 2: Categorize Each Contradiction**
            - Type: Which category of contradiction (from the 7 types above)
            - Sources: Which specific parts of the research contain the conflicting information
            - Severity: High (major impact on forecast), Medium (moderate impact), Low (minor impact)

            **Step 3: Document Evidence**
            - Quote the specific contradictory statements
            - Explain why they constitute a contradiction
            - Assess the potential impact on forecasting accuracy

            **Required Output Format:**
            **Contradiction Detection Summary**
            [Brief overview of how many contradictions found and their general nature]

            **Detailed Contradiction Analysis**
            For each contradiction identified:
            **Contradiction #[Number]**
            - Type: [Contradiction type]
            - Severity: [High/Medium/Low]
            - Source 1: "[Quote from first source]"
            - Source 2: "[Quote from contradicting source]"
            - Conflict Description: [Explanation of why these conflict]
            - Forecasting Impact: [How this contradiction affects forecast reliability]

            **Overall Assessment**
            - Total Contradictions: [Number]
            - High Severity: [Number]
            - Primary Areas of Conflict: [Main themes where contradictions occur]

            **Note:** If no significant contradictions are detected, explain why the research appears internally consistent and coherent. Focus on providing thorough analysis that enhances forecasting reliability.
            """
        )
    
    def _build_resolution_prompt(
        self,
        question: MetaculusQuestion,
        contradictions: List[dict],
        combined_text: str
    ) -> str:
        """
        Build the prompt for attempting to resolve contradictions.
        """
        contradictions_summary = "\n".join([
            f"Contradiction {i+1}: {c.get('description', 'No description')}"
            for i, c in enumerate(contradictions)
        ])
        
        return clean_indents(
            f"""
            You are a Contradiction Resolution Specialist tasked with attempting to reconcile conflicting evidence identified in research materials. Your goal is to find ways to synthesize contradictory information or determine if conflicts are irresolvable.

            ## The Forecasting Question
            {question.question_text}

            ## Identified Contradictions
            {contradictions_summary}

            ## Full Research Materials
            ---
            {combined_text}
            ---

            ## Your Task: Systematic Contradiction Resolution

            For each identified contradiction, attempt to find a resolution using the following strategies:

            ### Resolution Strategies:

            **1. Temporal Resolution:** Check if contradictions can be explained by different time periods or changing circumstances.

            **2. Scope Resolution:** Determine if contradictions arise from different definitions, boundaries, or scopes of analysis.

            **3. Source Quality Assessment:** Evaluate if one source is more reliable, recent, or authoritative than another.

            **4. Context Integration:** Look for additional context that might explain apparent contradictions.

            **5. Partial Truth Synthesis:** Determine if both contradictory statements could be partially true under different conditions.

            **6. Methodological Reconciliation:** Check if different methodologies or assumptions explain the contradictions.

            **7. Meta-Analysis:** Consider if contradictions reflect genuine uncertainty or evolving understanding of the topic.

            ### Resolution Process:

            **Step 1: Apply Resolution Strategies**
            For each contradiction, systematically apply the resolution strategies above.

            **Step 2: Assess Resolution Success**
            - Fully Resolved: Contradiction can be completely reconciled
            - Partially Resolved: Some aspects can be reconciled, others remain conflicted
            - Unresolved: No viable way to reconcile the contradiction

            **Step 3: Document Resolution Attempts**
            Clearly explain the reasoning and evidence for each resolution attempt.

            **Required Output Format:**
            **Resolution Attempts Summary**
            [Overview of resolution approach and general outcomes]

            **Detailed Resolution Analysis**
            For each contradiction:
            **Resolution for Contradiction #[Number]**
            - Strategy Applied: [Which resolution strategy was used]
            - Resolution Outcome: [Fully Resolved/Partially Resolved/Unresolved]
            - Explanation: [Detailed reasoning for the resolution attempt]
            - Remaining Conflicts: [If partially resolved or unresolved, what conflicts remain]
            - Confidence in Resolution: [High/Medium/Low]

            **Overall Resolution Assessment**
            - Total Contradictions Addressed: [Number]
            - Fully Resolved: [Number]
            - Partially Resolved: [Number]
            - Unresolved: [Number]
            - Key Insights: [Important findings from the resolution process]
            """
        )
    
    def _build_uncertainty_identification_prompt(
        self,
        question: MetaculusQuestion,
        contradictions: List[dict],
        resolutions: List[dict]
    ) -> str:
        """
        Build the prompt for identifying key uncertainties from unresolved contradictions.
        """
        return clean_indents(
            f"""
            You are an Uncertainty Identification Specialist tasked with determining how unresolved contradictions impact forecasting confidence and identifying key uncertainties that should be flagged in the final forecast.

            ## The Forecasting Question
            {question.question_text}

            ## Contradiction and Resolution Summary
            Total Contradictions: {len(contradictions)}
            Resolution Attempts: {len(resolutions)}

            ## Your Task: Key Uncertainty Identification

            Based on the contradictions and resolution attempts, identify:

            ### 1. Irresolvable Conflicts
            Contradictions that cannot be reconciled and represent fundamental disagreements or uncertainties in the evidence base.

            ### 2. Key Forecast Uncertainties
            The most important uncertainties that emerge from the contradictory information and should be explicitly acknowledged in the final forecast.

            ### 3. Confidence Impact Assessment
            How these unresolved contradictions should affect the overall confidence in the forecast.

            **Required Output Format:**
            **Irresolvable Conflicts**
            For each major unresolved contradiction:
            - Conflict Description: [Brief description of the irresolvable conflict]
            - Impact on Forecast: [How this conflict affects forecast reliability]
            - Uncertainty Level: [High/Medium/Low]

            **Key Uncertainties for Forecast**
            1. [First key uncertainty that should be flagged]
            2. [Second key uncertainty that should be flagged]
            3. [Continue as needed...]

            **Confidence Impact Assessment**
            - Overall Impact: [How contradictions affect forecast confidence]
            - Recommended Confidence Adjustment: [Should confidence be lowered and by how much]
            - Critical Unknowns: [Most important things that remain uncertain]
            """
        )
    
    def _build_coherence_assessment_prompt(
        self,
        question: MetaculusQuestion,
        contradictions: List[dict],
        resolutions: List[dict]
    ) -> str:
        """
        Build the prompt for assessing overall research coherence.
        """
        return clean_indents(
            f"""
            You are a Research Coherence Assessor tasked with evaluating the overall consistency and reliability of the research base after contradiction analysis and resolution attempts.

            ## The Forecasting Question
            {question.question_text}

            ## Analysis Summary
            - Total Contradictions Identified: {len(contradictions)}
            - Resolution Attempts Made: {len(resolutions)}

            ## Your Task: Overall Coherence Assessment

            Provide a comprehensive assessment of the research coherence and its implications for forecasting confidence.

            **Required Output Format:**
            **Overall Coherence Level**
            [High/Medium/Low] - [Explanation of why]

            **Confidence Impact**
            [Detailed description of how the contradiction analysis affects confidence in the forecast]

            **Research Quality Assessment**
            - Strength: [What aspects of the research are most reliable]
            - Weaknesses: [What aspects are most problematic due to contradictions]
            - Recommendations: [How to account for contradictions in the final forecast]
            """
        )
    
    def _parse_contradiction_detection_result(self, detection_result: str) -> List[Dict[str, Any]]:
        """
        Parse the contradiction detection result into structured format.
        This is a simplified parser - in production could be more sophisticated.
        """
        contradictions = []
        
        # Simple parsing logic - look for contradiction patterns
        lines = detection_result.split('\n')
        current_contradiction = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('**Contradiction #'):
                if current_contradiction:
                    contradictions.append(current_contradiction)
                current_contradiction = {'id': len(contradictions) + 1}
            elif line.startswith('- Type:'):
                current_contradiction['type'] = line.replace('- Type:', '').strip()
            elif line.startswith('- Severity:'):
                current_contradiction['severity'] = line.replace('- Severity:', '').strip()
            elif line.startswith('- Conflict Description:'):
                current_contradiction['description'] = line.replace('- Conflict Description:', '').strip()
            elif line.startswith('- Forecasting Impact:'):
                current_contradiction['impact'] = line.replace('- Forecasting Impact:', '').strip()
        
        # Add the last contradiction if it exists
        if current_contradiction:
            contradictions.append(current_contradiction)
        
        return contradictions
    
    def _parse_resolution_result(self, resolution_result: str, contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse the resolution result into structured format."""
        resolutions = []
        
        # Simple parsing logic
        lines = resolution_result.split('\n')
        current_resolution = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('**Resolution for Contradiction #'):
                if current_resolution:
                    resolutions.append(current_resolution)
                current_resolution = {'contradiction_id': len(resolutions) + 1}
            elif line.startswith('- Strategy Applied:'):
                current_resolution['strategy'] = line.replace('- Strategy Applied:', '').strip()
            elif line.startswith('- Resolution Outcome:'):
                current_resolution['outcome'] = line.replace('- Resolution Outcome:', '').strip()
            elif line.startswith('- Explanation:'):
                current_resolution['explanation'] = line.replace('- Explanation:', '').strip()
            elif line.startswith('- Confidence in Resolution:'):
                current_resolution['confidence'] = line.replace('- Confidence in Resolution:', '').strip()
        
        # Add the last resolution if it exists
        if current_resolution:
            resolutions.append(current_resolution)
        
        return resolutions
    
    def _parse_uncertainty_result(self, uncertainty_result: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """Parse the uncertainty identification result."""
        irresolvable = []
        uncertainties = []
        
        lines = uncertainty_result.split('\n')
        current_section = None
        current_conflict = {}
        
        # Debug: print the input for troubleshooting
        # print(f"DEBUG: Parsing uncertainty result:\n{uncertainty_result}")
        
        for line in lines:
            line = line.strip()
            # Debug: print current line and section
            # print(f"DEBUG: Line: '{line}', Section: {current_section}")
            
            if '**Irresolvable Conflicts**' in line:
                current_section = 'conflicts'
            elif '**Key Uncertainties for Forecast**' in line:
                current_section = 'uncertainties'
            elif current_section == 'conflicts':
                if line.startswith('- Conflict Description:'):
                    if current_conflict:
                        irresolvable.append(current_conflict)
                    current_conflict = {'description': line.replace('- Conflict Description:', '').strip()}
                elif line.startswith('- Impact on Forecast:') and current_conflict:
                    current_conflict['impact'] = line.replace('- Impact on Forecast:', '').strip()
            elif current_section == 'uncertainties':
                # Look for numbered list items or bullet points
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Extract the uncertainty text
                    if '.' in line and line[0].isdigit():
                        uncertainty_text = line.split('.', 1)[1].strip()
                    elif line.startswith('-'):
                        uncertainty_text = line[1:].strip()
                    else:
                        uncertainty_text = line
                    
                    if uncertainty_text:  # Only add non-empty uncertainties
                        uncertainties.append(uncertainty_text)
                        # Debug: print found uncertainty
                        # print(f"DEBUG: Found uncertainty: '{uncertainty_text}'")
        
        # Add the last conflict if it exists
        if current_conflict:
            irresolvable.append(current_conflict)
        
        # Debug: print final results
        # print(f"DEBUG: Final - Irresolvable: {len(irresolvable)}, Uncertainties: {len(uncertainties)}")
        
        return irresolvable, uncertainties
    
    def _parse_coherence_result(self, coherence_result: str) -> Dict[str, str]:
        """Parse the coherence assessment result."""
        assessment = {
            'coherence': 'Medium',  # Default
            'confidence_impact': 'Moderate impact on confidence due to some contradictions found.'  # Default
        }
        
        lines = coherence_result.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('**Overall Coherence Level**'):
                # Extract coherence level from next line or same line
                coherence_text = line.replace('**Overall Coherence Level**', '').strip()
                if coherence_text:
                    if 'High' in coherence_text:
                        assessment['coherence'] = 'High'
                    elif 'Low' in coherence_text:
                        assessment['coherence'] = 'Low'
                    else:
                        assessment['coherence'] = 'Medium'
            elif line.startswith('**Confidence Impact**'):
                # Look for the confidence impact in following lines
                impact_text = line.replace('**Confidence Impact**', '').strip()
                if impact_text:
                    assessment['confidence_impact'] = impact_text
        
        return assessment
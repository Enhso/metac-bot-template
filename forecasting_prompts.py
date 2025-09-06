from datetime import datetime
from typing import Optional
from forecasting_tools import clean_indents, MetaculusQuestion, MultipleChoiceQuestion, NumericQuestion


# ----------------------------- Shared Builders -----------------------------

def build_keyword_extractor_prompt(text: str) -> str:
    return clean_indents(
        f"""
        Your task is to extract critical search keywords from the user's text.
        Do not add any explanation, preamble, or formatting.
        Your entire response must be a single line of space-separated keywords.

        Here is an example:
        Text: "Will the US Federal Reserve raise interest rates in the next quarter of 2025 due to inflation concerns?"
        Keywords: US Federal Reserve interest rates 2025 inflation

        Now, perform this task on the following text:
        Text: "{text}"
        Keywords:
        """
    )


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ----------------------------- Strategy Prompts -----------------------------

def build_initial_prediction_prompt(question: MetaculusQuestion, initial_research: str) -> str:
    return clean_indents(
        f"""
        You are a superforecaster following the principles of Philip Tetlock. Your reasoning must be transparent, self-critical, and grounded in evidence. Your goal is to produce an initial, rigorously derived forecast.

        Today is {_today_str()}.

        ## Question:
        {question.question_text}

        ## Background:
        {question.background_info}

        ## Resolution Criteria:
        {question.resolution_criteria}

        ## Available Research:
        {initial_research}

        ## Your Task:
        Generate an initial forecast by following these four steps precisely.

        ### Step 1: Triage and Deconstruction (Fermi-ize)
        First, assess the question's tractability. Is it a "clock-like" or "cloud-like" problem? Then, break the core question down into smaller, more manageable, and quantifiable components. List these sub-questions.

        ### Step 2: Establish the Outside View
        Identify a suitable reference class for this event. What is the base rate of outcomes for similar situations? State the reference class and the resulting base rate probability. This will be your initial anchor.

        ### Step 3: Integrate the Inside View
        Now, analyze the unique, case-specific evidence from the provided research. How does this new information adjust your initial anchor from the outside view? Systematically discuss the evidence for and against, adjusting your probability estimate up or down. Mention any potential biases (e.g., availability bias, confirmation bias) that might be influencing your interpretation of the inside view.

        ### Step 4: Initial Forecast and Rationale Synthesis
        Synthesize your findings from the steps above into a coherent rationale. Clearly state your key assumptions and where your uncertainty lies. Conclude with your initial prediction in the precise format required.

        **Required Output Format:**
        **Step 1: Triage and Deconstruction**
        - [Your analysis of the question's tractability and your list of sub-questions]
        **Step 2: Outside View**
        - Reference Class: [Your identified reference class]
        - Base Rate/Anchor: [Your calculated base rate]
        **Step 3: Inside View**
        - [Your analysis adjusting the anchor based on case-specific evidence]
        **Step 4: Initial Forecast and Rationale Synthesis**
        - [Your synthesized rationale]
        - [Your final prediction in the required format]
        """
    )


def build_adversarial_critique_prompt(
    question: MetaculusQuestion, initial_prediction_text: str
) -> str:
    return clean_indents(
        f"""
        You are an intelligence analyst assigned to conduct a "red team" exercise. Your sole purpose is to challenge a colleague's forecast with constructive, aggressive skepticism. Do not be agreeable. Your goal is to expose every potential weakness.

        Today is {_today_str()}.

        ## The Original Question:
        {question.question_text}

        ## Colleague's Initial Forecast and Rationale:
        ---
        {initial_prediction_text}
        ---

        ## Your Task:
        Critique this forecast by addressing the following points:

        1.  **Challenge Core Assumptions:** Identify the 2-3 most critical stated or unstated assumptions in the initial forecast. Why might they be wrong?
        2.  **Propose an Alternative Perspective:** Actively consider the opposite conclusion. What key evidence or alternative interpretation was downplayed or missed entirely?
        3.  **Stress-Test the Outside View:** Was the chosen reference class appropriate? Propose at least one alternative reference class and explain how it might change the forecast.
        4.  **Generate High-Value Questions:** Conclude with a list of 2-3 specific, researchable questions. Each question should be self-contained and not refer to any external information. These questions should be designed to resolve the greatest points of uncertainty you've identified and have the highest potential to falsify the initial forecast.
        """
    )


def build_extract_questions_from_critique_prompt(critique_text: str) -> str:
    return clean_indents(
        f"""
        Extract the specific, researchable questions from the following text.
        List only the questions, each on a new line. If there are no questions, return an empty string.

        Text:
        ---
        {critique_text}
        ---
        """
    )


def build_refined_prediction_prompt(
    question: MetaculusQuestion,
    initial_research: str,
    initial_prediction_text: str,
    critique_text: str,
    targeted_research: str,
    final_answer_format_instruction: str,
    persona_prompt: Optional[str] = None,
) -> str:
    persona_section = f"\n\n{persona_prompt}\n\n" if persona_prompt else "\n\n"
    return clean_indents(
        f"""
        You are a superforecaster producing a final, synthesized prediction.{persona_section}

        Today is {_today_str()}.

        ## Question
        {question.question_text}

        ## Background
        {question.background_info}

        ## Resolution Criteria
        {question.resolution_criteria}

        ## Dossier
        ### 1. Initial Research
        {initial_research}
        ### 2. Initial Prediction (Thesis)
        {initial_prediction_text}
        ### 3. Adversarial Critique (Antithesis)
        {critique_text}
        ### 4. New, Targeted Research
        {targeted_research}

        ## Your Task:
        Follow this three-step process to generate your final analysis.

        ### Step 1: Synthesize Thesis, Antithesis, and New Evidence
        Adopt a "dragonfly eye" perspective. Explicitly discuss how the critique and the targeted research have altered your initial view. Which arguments from the initial forecast still hold, and which have been weakened or overturned? Weigh the conflicting points and synthesize them. Don't just discard one view for another; integrate them.

        ### Step 2: Final Rationale and Probabilistic Thinking
        Provide your final, comprehensive rationale. Explain how you are balancing the competing causal forces. Your reasoning should be granular, distinguishing between multiple degrees of uncertainty. Acknowledge what you still don't know and what key indicators could change your mind in the future.

        ### Step 3: Final Calibrated Prediction
        Conclude with your final prediction. Update your numerical forecast with precision, reflecting the synthesis above. Ensure it is in the precise format required.

        **Required Output Format:**
        **Step 1: Synthesis**
        - [Your discussion on how the critique and new data changed the forecast]
        **Step 2: Final Rationale**
        - [Your comprehensive final rationale and remaining uncertainties]
        **Step 3: Final Prediction**
        - [Your final prediction in the required format:
          {final_answer_format_instruction}]
        """
    )


# ----------------------------- Ensemble Personas -----------------------------

PERSONAS = {
    "The Skeptic": clean_indents(
        """
        ## Your Persona: The Skeptic
        You are playing the role of a cautious, skeptical analyst. Your primary goal is to identify and prioritize risks, potential failure points, and reasons why the event will not happen. Challenge assumptions and focus on the downside.
        """
    ),
    "The Proponent": clean_indents(
        """
        ## Your Persona: The Proponent
        You are playing the role of a forward-looking, optimistic analyst. Your primary goal is to identify catalysts, opportunities, and the strongest arguments for why the event will happen. Focus on the upside potential and the driving forces for success.
        """
    ),
    "The Quant": clean_indents(
        """
        ## Your Persona: The Quant
        You are playing the role of a data-driven quantitative analyst. Your reasoning must be strictly grounded in the provided data, base rates, and statistical evidence. Ignore narrative, anecdotal evidence, and qualitative arguments. Focus only on the numbers.
        """
    ),
}


# ----------------------------- Legacy (no-framework) Templates -----------------------------

BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Question background:
{background}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

NUMERIC_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Background:
{background}

{resolution_criteria}

{fine_print}

Units for answer: {units}

Your research assistant says:
{summary_report}

Today is {today}.

{lower_bound_message}
{upper_bound_message}


Formatting Instructions:
- Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unkowns.

The last thing you write is your final answer as:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

The options are: {options}


Background:
{background}

{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""

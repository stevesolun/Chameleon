"""
Distortion Constants - Single Source of Truth

All distortion-related constants, formulas, and rules are defined here.
This eliminates duplication across multiple files.
"""

import math
from typing import Dict, List

# ============================================================================
# MIU (μ) Distortion Rules
# ============================================================================

MIU_RULES: Dict[float, str] = {
    0.0: "NONE: Keep question exactly as is. No changes allowed.",
    0.1: "MINIMAL: Change only 1-2 words using simple synonyms. Keep EXACT sentence structure.",
    0.2: "LIGHT: Change 2-3 words with synonym replacements. Maintain sentence flow.",
    0.3: "MODERATE: Change 3-4 words. Some phrase restructuring allowed.",
    0.4: "HEAVY LEXICAL: Extensive vocabulary changes. Moderate restructuring.",
    0.5: "LIGHT MIXED: Combine lexical and structural changes.",
    0.6: "MODERATE MIXED: Significant changes. Different constructions allowed.",
    0.7: "HEAVY MIXED: Major restructuring. Can change voice.",
    0.8: "NEAR PARAPHRASE: Extensive restructuring. Different patterns.",
    0.9: "FULL PARAPHRASE: Complete reconstruction. Maximum variation.",
}

# Default miu values for distortion levels
DEFAULT_MIU_VALUES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Default number of distortions per question per miu level
DEFAULT_DISTORTIONS_PER_QUESTION: int = 10

# ============================================================================
# Temperature Calculation Formula
# ============================================================================

def calculate_temperature(miu: float) -> float:
    """
    Calculate optimal temperature for distortion based on miu value.
    
    The formula combines three components:
    1. Exponential base: Higher miu = more temperature
    2. Sigmoid activation: Sharper increase around miu=0.5
    3. Paraphrase boost: Extra temperature for high miu (>=0.7)
    
    Args:
        miu: Distortion intensity (0.0 to 1.0)
    
    Returns:
        Temperature value (clamped to 0.1-1.5 for API compatibility)
    """
    if miu == 0.0:
        return 0.1
    
    # Component 1: Exponential base
    exp_component = 0.3 + (miu ** 1.5) * 1.2
    
    # Component 2: Sigmoid activation around miu=0.5
    sig_component = 0.3 * (1 / (1 + math.exp(-10 * (miu - 0.5))))
    
    # Component 3: Paraphrase boost for high miu
    para_component = 0.2 * ((miu - 0.7) / 0.3) ** 2 if miu >= 0.7 else 0
    
    # Combine and clamp
    temperature = exp_component + sig_component + para_component
    return round(max(0.1, min(1.5, temperature)), 2)


# ============================================================================
# Prompt Templates
# ============================================================================

DISTORTION_SYSTEM_PROMPT = """You are a semantic distortion expert. Your task is to create lexically 
distorted versions of questions while preserving their meaning and correct answer.

CRITICAL RULES:
- Use ONLY synonyms, paraphrasing, and sentence restructuring
- NEVER add typos, misspellings, or character-level noise
- NEVER use leetspeak (no 3 for e, no 0 for o, etc.)
- NEVER add random characters or symbols
- NEVER use markdown formatting (no ** or ## or `)
- NEVER start with preambles like "Here are..." or "Sure, here..."
- Each distortion must be grammatically correct
- The correct answer must remain valid after distortion
- Output exactly the requested number of distortions"""


def get_distortion_prompt(question: str, miu: float, n_distortions: int) -> str:
    """
    Generate a prompt for distorting a single question.
    
    Args:
        question: The original question text
        miu: Distortion intensity level
        n_distortions: Number of unique distortions to generate
    
    Returns:
        Formatted prompt string
    """
    rule = MIU_RULES.get(miu, MIU_RULES[0.5])
    
    return f"""Distort this question {n_distortions} unique ways at μ={miu}.

DISTORTION RULE for μ={miu}: {rule}

QUESTION: {question}

STRICT REQUIREMENTS:
- Use ONLY synonyms and sentence restructuring
- NO typos, NO misspellings, NO character noise
- NO leetspeak, NO random characters
- NO markdown formatting (no ** or ##)
- NO preambles like "Here are..."
- Each distortion must be grammatically correct
- Preserve the correct answer exactly
- Follow the μ={miu} rule EXACTLY

OUTPUT (exactly {n_distortions} numbered lines):
1. [distortion]
2. [distortion]
...
{n_distortions}. [distortion]"""


def get_batch_distortion_prompt(questions: List[Dict[str, str]], miu: float, n_distortions: int) -> str:
    """
    Generate a prompt for distorting multiple questions in one API call.
    
    Args:
        questions: List of dicts with 'text' key
        miu: Distortion intensity level
        n_distortions: Number of unique distortions per question
    
    Returns:
        Formatted prompt string
    """
    rule = MIU_RULES.get(miu, MIU_RULES[0.5])
    q_text = "".join(f"\nQ{i}: {q['text']}\n" for i, q in enumerate(questions, 1))
    
    return f"""⚠️ MANDATORY: You MUST output EXACTLY {n_distortions} UNIQUE distortions for EACH question. NO MORE, NO LESS.

TASK: Distort each question at μ={miu} intensity level.

DISTORTION RULE for μ={miu}: {rule}
{q_text}

═══════════════════════════════════════════════════════════════════════
ABSOLUTE REQUIREMENTS (VIOLATION = FAILURE):
═══════════════════════════════════════════════════════════════════════

1. QUANTITY: Output EXACTLY {n_distortions} distortions per question
   - Not {n_distortions - 1}, not {n_distortions + 1} — EXACTLY {n_distortions}
   - Each must be numbered 1 through {n_distortions}

2. UNIQUENESS: All {n_distortions} distortions MUST be different from each other
   - NO duplicates allowed
   - NO near-duplicates (different by just 1-2 words)
   - Each distortion must have substantial unique variation

3. PRESERVATION: The correct answer must remain valid for all distortions

4. FORBIDDEN:
   ❌ NO typos, misspellings, or character noise
   ❌ NO leetspeak (3→e, 0→o, 4→a, @→a)
   ❌ NO random characters or symbols
   ❌ NO markdown (**bold**, ##headers, `code`)
   ❌ NO preambles ("Here are", "Sure", "Certainly")
   ❌ NO explanations or meta-commentary

5. ALLOWED:
   ✅ Synonyms and vocabulary changes
   ✅ Sentence restructuring
   ✅ Voice changes (active↔passive)
   ✅ Clause reordering

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (EXACT):
═══════════════════════════════════════════════════════════════════════

Q1:
1. [first unique distortion]
2. [second unique distortion]
3. [third unique distortion]
... continue to {n_distortions}
{n_distortions}. [{n_distortions}th unique distortion]

Q2:
1. [first unique distortion]
... continue for all questions

BEGIN OUTPUT NOW:"""


def get_retry_distortion_prompt(question: Dict, miu: float, needed: int) -> str:
    """
    Generate a prompt for retrying distortions when we already have some filled.
    Includes existing distortions so the LLM knows what to avoid.
    
    Args:
        question: Dict with 'text', 'existing' (list of existing distortions), 'needed' (how many more)
        miu: Distortion intensity level
        needed: Number of NEW distortions needed
    
    Returns:
        Formatted prompt string
    """
    rule = MIU_RULES.get(miu, MIU_RULES[0.5])
    q_text = question['text']
    existing = question.get('existing', [])
    
    existing_list = ""
    if existing:
        existing_list = "\n\n⚠️ ALREADY GENERATED (DO NOT REPEAT THESE OR SIMILAR):\n"
        for i, e in enumerate(existing, 1):
            existing_list += f"  {i}. {e}\n"
    
    return f"""⚠️ CRITICAL: Generate EXACTLY {needed} NEW unique distortion(s) for this question.

QUESTION: {q_text}

DISTORTION RULE for μ={miu}: {rule}
{existing_list}

═══════════════════════════════════════════════════════════════════════
REQUIREMENTS:
═══════════════════════════════════════════════════════════════════════

1. Generate EXACTLY {needed} NEW distortion(s)
2. Each must be COMPLETELY DIFFERENT from the existing {len(existing)} distortions above
3. Use different synonyms, different sentence structures, different word orders
4. NO duplicates or near-duplicates of existing ones
5. Follow the μ={miu} rule: {rule}

FORBIDDEN:
❌ NO typos, misspellings, or character noise
❌ NO leetspeak (3→e, 0→o)
❌ NO markdown (**bold**, ##headers)
❌ NO preambles ("Here are", "Sure")
❌ DO NOT repeat or slightly modify any existing distortion

OUTPUT FORMAT (EXACT):
1. [first NEW unique distortion]
{"2. [second NEW unique distortion]" if needed > 1 else ""}
... up to {needed}

BEGIN OUTPUT NOW:"""


# ============================================================================
# Evaluation Prompt (for target model)
# ============================================================================

EVALUATION_SYSTEM_PROMPT = """You are an expert exam-taker answering multiple choice questions.
Your ONLY output is the letter(s) of the correct answer(s).
NEVER explain, NEVER add text, NEVER show reasoning.
If multiple answers: separate with comma (e.g., "A, D")."""


def get_evaluation_prompt(question: str, choices: Dict[str, str]) -> str:
    """
    Generate a prompt for evaluating a question with the target model.
    
    RLHF-style prompt with scoring incentive.
    Handles both single (A, B, C, D) and multiple answer questions (A, D).
    
    Args:
        question: The (possibly distorted) question text
        choices: Dict mapping letters to choice text, e.g. {"A": "...", "B": "..."}
                 Keys can be lowercase or uppercase (treated the same)
    
    Returns:
        Formatted prompt string
    """
    # Normalize keys to uppercase for consistent display
    normalized_choices = {}
    for k, v in choices.items():
        key = k.strip().upper() if isinstance(k, str) else str(k).upper()
        # Handle keys like "A:" or "a:" - extract just the letter
        if key.endswith(':'):
            key = key[:-1]
        if len(key) == 1 and key in 'ABCDEFGH':
            normalized_choices[key] = v
    
    # Sort and format choices
    choices_text = "\n".join([f"{k}: {v}" for k, v in sorted(normalized_choices.items())])
    
    return f"""═══════════════════════════════════════════════════════════════════════
MEDICAL CERTIFICATION EXAMINATION
═══════════════════════════════════════════════════════════════════════

SCORING SYSTEM:
• Correct answer: +1 point
• Incorrect answer: -2 points penalty
• Your goal: Maximize total score

RULES:
1. Read the question VERY carefully
2. Analyze ALL options before answering
3. There may be ONE or MULTIPLE correct answers
4. Respond with ONLY the letter(s)
5. For multiple answers: use comma (e.g., "A, D" or "B, C")
6. Order does NOT matter (A, D = D, A)
7. NO explanations, NO reasoning, NO additional text

═══════════════════════════════════════════════════════════════════════
QUESTION:
{question}

OPTIONS:
{choices_text}
═══════════════════════════════════════════════════════════════════════

YOUR ANSWER (letter(s) only):"""


# ============================================================================
# API Configuration Defaults
# ============================================================================

API_DEFAULTS = {
    "mistral": {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "default_model": "mistral-large-latest",
        "max_tokens": 8000,  # Increased default for longer responses
        "timeout": 120,
    },
    "openai": {
        "default_model": "gpt-5.1",
        "max_tokens": 4000,
        "timeout": 120,
    },
}

# Batch processing defaults
BATCH_DEFAULTS = {
    "questions_per_batch": 5,  # Questions per API call for distortion
    "workers_per_miu": 2,      # Parallel workers per miu level
    "max_retries": 3,          # Retry attempts for failed API calls
    "save_interval": 30,       # Seconds between auto-saves
}


def calculate_max_tokens(questions: list, n_distortions: int, miu: float = 0.5, buffer_pct: float = 0.20) -> int:
    """
    Calculate max_tokens dynamically based on question length and miu.
    
    Higher miu = more paraphrasing = potentially longer output.
    
    Length multiplier by miu:
    - miu 0.0-0.3: 1.2x (minimal changes, similar length)
    - miu 0.4-0.6: 1.4x (moderate changes)
    - miu 0.7-0.9: 1.6x (heavy paraphrasing, can be longer)
    
    Args:
        questions: List of question dicts with 'text' key
        n_distortions: Number of distortions per question
        miu: Distortion intensity (0.0-1.0)
        buffer_pct: Additional buffer percentage (default 20%)
    
    Returns:
        Recommended max_tokens for the API call
    """
    # Length multiplier based on miu (higher miu = more expansion allowed)
    if miu <= 0.3:
        length_multiplier = 1.2
    elif miu <= 0.6:
        length_multiplier = 1.4
    else:
        length_multiplier = 1.6
    
    # Estimate tokens (roughly 1.5 tokens per word for English)
    total_words = sum(len(q.get('text', '').split()) for q in questions)
    
    # Each question needs N distortions, each distortion ~multiplier * original length
    # Plus formatting overhead (~20 tokens per distortion for "Q1:", "1.", "2.", newlines etc.)
    estimated_output_words = total_words * length_multiplier * n_distortions
    formatting_overhead = len(questions) * n_distortions * 20  # Q markers + numbering + newlines
    
    # Convert words to tokens (1.5x for safety) and add buffer
    estimated_tokens = int((estimated_output_words * 1.5 + formatting_overhead) * (1 + buffer_pct))
    
    # Clamp between reasonable bounds
    min_tokens = 2000  # Minimum for any reasonable response
    max_tokens = 16000  # Mistral large supports up to 32k context
    
    return max(min_tokens, min(estimated_tokens, max_tokens))


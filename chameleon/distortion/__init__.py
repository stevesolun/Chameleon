"""
Distortion module for generating lexically distorted questions.

This module provides:
- Constants and configuration for distortion parameters
- Validation for distortion quality
- Runner for executing distortion generation
- Engine abstraction for different LLM backends
"""

from chameleon.distortion.constants import (
    MIU_RULES,
    DEFAULT_MIU_VALUES,
    DEFAULT_DISTORTIONS_PER_QUESTION,
    calculate_temperature,
    get_distortion_prompt,
    get_batch_distortion_prompt,
    get_evaluation_prompt,
    DISTORTION_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    API_DEFAULTS,
    BATCH_DEFAULTS,
)

from chameleon.distortion.validator import (
    ValidationFailure,
    ValidationResult,
    validate_distortion,
    validate_batch,
    clean_distortion,
    parse_llm_response,
    get_validation_stats,
)

from chameleon.distortion.runner import (
    DistortionConfig,
    DistortionProgress,
    DistortionRunner,
    run_distortions,
)

__all__ = [
    # Constants
    "MIU_RULES",
    "DEFAULT_MIU_VALUES",
    "DEFAULT_DISTORTIONS_PER_QUESTION",
    "calculate_temperature",
    "get_distortion_prompt",
    "get_batch_distortion_prompt",
    "get_evaluation_prompt",
    "DISTORTION_SYSTEM_PROMPT",
    "EVALUATION_SYSTEM_PROMPT",
    "API_DEFAULTS",
    "BATCH_DEFAULTS",
    # Validation
    "ValidationFailure",
    "ValidationResult",
    "validate_distortion",
    "validate_batch",
    "clean_distortion",
    "parse_llm_response",
    "get_validation_stats",
    # Runner
    "DistortionConfig",
    "DistortionProgress",
    "DistortionRunner",
    "run_distortions",
]

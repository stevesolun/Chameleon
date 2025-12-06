"""Analysis module - Metrics, statistical tests, and visualizations."""

from chameleon.analysis.metrics import (
    calculate_accuracy,
    calculate_accuracy_by_group,
    calculate_degradation,
)
from chameleon.analysis.mcnemar import (
    mcnemar_test,
    analyze_distortion_significance,
    analyze_subject_significance,
)
from chameleon.analysis.visualizations import (
    create_degradation_heatmap,
    create_accuracy_plots,
    create_key_insights_summary,
)
from chameleon.analysis.run_analysis import run_full_analysis

__all__ = [
    "calculate_accuracy",
    "calculate_accuracy_by_group",
    "calculate_degradation",
    "mcnemar_test",
    "analyze_distortion_significance",
    "analyze_subject_significance",
    "create_degradation_heatmap",
    "create_accuracy_plots",
    "create_key_insights_summary",
    "run_full_analysis",
]



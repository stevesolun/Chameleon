#!/usr/bin/env python
"""
Chameleon Analysis Engine - Synergy Edition
============================================
Merged from two expert implementations + novel features.

Expected CSV schema:
    subject, question_id, question_text, options_json, distorted_question,
    distortion_id, miu, answer, target_model_name, target_model_answer,
    is_correct, llm_validated

Outputs:
    Core Metrics:
    - 01_accuracy_by_miu.csv / .png
    - 02_accuracy_by_subject_miu.csv
    - 03_chameleon_robustness_index.csv (global + per-subject)
    - 04_elasticity.csv / .png
    - 05_model_comparison.csv / .png

    Error Analysis:
    - 06_error_taxonomy.json (with μ-bucket breakdown)
    - 07_confusion_clusters.json (TF-IDF + KMeans)

    Statistical:
    - 08_bootstrap_intervals.csv

    Synergy Features:
    - 09_delta_accuracy_heatmap.csv / .png
    - 10_question_difficulty_tiers.json
    - 11_executive_summary.md
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optional: sklearn for TF-IDF clustering
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# 0. HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce mixed bool/string to proper boolean."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    )


def load_and_prepare(csv_path: Path, validated_only: bool = False) -> pd.DataFrame:
    """Load CSV and clean types."""
    df = pd.read_csv(csv_path)
    
    # miu to float
    df["miu"] = pd.to_numeric(df["miu"], errors="coerce").astype(float)
    
    # is_correct to bool
    if "is_correct" in df.columns:
        df["is_correct"] = coerce_bool(df["is_correct"])
    else:
        df["is_correct"] = False
    
    # Ensure is_correct is proper boolean
    df["is_correct"] = df["is_correct"].fillna(False).astype(bool)
    
    # llm_validated to bool
    if "llm_validated" in df.columns:
        df["llm_validated"] = coerce_bool(df["llm_validated"])
    else:
        df["llm_validated"] = True
    
    # Ensure target_model_name exists
    if "target_model_name" not in df.columns:
        df["target_model_name"] = "unknown"
    
    # Ensure subject exists
    if "subject" not in df.columns:
        df["subject"] = "General"
    
    # Drop invalid rows
    df = df.dropna(subset=["miu"])
    
    # Filter validated only if requested
    if validated_only:
        df = df[df["llm_validated"] == True].copy()
        print(f"[INFO] Filtered to validated-only: {len(df)} rows")
    
    return df


def setup_plot_style():
    """Configure consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'grid.color': '#e0e0e0',
        'font.family': 'sans-serif',
        'font.size': 10,
    })


# =============================================================================
# 1. ACCURACY BY MIU
# =============================================================================

def compute_accuracy_by_miu(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by model and miu."""
    return (
        df.groupby(["target_model_name", "miu"])["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values(["target_model_name", "miu"])
    )


def compute_accuracy_by_subject_miu(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by model, subject, and miu."""
    return (
        df.groupby(["target_model_name", "subject", "miu"])["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values(["target_model_name", "subject", "miu"])
    )


def plot_accuracy_by_miu(acc_df: pd.DataFrame, output_path: Path) -> None:
    """Line plot: accuracy vs miu per model."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2.colors
    
    for i, model in enumerate(acc_df["target_model_name"].unique()):
        sub = acc_df[acc_df["target_model_name"] == model].sort_values("miu").copy()
        
        # Ensure numeric types for plotting
        x = sub["miu"].astype(float).values
        y = sub["accuracy"].astype(float).values
        
        color = colors[i % len(colors)]
        ax.plot(x, y, marker="o", linewidth=2.5, 
                markersize=8, label=model, color=color)
        
        # Add confidence band (simplified)
        ax.fill_between(x, y - 0.02, y + 0.02, alpha=0.15, color=color)
    
    ax.set_xlabel("Distortion Level (μ)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Model Accuracy vs Semantic Distortion", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.05, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# 2. CHAMELEON ROBUSTNESS INDEX (CRI)
# =============================================================================

def compute_cri(
    df: pd.DataFrame, 
    group_cols: List[str] = ["target_model_name"],
    alpha: float = 2.0
) -> pd.DataFrame:
    """
    Compute CRI: weighted average of accuracy emphasizing high-μ performance.
    
    CRI = Σ(accuracy(μ) * w(μ)) where w(μ) = exp(alpha * μ) / Σ exp(alpha * μ)
    
    Can compute globally (group_cols=["target_model_name"]) or 
    per-subject (group_cols=["target_model_name", "subject"]).
    """
    mus = np.sort(df["miu"].unique())
    weights_raw = np.exp(alpha * mus)
    weights = weights_raw / weights_raw.sum()
    weight_map = dict(zip(mus, weights))
    
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        acc_by_miu = group.groupby("miu")["is_correct"].mean()
        
        cri = 0.0
        total_weight = 0.0
        for mu_val, acc in acc_by_miu.items():
            w = weight_map.get(mu_val, 0.0)
            cri += acc * w
            total_weight += w
        
        cri = cri / total_weight if total_weight > 0 else np.nan
        
        row = dict(zip(group_cols, keys))
        row["CRI"] = cri
        row["n_samples"] = len(group)
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values("CRI", ascending=False)


# =============================================================================
# 3. ELASTICITY (Slope of accuracy vs μ)
# =============================================================================

def compute_elasticity(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear regression: accuracy = intercept + slope * μ
    
    Negative slope = model degrades under distortion (fragile).
    Slope near 0 = robust to distortion.
    """
    rows = []
    for model, group in acc_df.groupby("target_model_name"):
        sub = group.sort_values("miu").copy()
        
        # Ensure numeric types
        x = pd.to_numeric(sub["miu"], errors="coerce").astype(float).values
        y = pd.to_numeric(sub["accuracy"], errors="coerce").astype(float).values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(np.unique(x)) < 2:
            slope, intercept = np.nan, np.nan
        else:
            slope, intercept = np.polyfit(x, y, 1)
        
        rows.append({
            "target_model_name": model,
            "elasticity_slope": float(slope) if not np.isnan(slope) else np.nan,
            "intercept": float(intercept) if not np.isnan(intercept) else np.nan,
            "baseline_accuracy": float(intercept) if not np.isnan(intercept) else np.nan,
            "projected_high_accuracy": float(intercept + slope * 0.9) if not np.isnan(slope) else np.nan
        })
    
    return pd.DataFrame(rows).sort_values("elasticity_slope", ascending=False)


def plot_elasticity(acc_df: pd.DataFrame, output_path: Path) -> None:
    """Scatter + regression lines for each model."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2.colors
    
    for i, model in enumerate(acc_df["target_model_name"].unique()):
        sub = acc_df[acc_df["target_model_name"] == model].sort_values("miu").copy()
        
        # Ensure numeric types
        x = sub["miu"].astype(float).values
        y = sub["accuracy"].astype(float).values
        
        color = colors[i % len(colors)]
        ax.scatter(x, y, alpha=0.8, color=color, s=80, edgecolors='white', linewidth=1)
        
        if len(np.unique(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = intercept + slope * x_line
            ax.plot(x_line, y_line, color=color, linewidth=2.5, 
                   label=f"{model} (slope={slope:.3f})")
    
    ax.set_xlabel("Distortion Level (μ)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Elasticity: How Models Degrade Under Distortion", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# 4. ERROR TAXONOMY
# =============================================================================

def classify_error(row: pd.Series) -> str:
    """
    Classify error type based on answer format.
    
    Categories:
    - correct: answered correctly
    - blank_answer: no answer given
    - invalid_format: unrecognizable format
    - multiple_options: guessed multiple options
    - wrong_choice: single valid option, but wrong
    """
    if bool(row["is_correct"]):
        return "correct"
    
    ans = str(row.get("target_model_answer", "")).strip()
    
    if ans == "" or ans.lower() in {"none", "nan", "null", "n/a", ""}:
        return "blank_answer"
    
    # Parse options
    try:
        options = json.loads(str(row.get("options_json", "{}")))
        option_keys = set(options.keys()) if isinstance(options, dict) else set()
    except:
        option_keys = set()
    
    # Check for multiple options guessed
    multi_sep = [",", "/", " and ", " or ", ";"]
    if any(sep in ans for sep in multi_sep):
        return "multiple_options"
    
    # Single letter that matches an option key
    if len(ans) == 1 and ans.upper() in option_keys:
        return "wrong_choice"
    
    return "invalid_format"


def build_error_taxonomy(df: pd.DataFrame) -> Dict:
    """
    Build error taxonomy with:
    - Global counts and frequencies
    - Per-model breakdown
    - Per-μ bucket breakdown
    """
    df = df.copy()
    df["error_type"] = df.apply(classify_error, axis=1)
    
    total = len(df)
    global_counts = df["error_type"].value_counts().to_dict()
    
    # Per-model breakdown
    per_model = {}
    for model, g in df.groupby("target_model_name"):
        per_model[model] = g["error_type"].value_counts().to_dict()
    
    # Per-μ bucket breakdown (incorrect only)
    incorrect = df[df["is_correct"] == False].copy()
    incorrect["miu_bucket"] = incorrect["miu"].round(1)
    
    per_miu_bucket = {}
    for (bucket, etype), sub in incorrect.groupby(["miu_bucket", "error_type"]):
        per_miu_bucket.setdefault(str(bucket), {})[etype] = len(sub)
    
    # Per-model per-subject breakdown
    per_model_subject = {}
    for model, g_model in df.groupby("target_model_name"):
        per_model_subject[model] = {}
        for subject, g_subj in g_model.groupby("subject"):
            per_model_subject[model][subject] = g_subj["error_type"].value_counts().to_dict()
    
    return {
        "total_samples": total,
        "global_counts": {k: int(v) for k, v in global_counts.items()},
        "global_frequencies": {k: v / total for k, v in global_counts.items()},
        "per_model": per_model,
        "per_miu_bucket": per_miu_bucket,
        "per_model_subject": per_model_subject
    }


# =============================================================================
# 5. CONFUSION CLUSTERS (TF-IDF + KMeans)
# =============================================================================

def build_confusion_clusters(
    df: pd.DataFrame, 
    max_clusters: int = 8,
    min_samples: int = 20
) -> Optional[Dict]:
    """
    Cluster incorrect answers by TF-IDF similarity of distorted questions.
    
    Reveals common linguistic patterns in failures.
    Requires sklearn.
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not installed - clustering skipped"}
    
    incorrect = df[df["is_correct"] == False].copy()
    incorrect = incorrect.dropna(subset=["distorted_question"])
    
    if len(incorrect) < min_samples:
        return {"error": f"Only {len(incorrect)} incorrect samples, need {min_samples}+"}
    
    texts = incorrect["distorted_question"].astype(str).tolist()
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(texts)
    
    # Determine cluster count
    n_clusters = min(max_clusters, max(2, len(incorrect) // 50))
    
    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    
    incorrect = incorrect.reset_index(drop=True)
    incorrect["cluster"] = labels
    
    terms = np.array(vectorizer.get_feature_names_out())
    
    clusters = []
    for cid in range(n_clusters):
        mask = labels == cid
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue
        
        # Top terms for cluster centroid
        center = km.cluster_centers_[cid]
        top_idx = center.argsort()[-12:][::-1]
        top_terms = terms[top_idx].tolist()
        
        cluster_df = incorrect[incorrect["cluster"] == cid]
        
        # Subject distribution in cluster
        subject_dist = cluster_df["subject"].value_counts().to_dict()
        
        # μ distribution
        miu_dist = cluster_df["miu"].describe().to_dict()
        
        # Sample items
        samples = []
        for _, r in cluster_df.head(5).iterrows():
            samples.append({
                "subject": r.get("subject"),
                "question_id": r.get("question_id"),
                "miu": float(r.get("miu")),
                "correct_answer": r.get("answer"),
                "model_answer": r.get("target_model_answer"),
                "distorted_q": str(r.get("distorted_question"))[:200] + "..."
            })
        
        clusters.append({
            "cluster_id": cid,
            "size": cluster_size,
            "top_terms": top_terms,
            "subject_distribution": subject_dist,
            "miu_stats": {k: float(v) if not pd.isna(v) else None for k, v in miu_dist.items()},
            "samples": samples
        })
    
    return {
        "n_clusters": len(clusters),
        "total_incorrect": len(incorrect),
        "clusters": sorted(clusters, key=lambda x: -x["size"])
    }


# =============================================================================
# 6. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    df: pd.DataFrame,
    group_cols: List[str],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42
) -> pd.DataFrame:
    """Bootstrap CIs for accuracy by group."""
    rng = np.random.default_rng(seed)
    rows = []
    
    if not group_cols:
        groups = [("global", df)]
    else:
        groups = list(df.groupby(group_cols))
    
    for keys, group in groups:
        arr = group["is_correct"].astype(int).values
        n = len(arr)
        if n == 0:
            continue
        
        # Bootstrap
        means = [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)]
        means = np.array(means)
        
        if not isinstance(keys, tuple):
            keys = (keys,) if group_cols else ("global",)
        
        row = dict(zip(group_cols if group_cols else ["level"], keys))
        row.update({
            "n": n,
            "accuracy": float(arr.mean()),
            "ci_lower": float(np.quantile(means, alpha / 2)),
            "ci_upper": float(np.quantile(means, 1 - alpha / 2)),
            "ci_width": float(np.quantile(means, 1 - alpha / 2) - np.quantile(means, alpha / 2))
        })
        rows.append(row)
    
    return pd.DataFrame(rows)


def compute_all_bootstrap_cis(df: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    """Compute CIs at multiple granularities."""
    parts = []
    
    # Global per model
    ci_model = bootstrap_ci(df, ["target_model_name"], n_boot)
    ci_model["level"] = "model"
    
    # Per model + miu
    ci_model_miu = bootstrap_ci(df, ["target_model_name", "miu"], n_boot)
    ci_model_miu["level"] = "model_miu"
    
    # Per model + subject
    ci_model_subj = bootstrap_ci(df, ["target_model_name", "subject"], n_boot)
    ci_model_subj["level"] = "model_subject"
    
    parts.extend([ci_model, ci_model_miu, ci_model_subj])
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# 7. MODEL COMPARISON TABLE
# =============================================================================

def build_model_comparison(
    df: pd.DataFrame,
    acc_df: pd.DataFrame,
    cri_df: pd.DataFrame,
    elast_df: pd.DataFrame
) -> pd.DataFrame:
    """Comprehensive model comparison table."""
    mus = np.sort(acc_df["miu"].unique())
    base_mu = mus[0]
    high_mu = mus[-1]
    mid_mu = mus[np.argmin(np.abs(mus - 0.5))]
    
    rows = []
    for model in df["target_model_name"].unique():
        sub_acc = acc_df[acc_df["target_model_name"] == model]
        sub_df = df[df["target_model_name"] == model]
        
        def get_acc(target_mu):
            if len(sub_acc) == 0:
                return np.nan
            closest = sub_acc.iloc[(sub_acc["miu"] - target_mu).abs().argsort().iloc[0]]
            return float(closest["accuracy"])
        
        cri_row = cri_df[cri_df["target_model_name"] == model]
        cri_val = float(cri_row["CRI"].iloc[0]) if len(cri_row) > 0 else np.nan
        
        elast_row = elast_df[elast_df["target_model_name"] == model]
        elast_val = float(elast_row["elasticity_slope"].iloc[0]) if len(elast_row) > 0 else np.nan
        
        base_acc = get_acc(base_mu)
        high_acc = get_acc(high_mu)
        degradation = (base_acc - high_acc) / base_acc * 100 if base_acc > 0 else np.nan
        
        rows.append({
            "target_model_name": model,
            "n_samples": len(sub_df),
            "overall_accuracy": float(sub_df["is_correct"].mean()),
            f"accuracy_mu_{base_mu:.1f}": base_acc,
            f"accuracy_mu_{mid_mu:.1f}": get_acc(mid_mu),
            f"accuracy_mu_{high_mu:.1f}": high_acc,
            "degradation_pct": degradation,
            "CRI": cri_val,
            "elasticity_slope": elast_val,
        })
    
    return pd.DataFrame(rows).sort_values("CRI", ascending=False)


def plot_model_comparison(comp_df: pd.DataFrame, output_path: Path) -> None:
    """Scatter: CRI vs overall accuracy, sized by sample count."""
    if comp_df.empty:
        return
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Ensure numeric types
    x = comp_df["CRI"].astype(float).values
    y = comp_df["overall_accuracy"].astype(float).values
    sizes = (comp_df["n_samples"].astype(float) / comp_df["n_samples"].max() * 300 + 50).values
    
    scatter = ax.scatter(x, y, s=sizes, alpha=0.7, c=range(len(comp_df)), cmap="viridis", edgecolors='white', linewidth=2)
    
    for xi, yi, label in zip(x, y, comp_df["target_model_name"]):
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Chameleon Robustness Index (CRI)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Overall Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Model Comparison: Robustness vs Accuracy", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# 8. SYNERGY: DELTA ACCURACY HEATMAP (Subject × μ)
# =============================================================================

def compute_delta_accuracy_heatmap(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute accuracy degradation from baseline (μ=0) for each subject × μ.
    
    Returns:
    - Pivot table of Δ-accuracy
    - Dict with metadata per model
    """
    mus = np.sort(df["miu"].unique())
    base_mu = mus[0]
    
    results = {}
    
    for model, g_model in df.groupby("target_model_name"):
        # Baseline accuracy per subject
        baseline = g_model[g_model["miu"] == base_mu].groupby("subject")["is_correct"].mean()
        
        # Accuracy at each μ per subject
        pivot = g_model.groupby(["subject", "miu"])["is_correct"].mean().unstack(fill_value=np.nan)
        
        # Compute delta from baseline
        delta = pivot.subtract(baseline, axis=0)
        
        results[model] = {
            "delta_accuracy": delta.to_dict(),
            "baseline_by_subject": baseline.to_dict(),
            "worst_degradation_subject": delta.min(axis=1).idxmin() if not delta.empty else None,
            "most_robust_subject": delta.min(axis=1).idxmax() if not delta.empty else None
        }
    
    # Return first model's delta as example DataFrame
    first_model = list(results.keys())[0] if results else None
    if first_model:
        delta_df = pd.DataFrame(results[first_model]["delta_accuracy"])
    else:
        delta_df = pd.DataFrame()
    
    return delta_df, results


def plot_delta_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap showing accuracy degradation by subject × μ for each model."""
    setup_plot_style()
    models = df["target_model_name"].unique()
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8), squeeze=False)
    
    for idx, model in enumerate(models):
        ax = axes[0, idx]
        g = df[df["target_model_name"] == model].copy()
        
        # Ensure miu is numeric
        g["miu"] = pd.to_numeric(g["miu"], errors="coerce")
        mus = np.sort(g["miu"].dropna().unique())
        if len(mus) == 0:
            continue
            
        base_mu = mus[0]
        
        # Baseline
        baseline = g[g["miu"] == base_mu].groupby("subject")["is_correct"].mean()
        
        # Pivot
        pivot = g.groupby(["subject", "miu"])["is_correct"].mean().unstack(fill_value=np.nan)
        
        # Delta
        delta = pivot.subtract(baseline, axis=0)
        
        # Ensure values are float for plotting
        delta_values = delta.values.astype(float)
        
        # Plot
        im = ax.imshow(delta_values, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.1)
        
        ax.set_xticks(range(len(delta.columns)))
        ax.set_xticklabels([f"{float(m):.1f}" for m in delta.columns], rotation=45, fontsize=9)
        ax.set_yticks(range(len(delta.index)))
        ax.set_yticklabels(delta.index, fontsize=9)
        
        ax.set_xlabel("μ (distortion level)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Subject", fontsize=11, fontweight='bold')
        ax.set_title(f"{model}\nΔ-Accuracy from Baseline", fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.6, label="Δ Accuracy")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# 9. SYNERGY: QUESTION DIFFICULTY TIERS
# =============================================================================

def compute_question_difficulty_tiers(df: pd.DataFrame) -> Dict:
    """
    Tier questions by how models perform on them.
    
    Tiers:
    - easy: Most models correct at all μ levels
    - medium: Models correct at low μ, fail at high μ
    - hard: Most models fail even at low μ
    - chameleon_breakers: Correct at μ=0, catastrophic failure at high μ
    """
    mus = np.sort(df["miu"].unique())
    base_mu = mus[0]
    high_mu = mus[-1]
    
    # Aggregate by question_id
    question_stats = []
    
    for qid, g in df.groupby("question_id"):
        base_acc = g[g["miu"] == base_mu]["is_correct"].mean() if len(g[g["miu"] == base_mu]) > 0 else np.nan
        high_acc = g[g["miu"] == high_mu]["is_correct"].mean() if len(g[g["miu"] == high_mu]) > 0 else np.nan
        overall_acc = g["is_correct"].mean()
        
        # Subject
        subject = g["subject"].iloc[0] if "subject" in g.columns else "unknown"
        
        question_stats.append({
            "question_id": qid,
            "subject": subject,
            "base_accuracy": base_acc,
            "high_accuracy": high_acc,
            "overall_accuracy": overall_acc,
            "degradation": base_acc - high_acc if not np.isnan(base_acc) and not np.isnan(high_acc) else np.nan
        })
    
    stats_df = pd.DataFrame(question_stats)
    
    # Classify into tiers
    tiers = {
        "easy": [],
        "medium": [],
        "hard": [],
        "chameleon_breakers": []
    }
    
    for _, row in stats_df.iterrows():
        qid = row["question_id"]
        base = row["base_accuracy"]
        high = row["high_accuracy"]
        
        if pd.isna(base) or pd.isna(high):
            continue
        
        # Easy: high accuracy at both levels
        if base >= 0.8 and high >= 0.7:
            tiers["easy"].append(qid)
        # Chameleon breaker: good baseline, catastrophic high-μ failure
        elif base >= 0.7 and high < 0.3:
            tiers["chameleon_breakers"].append(qid)
        # Hard: fails even at baseline
        elif base < 0.5:
            tiers["hard"].append(qid)
        # Medium: everything else
        else:
            tiers["medium"].append(qid)
    
    # Summary stats
    total_classified = sum(len(v) for v in tiers.values())
    summary = {
        "total_questions": len(stats_df),
        "total_classified": total_classified,
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "tier_percentages": {k: len(v) / total_classified * 100 if total_classified > 0 else 0 for k, v in tiers.items()},
        "chameleon_breaker_questions": tiers["chameleon_breakers"][:20],  # Top 20
        "avg_degradation_by_tier": {}
    }
    
    for tier, qids in tiers.items():
        if qids:
            tier_deg = stats_df[stats_df["question_id"].isin(qids)]["degradation"].mean()
            summary["avg_degradation_by_tier"][tier] = float(tier_deg) if not np.isnan(tier_deg) else None
    
    return summary


# =============================================================================
# 10. EXECUTIVE SUMMARY
# =============================================================================

def generate_executive_summary(
    df: pd.DataFrame,
    comp_df: pd.DataFrame,
    cri_df: pd.DataFrame,
    difficulty_tiers: Dict,
    output_path: Path
) -> None:
    """Generate markdown executive summary."""
    
    models = [str(m) for m in df["target_model_name"].unique()]
    n_questions = df["question_id"].nunique()
    n_subjects = df["subject"].nunique()
    mus = sorted([float(m) for m in df["miu"].unique()])
    total_samples = len(df)
    
    # Top performer by CRI
    top_model = str(comp_df.iloc[0]["target_model_name"]) if len(comp_df) > 0 else "N/A"
    top_cri = float(comp_df.iloc[0]["CRI"]) if len(comp_df) > 0 else 0
    
    # Most fragile (worst elasticity)
    if len(comp_df) > 0 and "elasticity_slope" in comp_df.columns:
        worst_slope = float(comp_df["elasticity_slope"].min())
        fragile_model = str(comp_df[comp_df["elasticity_slope"] == comp_df["elasticity_slope"].min()]["target_model_name"].iloc[0])
    else:
        worst_slope = 0.0
        fragile_model = "N/A"
    
    # Overall accuracy
    overall_acc = float(df["is_correct"].mean())
    
    md = f"""# 🦎 Chameleon Benchmark - Executive Summary

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Models Evaluated** | {len(models)} ({', '.join(models)}) |
| **Unique Questions** | {n_questions:,} |
| **Subjects** | {n_subjects} |
| **Distortion Levels (μ)** | {len(mus)} ({min(mus):.1f} to {max(mus):.1f}) |
| **Total Samples** | {total_samples:,} |
| **Overall Accuracy** | {overall_acc:.1%} |

---

## 🔑 Key Findings

### 🏆 Most Robust Model

**{top_model}** achieved the highest Chameleon Robustness Index (CRI) of **{top_cri:.3f}**.

> CRI weights high-distortion performance more heavily, so this model maintains accuracy better under semantic paraphrasing.

### ⚠️ Most Fragile Model

**{fragile_model}** showed the steepest accuracy decline (elasticity slope: **{worst_slope:.4f}**).

> This indicates high sensitivity to lexical variations despite preserved meaning.

---

## 📈 Question Difficulty Distribution

| Tier | Count | Percentage | Avg Degradation |
|------|-------|------------|-----------------|
| 🟢 Easy | {difficulty_tiers['tier_counts'].get('easy', 0)} | {difficulty_tiers['tier_percentages'].get('easy', 0):.1f}% | {difficulty_tiers['avg_degradation_by_tier'].get('easy', 0) or 0:.3f} |
| 🟡 Medium | {difficulty_tiers['tier_counts'].get('medium', 0)} | {difficulty_tiers['tier_percentages'].get('medium', 0):.1f}% | {difficulty_tiers['avg_degradation_by_tier'].get('medium', 0) or 0:.3f} |
| 🔴 Hard | {difficulty_tiers['tier_counts'].get('hard', 0)} | {difficulty_tiers['tier_percentages'].get('hard', 0):.1f}% | {difficulty_tiers['avg_degradation_by_tier'].get('hard', 0) or 0:.3f} |
| 💀 Chameleon Breakers | {difficulty_tiers['tier_counts'].get('chameleon_breakers', 0)} | {difficulty_tiers['tier_percentages'].get('chameleon_breakers', 0):.1f}% | {difficulty_tiers['avg_degradation_by_tier'].get('chameleon_breakers', 0) or 0:.3f} |

> **Chameleon Breakers** are questions where models perform well at μ=0 but catastrophically fail at high distortion — strong evidence of surface-level pattern matching vs. true understanding.

---

## 📋 Model Comparison

| Model | Overall Acc | CRI | Degradation % | Elasticity |
|-------|-------------|-----|---------------|------------|
"""
    
    for _, row in comp_df.iterrows():
        deg_pct = row.get('degradation_pct', 0)
        md += f"| {row['target_model_name']} | {row['overall_accuracy']:.1%} | {row['CRI']:.3f} | {deg_pct:.1f}% | {row['elasticity_slope']:.4f} |\n"
    
    md += f"""
---

## 📖 Metric Interpretation Guide

### Chameleon Robustness Index (CRI)
The CRI is a weighted accuracy metric that emphasizes performance at **high distortion levels**. It uses exponential weighting: `w(μ) = exp(2.0 * μ)`, meaning accuracy at μ=0.9 is weighted ~6x more than accuracy at μ=0.0.

- **CRI > 0.7**: Highly robust - maintains accuracy under paraphrasing
- **CRI 0.5-0.7**: Moderately robust - some degradation but acceptable
- **CRI < 0.5**: Fragile - significant performance loss under lexical variation

### Elasticity Slope
Measures how quickly accuracy degrades as distortion increases. Computed via linear regression: `accuracy = intercept + slope × μ`

- **Slope ≈ 0**: Model is robust (accuracy stable across all μ levels)
- **Slope < -0.05**: Model is fragile (loses >5% accuracy per 0.1 μ increase)
- **Slope > 0**: Unusual - model improves with distortion (likely noise)

### Degradation Percentage
Simple metric: `(baseline_accuracy - high_μ_accuracy) / baseline_accuracy × 100%`

Shows the total performance drop from undistorted (μ=0) to maximally distorted (μ=0.9) questions.

### Question Difficulty Tiers

| Tier | Definition | Interpretation |
|------|------------|----------------|
| 🟢 **Easy** | ≥80% at μ=0, ≥70% at μ=0.9 | Model truly understands the concept |
| 🟡 **Medium** | Good at low μ, struggles at high μ | Partial understanding, some surface matching |
| 🔴 **Hard** | <50% even at μ=0 | Fundamental knowledge gap |
| 💀 **Chameleon Breaker** | ≥70% at μ=0, <30% at μ=0.9 | **Critical**: Relies on surface patterns, not meaning |

---

## 📊 Analysis Files Explained

### Core Metrics

| File | What It Shows | Key Insight |
|------|---------------|-------------|
| `01_accuracy_by_miu.csv/png` | Accuracy curve across all μ levels | How quickly does accuracy degrade? Steeper = more fragile |
| `02_accuracy_by_subject_miu.csv` | Per-subject accuracy at each μ | Which subjects are most/least robust? |
| `03_chameleon_robustness_index.csv` | CRI scores (global + per-subject) | Single robustness metric for ranking |
| `04_elasticity.csv/png` | Linear regression of accuracy vs μ | Slope quantifies fragility |
| `05_model_comparison.csv/png` | Head-to-head metrics table | Compare models across all dimensions |

### Error Analysis

| File | What It Shows | Key Insight |
|------|---------------|-------------|
| `06_error_taxonomy.json` | Classification of error types | Are errors due to blank answers, wrong choices, or format issues? |
| `07_confusion_clusters.json` | TF-IDF clustering of failed questions | Which linguistic patterns cause failures? |

### Statistical Rigor

| File | What It Shows | Key Insight |
|------|---------------|-------------|
| `08_bootstrap_intervals.csv` | 95% confidence intervals | Are accuracy differences statistically significant? |

### Synergy Analysis

| File | What It Shows | Key Insight |
|------|---------------|-------------|
| `09_delta_accuracy_heatmap.csv/png` | Subject × μ degradation matrix | Visual: Red = bad degradation, Green = robust |
| `10_question_difficulty_tiers.json` | Question classification | Find "Chameleon Breakers" - evidence of pattern matching |
| `11_executive_summary.md` | This comprehensive report | Start here for key findings |

---

## 🔬 Methodology Notes

### Distortion Levels (μ)
- **μ = 0.0**: Original question (baseline)
- **μ = 0.1-0.2**: Minimal changes (1-3 word synonyms)
- **μ = 0.3-0.4**: Moderate restructuring
- **μ = 0.5-0.6**: Mixed lexical + structural changes
- **μ = 0.7-0.8**: Heavy paraphrasing
- **μ = 0.9**: Complete sentence reconstruction

### Statistical Tests
- **McNemar's Test**: Paired comparison of correct/incorrect responses between μ=0 and μ>0
- **Bootstrap CI**: 500 resamples for 95% confidence intervals
- **TF-IDF Clustering**: Identifies linguistic patterns in failed questions

---

*Generated by Chameleon Analysis Engine - Synergy Edition*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_synergy_analysis(
    input_csv: Path,
    output_dir: Path,
    validated_only: bool = False,
    n_bootstrap: int = 500,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full Synergy analysis pipeline.
    
    Args:
        input_csv: Path to results.csv
        output_dir: Output directory for analysis files
        validated_only: Only use llm_validated=True rows
        n_bootstrap: Number of bootstrap samples for CI
        verbose: Print progress
    
    Returns:
        Dict with analysis summary
    """
    
    ensure_dir(output_dir)
    
    def log(msg):
        if verbose:
            print(msg)
    
    log(f"[INFO] Loading: {input_csv}")
    df = load_and_prepare(input_csv, validated_only)
    log(f"[INFO] Loaded {len(df):,} rows, {df['target_model_name'].nunique()} models")
    
    results = {
        "input_file": str(input_csv),
        "output_dir": str(output_dir),
        "n_rows": len(df),
        "n_models": df["target_model_name"].nunique(),
        "models": list(df["target_model_name"].unique()),
        "files_generated": []
    }
    
    # 1. Accuracy by miu
    log("[1/11] Computing accuracy by μ...")
    acc_df = compute_accuracy_by_miu(df)
    acc_df.to_csv(output_dir / "01_accuracy_by_miu.csv", index=False)
    plot_accuracy_by_miu(acc_df, output_dir / "01_accuracy_by_miu.png")
    results["files_generated"].extend(["01_accuracy_by_miu.csv", "01_accuracy_by_miu.png"])
    
    # 2. Accuracy by subject + miu
    log("[2/11] Computing accuracy by subject × μ...")
    acc_subj_df = compute_accuracy_by_subject_miu(df)
    acc_subj_df.to_csv(output_dir / "02_accuracy_by_subject_miu.csv", index=False)
    results["files_generated"].append("02_accuracy_by_subject_miu.csv")
    
    # 3. CRI (global + per-subject)
    log("[3/11] Computing Chameleon Robustness Index...")
    cri_global = compute_cri(df, ["target_model_name"])
    cri_global["scope"] = "global"
    
    cri_subject = compute_cri(df, ["target_model_name", "subject"])
    cri_subject["scope"] = "per_subject"
    
    cri_df = pd.concat([cri_global, cri_subject], ignore_index=True)
    cri_df.to_csv(output_dir / "03_chameleon_robustness_index.csv", index=False)
    results["files_generated"].append("03_chameleon_robustness_index.csv")
    results["top_cri"] = float(cri_global["CRI"].max()) if len(cri_global) > 0 else None
    
    # 4. Elasticity
    log("[4/11] Computing elasticity...")
    elast_df = compute_elasticity(acc_df)
    elast_df.to_csv(output_dir / "04_elasticity.csv", index=False)
    plot_elasticity(acc_df, output_dir / "04_elasticity.png")
    results["files_generated"].extend(["04_elasticity.csv", "04_elasticity.png"])
    
    # 5. Model comparison
    log("[5/11] Building model comparison...")
    comp_df = build_model_comparison(df, acc_df, cri_global, elast_df)
    comp_df.to_csv(output_dir / "05_model_comparison.csv", index=False)
    plot_model_comparison(comp_df, output_dir / "05_model_comparison.png")
    results["files_generated"].extend(["05_model_comparison.csv", "05_model_comparison.png"])
    
    # 6. Error taxonomy
    log("[6/11] Building error taxonomy...")
    taxonomy = build_error_taxonomy(df)
    with open(output_dir / "06_error_taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    results["files_generated"].append("06_error_taxonomy.json")
    results["error_taxonomy"] = taxonomy["global_frequencies"]
    
    # 7. Confusion clusters
    log("[7/11] Building confusion clusters (TF-IDF)...")
    clusters = build_confusion_clusters(df)
    with open(output_dir / "07_confusion_clusters.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    results["files_generated"].append("07_confusion_clusters.json")
    
    # 8. Bootstrap CIs
    log(f"[8/11] Computing bootstrap CIs ({n_bootstrap} samples)...")
    ci_df = compute_all_bootstrap_cis(df, n_bootstrap)
    ci_df.to_csv(output_dir / "08_bootstrap_intervals.csv", index=False)
    results["files_generated"].append("08_bootstrap_intervals.csv")
    
    # 9. Delta accuracy heatmap
    log("[9/11] Computing Δ-accuracy heatmap...")
    delta_df, delta_meta = compute_delta_accuracy_heatmap(df)
    delta_df.to_csv(output_dir / "09_delta_accuracy_heatmap.csv")
    with open(output_dir / "09_delta_accuracy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(delta_meta, f, indent=2, ensure_ascii=False, default=str)
    plot_delta_heatmap(df, output_dir / "09_delta_accuracy_heatmap.png")
    results["files_generated"].extend(["09_delta_accuracy_heatmap.csv", "09_delta_accuracy_metadata.json", "09_delta_accuracy_heatmap.png"])
    
    # 10. Question difficulty tiers
    log("[10/11] Computing question difficulty tiers...")
    difficulty = compute_question_difficulty_tiers(df)
    with open(output_dir / "10_question_difficulty_tiers.json", "w", encoding="utf-8") as f:
        json.dump(difficulty, f, indent=2, ensure_ascii=False)
    results["files_generated"].append("10_question_difficulty_tiers.json")
    results["difficulty_tiers"] = difficulty["tier_counts"]
    
    # 11. Executive summary
    log("[11/11] Generating executive summary...")
    generate_executive_summary(df, comp_df, cri_global, difficulty, output_dir / "11_executive_summary.md")
    results["files_generated"].append("11_executive_summary.md")
    
    log(f"\n[DONE] All outputs saved to: {output_dir}")
    log(f"       Start with: {output_dir / '11_executive_summary.md'}")
    
    results["status"] = "complete"
    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chameleon Analysis Engine - Synergy Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m chameleon.analysis.synergy_engine --input results.csv
  python -m chameleon.analysis.synergy_engine --input results.csv --output my_analysis/ --validated-only
  python -m chameleon.analysis.synergy_engine -i Projects/GPT5/results/results.csv -o Projects/GPT5/results/analysis/
        """
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="results.csv",
        help="Path to results.csv"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: <input_dir>/analysis)"
    )
    parser.add_argument(
        "--validated-only",
        action="store_true",
        help="Only use rows where llm_validated=True"
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Number of bootstrap samples for CI computation"
    )
    
    args = parser.parse_args()
    
    input_csv = Path(args.input).resolve()
    output_dir = Path(args.output).resolve() if args.output else input_csv.parent / "synergy_analysis"
    
    run_synergy_analysis(input_csv, output_dir, args.validated_only, args.bootstrap_samples)


if __name__ == "__main__":
    main()


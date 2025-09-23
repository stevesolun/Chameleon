#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis for Chameleon Project
======================================================

This script performs advanced statistical testing including:
1. McNemar's test for paired comparisons between distortion levels
2. Statistical significance testing for performance differences  
3. Subject-specific significance analysis
4. Confidence intervals for degradation patterns

McNemar's test is perfect for our use case because:
- We have paired binary outcomes (correct/incorrect)
- Same questions tested across different conditions (μ levels)
- Want to test if accuracy differences are statistically significant
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def mcnemar_test_paired(group1_correct, group2_correct):
    """
    Perform McNemar's test for paired binary outcomes.
    
    Args:
        group1_correct: Boolean array of correct answers for condition 1
        group2_correct: Boolean array of correct answers for condition 2
        
    Returns:
        dict: Test results with statistic, p-value, and interpretation
    """
    # Create contingency table
    # McNemar's test focuses on the discordant pairs (b and c)
    # Table: [[a, b], [c, d]] where:
    # a = both correct, b = group1 correct & group2 wrong
    # c = group1 wrong & group2 correct, d = both wrong
    
    both_correct = np.sum(group1_correct & group2_correct)  # a
    group1_only = np.sum(group1_correct & ~group2_correct)  # b  
    group2_only = np.sum(~group1_correct & group2_correct)  # c
    both_wrong = np.sum(~group1_correct & ~group2_correct)  # d
    
    # Contingency table for McNemar's test
    table = np.array([[both_correct, group1_only],
                     [group2_only, both_wrong]])
    
    # Perform McNemar's test
    try:
        result = mcnemar(table, exact=False, correction=True)
        statistic = result.statistic
        p_value = result.pvalue
    except ValueError:
        # If too few discordant pairs, use exact test
        try:
            result = mcnemar(table, exact=True)
            statistic = result.statistic  
            p_value = result.pvalue
        except:
            statistic = np.nan
            p_value = np.nan
    
    # Calculate effect size (odds ratio for discordant pairs)
    if group2_only > 0:
        odds_ratio = group1_only / group2_only
    else:
        odds_ratio = np.inf if group1_only > 0 else 1.0
    
    # Interpretation
    if pd.isna(p_value):
        significance = "Unable to calculate"
    elif p_value < 0.001:
        significance = "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "Very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "Significant (p < 0.05)"
    elif p_value < 0.1:
        significance = "Marginally significant (p < 0.1)"
    else:
        significance = "Not significant (p ≥ 0.1)"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'contingency_table': table,
        'discordant_pairs': group1_only + group2_only,
        'odds_ratio': odds_ratio,
        'significance': significance,
        'group1_accuracy': np.mean(group1_correct),
        'group2_accuracy': np.mean(group2_correct),
        'accuracy_difference': np.mean(group1_correct) - np.mean(group2_correct)
    }

def calculate_confidence_interval(accuracy, n, confidence=0.95):
    """
    Calculate confidence interval for accuracy using Wilson score interval.
    
    Args:
        accuracy: Proportion of correct answers
        n: Sample size
        confidence: Confidence level (default 0.95)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if n == 0:
        return (0, 0)
    
    z = stats.norm.ppf((1 + confidence) / 2)
    p = accuracy
    
    # Wilson score interval (more accurate for proportions)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    
    return (max(0, center - margin), min(1, center + margin))

def analyze_distortion_level_comparisons(df):
    """
    Perform McNemar's tests comparing each distortion level to baseline.
    
    Args:
        df: DataFrame with results
        
    Returns:
        DataFrame: Statistical test results
    """
    print('🔬 MCNEMAR\'S TEST: DISTORTION LEVELS vs BASELINE')
    print('=' * 60)
    
    results = []
    baseline_data = df[df['miu'] == 0.0]
    
    # Get baseline results indexed by question_id and subject for proper pairing
    baseline_correct = baseline_data.set_index(['question_id', 'subject'])['is_correct']
    
    for miu in sorted(df['miu'].unique()):
        if miu == 0.0:
            continue
            
        print(f'\\n📊 Testing μ={miu:.1f} vs Baseline (μ=0.0):')
        
        # Get distorted results for this miu level
        distorted_data = df[df['miu'] == miu]
        distorted_correct = distorted_data.set_index(['question_id', 'subject'])['is_correct']
        
        # Find common questions (should be all questions)
        common_questions = baseline_correct.index.intersection(distorted_correct.index)
        
        if len(common_questions) == 0:
            print(f'   ❌ No paired questions found for μ={miu:.1f}')
            continue
        
        # Get paired results
        baseline_paired = baseline_correct.loc[common_questions].astype(bool)
        distorted_paired = distorted_correct.loc[common_questions].astype(bool)
        
        # Perform McNemar's test
        test_result = mcnemar_test_paired(baseline_paired, distorted_paired)
        
        print(f'   📈 Baseline accuracy: {test_result["group1_accuracy"]:.3f}')
        print(f'   📉 μ={miu:.1f} accuracy: {test_result["group2_accuracy"]:.3f}')
        print(f'   📊 Accuracy difference: {test_result["accuracy_difference"]:.3f}')
        print(f'   🧮 McNemar statistic: {test_result["statistic"]:.3f}')
        print(f'   📋 P-value: {test_result["p_value"]:.6f}')
        print(f'   ✅ Result: {test_result["significance"]}')
        
        # Calculate confidence interval for the difference
        n = len(common_questions)
        baseline_ci = calculate_confidence_interval(test_result["group1_accuracy"], n)
        distorted_ci = calculate_confidence_interval(test_result["group2_accuracy"], n)
        
        results.append({
            'comparison': f'μ=0.0 vs μ={miu:.1f}',
            'miu_level': miu,
            'baseline_accuracy': test_result["group1_accuracy"],
            'distorted_accuracy': test_result["group2_accuracy"],
            'accuracy_difference': test_result["accuracy_difference"],
            'baseline_ci_lower': baseline_ci[0],
            'baseline_ci_upper': baseline_ci[1],
            'distorted_ci_lower': distorted_ci[0],
            'distorted_ci_upper': distorted_ci[1],
            'mcnemar_statistic': test_result["statistic"],
            'p_value': test_result["p_value"],
            'discordant_pairs': test_result["discordant_pairs"],
            'odds_ratio': test_result["odds_ratio"],
            'significance': test_result["significance"],
            'sample_size': n
        })
    
    return pd.DataFrame(results)

def analyze_subject_specific_significance(df):
    """
    Perform McNemar's tests for each subject comparing baseline to high distortion.
    
    Args:
        df: DataFrame with results
        
    Returns:
        DataFrame: Subject-specific statistical test results
    """
    print('\\n🔬 MCNEMAR\'S TEST: SUBJECT-SPECIFIC ANALYSIS')
    print('=' * 55)
    
    results = []
    
    for subject in sorted(df['subject'].unique()):
        print(f'\\n📚 Subject: {subject.replace("_", " ").title()}')
        
        subject_data = df[df['subject'] == subject]
        baseline_data = subject_data[subject_data['miu'] == 0.0]
        high_distortion_data = subject_data[subject_data['miu'] == 0.9]  # Use highest distortion
        
        if len(baseline_data) == 0 or len(high_distortion_data) == 0:
            print(f'   ❌ Insufficient data for {subject}')
            continue
        
        # Align by question_id for proper pairing
        baseline_correct = baseline_data.set_index('question_id')['is_correct']
        distorted_correct = high_distortion_data.set_index('question_id')['is_correct']
        
        common_questions = baseline_correct.index.intersection(distorted_correct.index)
        
        if len(common_questions) == 0:
            print(f'   ❌ No paired questions for {subject}')
            continue
        
        baseline_paired = baseline_correct.loc[common_questions].astype(bool)
        distorted_paired = distorted_correct.loc[common_questions].astype(bool)
        
        # Perform McNemar's test
        test_result = mcnemar_test_paired(baseline_paired, distorted_paired)
        
        # Calculate degradation
        degradation = (test_result["group1_accuracy"] - test_result["group2_accuracy"]) * 100
        
        print(f'   📈 Baseline accuracy: {test_result["group1_accuracy"]:.3f}')
        print(f'   📉 High distortion accuracy: {test_result["group2_accuracy"]:.3f}')
        print(f'   📊 Degradation: {degradation:.1f}%')
        print(f'   📋 P-value: {test_result["p_value"]:.6f}')
        print(f'   ✅ Result: {test_result["significance"]}')
        
        # Confidence intervals
        n = len(common_questions)
        baseline_ci = calculate_confidence_interval(test_result["group1_accuracy"], n)
        distorted_ci = calculate_confidence_interval(test_result["group2_accuracy"], n)
        
        results.append({
            'subject': subject,
            'subject_name': subject.replace('_', ' ').title(),
            'baseline_accuracy': test_result["group1_accuracy"],
            'high_distortion_accuracy': test_result["group2_accuracy"],
            'degradation_percent': degradation,
            'baseline_ci_lower': baseline_ci[0],
            'baseline_ci_upper': baseline_ci[1],
            'distorted_ci_lower': distorted_ci[0],
            'distorted_ci_upper': distorted_ci[1],
            'mcnemar_statistic': test_result["statistic"],
            'p_value': test_result["p_value"],
            'significance': test_result["significance"],
            'sample_size': n,
            'is_significant': test_result["p_value"] < 0.05 if not pd.isna(test_result["p_value"]) else False
        })
    
    return pd.DataFrame(results)

def analyze_pairwise_miu_comparisons(df):
    """
    Perform pairwise McNemar's tests between adjacent μ levels.
    
    Args:
        df: DataFrame with results
        
    Returns:
        DataFrame: Pairwise comparison results
    """
    print('\\n🔬 MCNEMAR\'S TEST: PAIRWISE μ LEVEL COMPARISONS')
    print('=' * 55)
    
    results = []
    miu_levels = sorted(df['miu'].unique())
    
    for i in range(len(miu_levels) - 1):
        miu1 = miu_levels[i]
        miu2 = miu_levels[i + 1]
        
        print(f'\\n📊 Testing μ={miu1:.1f} vs μ={miu2:.1f}:')
        
        data1 = df[df['miu'] == miu1]
        data2 = df[df['miu'] == miu2]
        
        # Align by question_id and subject
        correct1 = data1.set_index(['question_id', 'subject'])['is_correct']
        correct2 = data2.set_index(['question_id', 'subject'])['is_correct']
        
        common_questions = correct1.index.intersection(correct2.index)
        
        if len(common_questions) == 0:
            print(f'   ❌ No paired questions found')
            continue
        
        paired1 = correct1.loc[common_questions].astype(bool)
        paired2 = correct2.loc[common_questions].astype(bool)
        
        test_result = mcnemar_test_paired(paired1, paired2)
        
        print(f'   📈 μ={miu1:.1f} accuracy: {test_result["group1_accuracy"]:.3f}')
        print(f'   📉 μ={miu2:.1f} accuracy: {test_result["group2_accuracy"]:.3f}')
        print(f'   📊 Difference: {test_result["accuracy_difference"]:.3f}')
        print(f'   📋 P-value: {test_result["p_value"]:.6f}')
        print(f'   ✅ Result: {test_result["significance"]}')
        
        results.append({
            'comparison': f'μ={miu1:.1f} vs μ={miu2:.1f}',
            'miu1': miu1,
            'miu2': miu2,
            'miu1_accuracy': test_result["group1_accuracy"],
            'miu2_accuracy': test_result["group2_accuracy"],
            'accuracy_difference': test_result["accuracy_difference"],
            'mcnemar_statistic': test_result["statistic"],
            'p_value': test_result["p_value"],
            'significance': test_result["significance"],
            'sample_size': len(common_questions),
            'is_significant': test_result["p_value"] < 0.05 if not pd.isna(test_result["p_value"]) else False
        })
    
    return pd.DataFrame(results)

def create_statistical_visualizations(distortion_results, subject_results, pairwise_results, output_dir):
    """
    Create visualizations for statistical test results.
    """
    print('\\n📊 CREATING STATISTICAL VISUALIZATIONS')
    print('=' * 45)
    
    # 1. Distortion level significance plot
    plt.figure(figsize=(12, 8))
    x_pos = range(len(distortion_results))
    
    # Plot accuracy with confidence intervals
    plt.errorbar(x_pos, distortion_results['baseline_accuracy'], 
                yerr=[distortion_results['baseline_accuracy'] - distortion_results['baseline_ci_lower'],
                      distortion_results['baseline_ci_upper'] - distortion_results['baseline_accuracy']],
                label='Baseline (μ=0.0)', marker='o', capsize=5, linewidth=2)
    
    plt.errorbar(x_pos, distortion_results['distorted_accuracy'],
                yerr=[distortion_results['distorted_accuracy'] - distortion_results['distorted_ci_lower'],
                      distortion_results['distorted_ci_upper'] - distortion_results['distorted_accuracy']],
                label='Distorted', marker='s', capsize=5, linewidth=2)
    
    # Mark significant differences
    for i, row in distortion_results.iterrows():
        if row['p_value'] < 0.001:
            plt.text(i, row['distorted_accuracy'] - 0.02, '***', ha='center', fontweight='bold', fontsize=12)
        elif row['p_value'] < 0.01:
            plt.text(i, row['distorted_accuracy'] - 0.02, '**', ha='center', fontweight='bold', fontsize=12)
        elif row['p_value'] < 0.05:
            plt.text(i, row['distorted_accuracy'] - 0.02, '*', ha='center', fontweight='bold', fontsize=12)
    
    plt.xticks(x_pos, [f'μ={miu:.1f}' for miu in distortion_results['miu_level']])
    plt.ylabel('Accuracy', fontweight='bold', fontsize=12)
    plt.xlabel('Distortion Level', fontweight='bold', fontsize=12)
    plt.title('Statistical Significance of Distortion Effects\\n(* p<0.05, ** p<0.01, *** p<0.001)', 
              fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_significance_distortion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Subject-specific significance heatmap
    plt.figure(figsize=(14, 10))
    
    # Create significance matrix
    subjects = subject_results['subject_name'].tolist()
    p_values = subject_results['p_value'].tolist()
    degradations = subject_results['degradation_percent'].tolist()
    
    # Create data for heatmap
    heatmap_data = []
    significance_levels = []
    
    for i, (p_val, deg) in enumerate(zip(p_values, degradations)):
        heatmap_data.append([deg])
        if pd.isna(p_val):
            significance_levels.append('N/A')
        elif p_val < 0.001:
            significance_levels.append('p < 0.001')
        elif p_val < 0.01:
            significance_levels.append('p < 0.01')
        elif p_val < 0.05:
            significance_levels.append('p < 0.05')
        else:
            significance_levels.append('n.s.')
    
    # Create heatmap
    plt.imshow([[deg] for deg in degradations], cmap='Reds', aspect='auto')
    plt.colorbar(label='Degradation (%)', shrink=0.8)
    
    # Add significance annotations
    for i, sig in enumerate(significance_levels):
        color = 'white' if degradations[i] > 20 else 'black'
        plt.text(0, i, sig, ha='center', va='center', fontweight='bold', color=color)
    
    plt.yticks(range(len(subjects)), subjects, fontsize=10)
    plt.xticks([])
    plt.title('Subject-Specific Degradation Significance\\n(Baseline vs High Distortion μ=0.9)', 
              fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'subject_significance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✅ Statistical visualizations saved')

def generate_statistical_report(distortion_results, subject_results, pairwise_results, output_dir):
    """
    Generate comprehensive statistical report.
    """
    report_path = output_dir / 'Statistical_Analysis_Report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Chameleon Project: Statistical Analysis Report\\n\\n")
        f.write("## McNemar's Test Results for Paired Comparisons\\n\\n")
        f.write("This report presents comprehensive statistical analysis using McNemar's test ")
        f.write("for paired binary outcomes (correct/incorrect answers).\\n\\n")
        
        # Summary statistics
        f.write("## Executive Summary\\n\\n")
        significant_distortions = distortion_results[distortion_results['p_value'] < 0.05]
        significant_subjects = subject_results[subject_results['is_significant']]
        
        f.write(f"- **Total Distortion Levels Tested**: {len(distortion_results)}\\n")
        f.write(f"- **Significant Distortion Effects**: {len(significant_distortions)} ")
        f.write(f"({len(significant_distortions)/len(distortion_results)*100:.1f}%)\\n")
        f.write(f"- **Subjects with Significant Degradation**: {len(significant_subjects)} ")
        f.write(f"({len(significant_subjects)/len(subject_results)*100:.1f}%)\\n\\n")
        
        # Distortion level results
        f.write("## 1. Distortion Level Analysis (vs Baseline)\\n\\n")
        f.write("| μ Level | Baseline Acc | Distorted Acc | Difference | McNemar χ² | p-value | Significance |\\n")
        f.write("|---------|---------------|---------------|------------|-------------|---------|--------------|\\n")
        
        for _, row in distortion_results.iterrows():
            f.write(f"| {row['miu_level']:.1f} | {row['baseline_accuracy']:.3f} | ")
            f.write(f"{row['distorted_accuracy']:.3f} | {row['accuracy_difference']:+.3f} | ")
            f.write(f"{row['mcnemar_statistic']:.2f} | {row['p_value']:.6f} | {row['significance']} |\\n")
        
        # Subject-specific results
        f.write("\\n## 2. Subject-Specific Analysis (Baseline vs μ=0.9)\\n\\n")
        f.write("| Subject | Degradation | McNemar χ² | p-value | Significance |\\n")
        f.write("|---------|-------------|-------------|---------|--------------|\\n")
        
        subject_sorted = subject_results.sort_values('degradation_percent', ascending=False)
        for _, row in subject_sorted.iterrows():
            f.write(f"| {row['subject_name']} | {row['degradation_percent']:.1f}% | ")
            f.write(f"{row['mcnemar_statistic']:.2f} | {row['p_value']:.6f} | {row['significance']} |\\n")
        
        # Pairwise comparisons
        f.write("\\n## 3. Pairwise μ Level Comparisons\\n\\n")
        f.write("| Comparison | Accuracy Diff | McNemar χ² | p-value | Significance |\\n")
        f.write("|------------|---------------|-------------|---------|--------------|\\n")
        
        for _, row in pairwise_results.iterrows():
            f.write(f"| {row['comparison']} | {row['accuracy_difference']:+.3f} | ")
            f.write(f"{row['mcnemar_statistic']:.2f} | {row['p_value']:.6f} | {row['significance']} |\\n")
        
        f.write("\\n## Statistical Interpretation\\n\\n")
        f.write("**McNemar's Test** is used for paired binary data to test whether the marginal ")
        f.write("frequencies of correct/incorrect answers differ between two conditions.\\n\\n")
        f.write("- **Null Hypothesis**: No difference in accuracy between conditions\\n")
        f.write("- **Alternative Hypothesis**: Significant difference in accuracy\\n")
        f.write("- **Significance Levels**: * p<0.05, ** p<0.01, *** p<0.001\\n\\n")
        f.write("**Confidence Intervals** use Wilson score intervals for proportions.\\n")
    
    print(f'✅ Statistical report saved: {report_path}')

def main():
    """
    Main function to run comprehensive statistical analysis.
    """
    print('🧮 CHAMELEON PROJECT - COMPREHENSIVE STATISTICAL ANALYSIS')
    print('=' * 70)
    
    # Load data
    csv_path = 'distortions/chameleon_dataset.csv'
    print(f'📊 Loading dataset: {csv_path}')
    
    df = pd.read_csv(csv_path)
    print(f'   Total rows: {len(df):,}')
    print(f'   Subjects: {len(df["subject"].unique())}')
    print(f'   μ levels: {sorted(df["miu"].unique())}')
    
    # Ensure output directory exists
    output_dir = Path('analysis_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Distortion level comparisons
    distortion_results = analyze_distortion_level_comparisons(df)
    
    # 2. Subject-specific analysis
    subject_results = analyze_subject_specific_significance(df)
    
    # 3. Pairwise μ level comparisons
    pairwise_results = analyze_pairwise_miu_comparisons(df)
    
    # 4. Create visualizations
    create_statistical_visualizations(distortion_results, subject_results, pairwise_results, output_dir)
    
    # 5. Generate comprehensive report
    generate_statistical_report(distortion_results, subject_results, pairwise_results, output_dir)
    
    # Save detailed results
    distortion_results.to_csv(output_dir / 'mcnemar_distortion_results.csv', index=False)
    subject_results.to_csv(output_dir / 'mcnemar_subject_results.csv', index=False)
    pairwise_results.to_csv(output_dir / 'mcnemar_pairwise_results.csv', index=False)
    
    print('\\n🎉 COMPREHENSIVE STATISTICAL ANALYSIS COMPLETE!')
    print(f'📁 Results saved in: {output_dir.absolute()}')
    print('📊 Generated files:')
    print('   - mcnemar_distortion_results.csv')
    print('   - mcnemar_subject_results.csv') 
    print('   - mcnemar_pairwise_results.csv')
    print('   - statistical_significance_distortion.png')
    print('   - subject_significance_heatmap.png')
    print('   - Statistical_Analysis_Report.md')

if __name__ == '__main__':
    main()

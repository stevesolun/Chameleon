#!/usr/bin/env python3
"""
Chameleon Project - Comprehensive Visualization Generator
========================================================

This script creates all the analysis visualizations for the Chameleon project,
which studies GPT-5 performance degradation under lexical distortions.

The script generates:
1. Enhanced Performance Heatmap - Shows accuracy % by subject and distortion level (μ)
2. Enhanced Degradation Heatmap - Shows performance degradation from baseline
3. Comprehensive Degradation Analysis - 6-panel detailed analysis
4. Key Insights Summary - 4-panel highlights showing outliers and patterns

Requirements:
- pandas
- matplotlib
- seaborn
- numpy
- Pillow

Usage:
    python3 create_visualizations.py

Input:
    distortions/comprehensive_distortion_dataset_FINAL_20250922_015000.csv

Output:
    analysis_plots/ directory with all visualization files
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path):
    """
    Load the dataset and calculate performance metrics.
    
    Args:
        csv_path (str): Path to the CSV file with results
        
    Returns:
        tuple: (original_df, performance_df) where performance_df contains
               calculated accuracy and degradation metrics
    """
    print('📊 Loading and preparing data...')
    
    # Load the main dataset
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df):,} questions across {len(df["subject"].unique())} subjects')
    
    # Calculate performance metrics for each subject-miu combination
    performance_data = []
    
    for subject in df['subject'].unique():
        # Get baseline performance (μ=0.0) for this subject
        baseline_subset = df[(df['subject'] == subject) & (df['miu'] == 0.0)]
        baseline_acc = (baseline_subset['gpt5_answer'] == baseline_subset['correct_answer']).mean() * 100 if len(baseline_subset) > 0 else 0
        
        # Calculate performance at each distortion level
        for miu in sorted(df['miu'].unique()):
            subset = df[(df['subject'] == subject) & (df['miu'] == miu)]
            if len(subset) > 0:
                accuracy = (subset['gpt5_answer'] == subset['correct_answer']).mean() * 100
                degradation = baseline_acc - accuracy if miu > 0 else 0
                
                performance_data.append({
                    'subject': subject,
                    'miu': miu,
                    'accuracy': accuracy,
                    'baseline_accuracy': baseline_acc,
                    'degradation': degradation,
                    'question_count': len(subset)
                })
    
    performance_df = pd.DataFrame(performance_data)
    print(f'Calculated performance for {len(performance_df)} subject-miu combinations')
    
    return df, performance_df


def create_enhanced_degradation_heatmap(performance_df, output_dir):
    """
    Create an enhanced degradation heatmap showing performance drops from baseline.
    
    Args:
        performance_df (DataFrame): Performance data with degradation metrics
        output_dir (Path): Output directory for saving plots
    """
    print('📉 Creating enhanced degradation heatmap...')
    
    # Create pivot table for degradation data (exclude μ=0.0)
    degradation_data = performance_df[performance_df['miu'] > 0]
    degradation_pivot = degradation_data.pivot(index='subject', columns='miu', values='degradation')
    
    # Create the plot
    plt.figure(figsize=(12, 16))
    mask = degradation_pivot.isna()
    
    # Create heatmap with red color scheme for degradation
    sns.heatmap(degradation_pivot, 
                annot=True,           # Show values in cells
                fmt='.1f',            # Format numbers to 1 decimal place
                cmap='Reds',          # Red colormap (darker = worse degradation)
                center=10,            # Center at 10% degradation
                mask=mask,            # Mask missing values
                cbar_kws={'label': 'Performance Degradation (%)'},
                linewidths=0.5)       # Add grid lines
    
    # Customize labels and title
    plt.title('Performance Degradation from Baseline\n(Darker Red = Higher Degradation)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Distortion Level (μ)', fontsize=12, fontweight='bold')
    plt.ylabel('Academic Subject', fontsize=12, fontweight='bold')
    
    # Improve axis labels
    y_labels = [label.replace('_', ' ').title() for label in degradation_pivot.index]
    x_labels = [f'μ={x:.1f}' for x in degradation_pivot.columns]
    plt.xticks(np.arange(len(x_labels)) + 0.5, x_labels, fontsize=10)
    plt.yticks(np.arange(len(y_labels)) + 0.5, y_labels, fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_degradation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Enhanced degradation heatmap saved')

def create_individual_plots(performance_df, output_dir):
    """
    Create 6 individual plots with proper spacing and readability.
    
    Args:
        performance_df (DataFrame): Performance data
        output_dir (Path): Output directory for saving plots
    """
    print('📊 Creating 6 individual plots with proper spacing...')
    
    # Calculate common data
    subject_avg_degradation = performance_df.groupby('subject')['degradation'].mean().sort_values(ascending=False)
    miu_degradation = performance_df.groupby('miu')['degradation'].mean()
    
    # 1. Subject degradation ranking
    plt.figure(figsize=(12, 10))
    subject_avg_degradation.plot(kind='barh', color='lightcoral', figsize=(12, 10))
    plt.title('Average Performance Degradation by Subject', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Average Degradation (%)', fontweight='bold', fontsize=14)
    plt.ylabel('Academic Subject', fontweight='bold', fontsize=14)
    y_labels = [label.replace('_', ' ').title() for label in subject_avg_degradation.index]
    plt.yticks(range(len(y_labels)), y_labels, fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / '1_subject_degradation_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Degradation by μ level
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(miu_degradation)), miu_degradation.values, 
                   color=['lightblue' if x < 8 else 'orange' if x < 12 else 'red' for x in miu_degradation.values])
    plt.title('Average Degradation by Distortion Level', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Distortion Level (μ)', fontweight='bold', fontsize=14)
    plt.ylabel('Average Degradation (%)', fontweight='bold', fontsize=14)
    plt.xticks(range(len(miu_degradation)), [f'μ={x:.1f}' for x in miu_degradation.index], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, v in enumerate(miu_degradation.values):
        plt.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / '2_degradation_by_miu_level.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Degradation distribution histogram
    plt.figure(figsize=(12, 8))
    plt.hist(performance_df['degradation'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Performance Degradation', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Degradation (%)', fontweight='bold', fontsize=14)
    plt.ylabel('Frequency', fontweight='bold', fontsize=14)
    plt.axvline(performance_df['degradation'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {performance_df["degradation"].mean():.1f}%')
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '3_degradation_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Degradation progression lines
    plt.figure(figsize=(12, 8))
    vulnerable_subjects = subject_avg_degradation.head(5).index
    colors = ['red', 'orange', 'gold', 'lightcoral', 'pink']
    
    for i, subject in enumerate(vulnerable_subjects):
        subject_data = performance_df[performance_df['subject'] == subject].sort_values('miu')
        subject_data = subject_data[subject_data['miu'] > 0]  # Exclude μ=0
        plt.plot(subject_data['miu'], subject_data['degradation'], 
                marker='o', linewidth=3, markersize=8, 
                label=subject.replace('_', ' ').title(), color=colors[i])
    
    plt.xlabel('Distortion Level (μ)', fontweight='bold', fontsize=14)
    plt.ylabel('Degradation (%)', fontweight='bold', fontsize=14)
    plt.title('Degradation Progression: Most Vulnerable Subjects', fontweight='bold', fontsize=16, pad=20)
    plt.legend(fontsize=12, loc='upper left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '4_degradation_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Subject resilience ranking
    plt.figure(figsize=(12, 10))
    resilience_score = 100 - subject_avg_degradation  # Higher = more resilient
    resilience_score.sort_values(ascending=True).plot(kind='barh', color='gold', figsize=(12, 10))
    plt.title('Subject Resilience Ranking\n(Higher Score = More Resistant to Distortion)', 
              fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Resilience Score (100 - avg degradation)', fontweight='bold', fontsize=14)
    plt.ylabel('Academic Subject', fontweight='bold', fontsize=14)
    y_labels = [label.replace('_', ' ').title() for label in resilience_score.sort_values(ascending=True).index]
    plt.yticks(range(len(y_labels)), y_labels, fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / '5_subject_resilience_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Key distortion levels heatmap
    plt.figure(figsize=(10, 12))
    key_mius = [0.1, 0.3, 0.5, 0.7, 0.9]
    degradation_pivot = performance_df.pivot(index='subject', columns='miu', values='degradation')
    key_degradation = degradation_pivot[key_mius]
    
    sns.heatmap(key_degradation, 
                annot=True, 
                fmt='.1f', 
                cmap='Reds',
                cbar_kws={'label': 'Degradation (%)', 'shrink': 0.8},
                linewidths=0.5)
    plt.title('Performance Degradation at Key Distortion Levels', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Distortion Level (μ)', fontweight='bold', fontsize=14)
    plt.ylabel('Academic Subject', fontweight='bold', fontsize=14)
    plt.xticks(np.arange(len(key_mius)) + 0.5, [f'μ={x:.1f}' for x in key_mius], fontsize=12)
    y_labels = [label.replace('_', ' ').title() for label in key_degradation.index]
    plt.yticks(np.arange(len(y_labels)) + 0.5, y_labels, fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / '6_key_distortion_levels_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✅ All 6 individual plots saved with proper spacing')

def create_key_insights_summary(performance_df, output_dir):
    """
    Create a 4-panel summary highlighting key insights and outliers.
    
    Args:
        performance_df (DataFrame): Performance data
        output_dir (Path): Output directory for saving plots
    """
    print('🎯 Creating key insights summary...')
    
    # Set up 2x2 subplot grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top degradation subjects (worst performers)
    subject_avg_degradation = performance_df.groupby('subject')['degradation'].mean().sort_values(ascending=False).head(10)
    bars1 = ax1.barh(range(len(subject_avg_degradation)), subject_avg_degradation.values, 
                    color=['red' if x > 20 else 'orange' if x > 10 else 'yellow' for x in subject_avg_degradation.values])
    ax1.set_yticks(range(len(subject_avg_degradation)))
    ax1.set_yticklabels([s.replace('_', ' ').title() for s in subject_avg_degradation.index], fontsize=10)
    ax1.set_xlabel('Average Degradation (%)', fontweight='bold')
    ax1.set_title('WORST DEGRADATION: Top 10 Most Vulnerable Subjects', fontweight='bold', color='darkred')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Most resilient subjects (best performers under distortion)
    subject_resilience = performance_df.groupby('subject')['degradation'].mean().sort_values(ascending=True).head(10)
    bars2 = ax2.barh(range(len(subject_resilience)), subject_resilience.values, 
                    color=['green' if x < 5 else 'lightgreen' if x < 10 else 'yellow' for x in subject_resilience.values])
    ax2.set_yticks(range(len(subject_resilience)))
    ax2.set_yticklabels([s.replace('_', ' ').title() for s in subject_resilience.index], fontsize=10)
    ax2.set_xlabel('Average Degradation (%)', fontweight='bold')
    ax2.set_title('MOST RESILIENT: Top 10 Distortion-Resistant Subjects', fontweight='bold', color='darkgreen')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Degradation by miu level (showing critical thresholds)
    miu_degradation = performance_df.groupby('miu')['degradation'].mean()
    miu_degradation = miu_degradation[miu_degradation.index > 0]  # Exclude miu=0
    bars3 = ax3.bar(range(len(miu_degradation)), miu_degradation.values,
                   color=['lightblue' if x < 8 else 'orange' if x < 12 else 'red' for x in miu_degradation.values])
    ax3.set_xticks(range(len(miu_degradation)))
    ax3.set_xticklabels([f'μ={x:.1f}' for x in miu_degradation.index], fontsize=10)
    ax3.set_ylabel('Average Degradation (%)', fontweight='bold')
    ax3.set_title('DEGRADATION SPIKES: Performance Drop by μ Level', fontweight='bold', color='darkorange')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(miu_degradation.values):
        ax3.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 4. Extreme cases analysis (worst individual performance drops)
    extreme_cases = performance_df.nlargest(15, 'degradation')
    scatter = ax4.scatter(extreme_cases['miu'], extreme_cases['degradation'], 
                         s=100, c=extreme_cases['degradation'], cmap='Reds', alpha=0.8, edgecolors='black')
    ax4.set_xlabel('Distortion Level (μ)', fontweight='bold')
    ax4.set_ylabel('Degradation (%)', fontweight='bold')
    ax4.set_title('EXTREME CASES: Worst Individual Degradations', fontweight='bold', color='darkred')
    ax4.grid(True, alpha=0.3)
    
    # Annotate the worst case
    worst_case = extreme_cases.iloc[0]
    ax4.annotate(f'{worst_case["subject"].replace("_", " ").title()}\n{worst_case["degradation"]:.1f}% drop', 
                xy=(worst_case['miu'], worst_case['degradation']),
                xytext=(worst_case['miu'] + 0.1, worst_case['degradation'] - 5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'key_insights_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Key insights summary saved')

def print_key_findings(performance_df):
    """
    Print key numerical findings from the analysis.
    
    Args:
        performance_df (DataFrame): Performance data
    """
    print('\n🔍 KEY FINDINGS:')
    print('=' * 50)
    
    # Calculate key metrics
    subject_avg_degradation = performance_df.groupby('subject')['degradation'].mean().sort_values(ascending=False)
    subject_resilience = performance_df.groupby('subject')['degradation'].mean().sort_values(ascending=True)
    miu_degradation = performance_df.groupby('miu')['degradation'].mean()
    extreme_case = performance_df.loc[performance_df['degradation'].idxmax()]
    
    print(f'📉 Worst degradation subject: {subject_avg_degradation.index[0].replace("_", " ").title()} ({subject_avg_degradation.iloc[0]:.1f}% avg)')
    print(f'🛡️ Most resilient subject: {subject_resilience.index[0].replace("_", " ").title()} ({subject_resilience.iloc[0]:.1f}% avg)')
    print(f'⚡ Worst μ level: μ={miu_degradation.idxmax():.1f} ({miu_degradation.max():.1f}% avg degradation)')
    print(f'🚨 Extreme case: {extreme_case["subject"].replace("_", " ").title()} at μ={extreme_case["miu"]:.1f} ({extreme_case["degradation"]:.1f}% drop)')
    
    # Domain analysis
    print(f'\n📊 DOMAIN PATTERNS:')
    logic_math_subjects = ['formal_logic', 'college_mathematics', 'econometrics']
    medical_subjects = ['professional_medicine', 'medical_genetics', 'high_school_biology']
    
    logic_math_avg = performance_df[performance_df['subject'].isin(logic_math_subjects)]['degradation'].mean()
    medical_avg = performance_df[performance_df['subject'].isin(medical_subjects)]['degradation'].mean()
    
    print(f'Logic/Math subjects average degradation: {logic_math_avg:.1f}%')
    print(f'Medical subjects average degradation: {medical_avg:.1f}%')
    print(f'Difference: {logic_math_avg - medical_avg:.1f} percentage points')

def main():
    """
    Main function to generate all visualizations.
    """
    print('🎨 CHAMELEON PROJECT - VISUALIZATION GENERATOR')
    print('=' * 60)
    
    # Set up paths
    csv_path = 'distortions/chameleon_dataset.csv'
    output_dir = Path('analysis_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Load and prepare data
    original_df, performance_df = load_and_prepare_data(csv_path)
    
    # Generate all visualizations
    create_enhanced_degradation_heatmap(performance_df, output_dir)
    create_individual_plots(performance_df, output_dir)
    create_key_insights_summary(performance_df, output_dir)
    
    # Print findings
    print_key_findings(performance_df)
    
    print('\n✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!')
    print(f'📁 Output directory: {output_dir.absolute()}')
    print('📊 Generated files:')
    for plot_file in sorted(output_dir.glob('*.png')):
        size_mb = plot_file.stat().st_size / (1024 * 1024)
        print(f'  - {plot_file.name} ({size_mb:.1f} MB)')

if __name__ == '__main__':
    main()

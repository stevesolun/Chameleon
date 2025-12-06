"""
Comprehensive Analysis Runner for Chameleon Results

Generates all charts, heatmaps, McNemar tests, and statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

from .mcnemar import (
    analyze_distortion_significance,
    analyze_subject_significance,
    analyze_pairwise_levels,
)
from .visualizations import (
    setup_plot_style,
    create_degradation_heatmap,
    create_accuracy_plots,
    create_statistical_significance_plot,
    create_key_insights_summary,
    create_subject_significance_heatmap,
)
from .reports import (
    generate_statistical_report,
    generate_summary_statistics,
    export_results_to_csv,
    create_key_findings_summary,
)


def run_full_analysis(
    project_name: str,
    projects_dir: str = "Projects",
    results_file: Optional[str] = None
) -> dict:
    """
    Run comprehensive analysis on evaluation results.
    
    Generates:
    1. Accuracy by miu level plot
    2. Subject ranking plot
    3. Degradation heatmap
    4. Key insights summary (4-panel)
    5. Statistical significance plot
    6. Subject significance heatmap
    7. McNemar test CSVs
    8. Statistical report (Markdown)
    
    Args:
        project_name: Name of the project
        projects_dir: Base projects directory
        results_file: Optional specific results file path
    
    Returns:
        Summary dictionary with paths to generated files
    """
    print("\n" + "═" * 60)
    print("📊 CHAMELEON ANALYSIS PIPELINE")
    print("═" * 60)
    
    project_path = Path(projects_dir) / project_name
    results_dir = project_path / "results"
    
    # Find results file
    if results_file:
        csv_path = Path(results_file)
    else:
        csv_path = results_dir / "results.csv"
        if not csv_path.exists():
            # Fallback to distortions_complete
            csv_path = project_path / "distorted_data" / "distortions_complete.csv"
    
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return {"status": "error", "message": f"File not found: {csv_path}"}
    
    print(f"\n📁 Project: {project_name}")
    print(f"📄 Results file: {csv_path}")
    
    # Load data
    print("\n" + "-" * 40)
    print("STEP 1: Loading Data")
    print("-" * 40)
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"✅ Loaded {len(df):,} rows")
    
    # Validate required columns
    required = ['subject', 'question_id', 'miu', 'is_correct']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return {"status": "error", "message": f"Missing columns: {missing}"}
    
    # Ensure is_correct is boolean
    df['is_correct'] = df['is_correct'].astype(bool)
    
    # Create unified output directory for all analysis
    # Consolidates previous analysis_plots/ and synergy_analysis/ into single analysis/
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # For backward compatibility, also reference as plots_dir
    plots_dir = analysis_dir
    
    generated_files = []
    
    # Calculate performance metrics
    print("\n" + "-" * 40)
    print("STEP 2: Calculating Performance Metrics")
    print("-" * 40)
    
    # Per subject and miu accuracy
    perf_df = df.groupby(['subject', 'miu']).agg(
        accuracy=('is_correct', 'mean'),
        total=('is_correct', 'count'),
        correct=('is_correct', 'sum')
    ).reset_index()
    
    # Calculate degradation from baseline (miu=0.0)
    baseline = perf_df[perf_df['miu'] == 0.0][['subject', 'accuracy']].rename(
        columns={'accuracy': 'baseline_accuracy'}
    )
    perf_df = perf_df.merge(baseline, on='subject', how='left')
    perf_df['degradation'] = (perf_df['baseline_accuracy'] - perf_df['accuracy']) * 100
    
    print(f"✅ Calculated metrics for {perf_df['subject'].nunique()} subjects × {perf_df['miu'].nunique()} miu levels")
    
    # Overall stats
    overall_acc = df['is_correct'].mean() * 100
    print(f"   Overall accuracy: {overall_acc:.1f}%")
    
    # Save performance metrics
    perf_csv = plots_dir / "performance_metrics.csv"
    perf_df.to_csv(perf_csv, index=False)
    generated_files.append(str(perf_csv))
    
    # Step 3: McNemar Tests
    print("\n" + "-" * 40)
    print("STEP 3: Running McNemar's Tests")
    print("-" * 40)
    
    # Distortion level significance
    print("   Analyzing distortion level significance...")
    distortion_results = analyze_distortion_significance(
        df, baseline_col='miu', baseline_value=0.0,
        question_id_col='question_id', subject_col='subject', is_correct_col='is_correct'
    )
    dist_csv = plots_dir / "mcnemar_distortion_results.csv"
    distortion_results.to_csv(dist_csv, index=False)
    generated_files.append(str(dist_csv))
    print(f"   ✅ Saved: {dist_csv.name}")
    
    # Subject significance
    print("   Analyzing subject-specific significance...")
    subject_results = analyze_subject_significance(
        df, subject_col='subject', baseline_col='miu', baseline_value=0.0,
        comparison_value=0.9, question_id_col='question_id', is_correct_col='is_correct'
    )
    subj_csv = plots_dir / "mcnemar_subject_results.csv"
    subject_results.to_csv(subj_csv, index=False)
    generated_files.append(str(subj_csv))
    print(f"   ✅ Saved: {subj_csv.name}")
    
    # Pairwise comparisons
    print("   Analyzing pairwise level comparisons...")
    pairwise_results = analyze_pairwise_levels(
        df, level_col='miu', question_id_col='question_id',
        subject_col='subject', is_correct_col='is_correct'
    )
    pair_csv = plots_dir / "mcnemar_pairwise_results.csv"
    pairwise_results.to_csv(pair_csv, index=False)
    generated_files.append(str(pair_csv))
    print(f"   ✅ Saved: {pair_csv.name}")
    
    # Step 4: Generate Visualizations
    print("\n" + "-" * 40)
    print("STEP 4: Generating Visualizations")
    print("-" * 40)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        setup_plot_style()
        
        # 1. Accuracy by miu level
        print("   Creating accuracy plots...")
        acc_figs = create_accuracy_plots(
            perf_df, level_col='miu', accuracy_col='accuracy',
            subject_col='subject', output_dir=plots_dir, prefix='1_'
        )
        generated_files.append(str(plots_dir / '1_accuracy_by_level.png'))
        generated_files.append(str(plots_dir / '1_subject_accuracy_ranking.png'))
        plt.close('all')
        print(f"   ✅ Saved: 1_accuracy_by_level.png, 1_subject_accuracy_ranking.png")
        
        # 2. Degradation heatmap
        print("   Creating degradation heatmap...")
        # Filter non-baseline data for heatmap
        heatmap_df = perf_df[perf_df['miu'] > 0].copy()
        if len(heatmap_df) > 0:
            heatmap_fig = create_degradation_heatmap(
                heatmap_df, subject_col='subject', level_col='miu',
                value_col='degradation', output_path=plots_dir / '2_degradation_heatmap.png',
                title="Performance Degradation from Baseline (μ=0.0)"
            )
            generated_files.append(str(plots_dir / '2_degradation_heatmap.png'))
            plt.close('all')
            print(f"   ✅ Saved: 2_degradation_heatmap.png")
        
        # 3. Key insights summary
        print("   Creating key insights summary...")
        insights_fig = create_key_insights_summary(
            heatmap_df, subject_col='subject', level_col='miu',
            degradation_col='degradation', output_path=plots_dir / '3_key_insights.png'
        )
        generated_files.append(str(plots_dir / '3_key_insights.png'))
        plt.close('all')
        print(f"   ✅ Saved: 3_key_insights.png")
        
        # 4. Statistical significance plot
        if len(distortion_results) > 0:
            print("   Creating statistical significance plot...")
            sig_fig = create_statistical_significance_plot(
                distortion_results,
                level_col='miu_level',
                baseline_acc_col='baseline_accuracy',
                comp_acc_col='comparison_accuracy',
                p_value_col='p_value',
                output_path=plots_dir / '4_statistical_significance.png'
            )
            generated_files.append(str(plots_dir / '4_statistical_significance.png'))
            plt.close('all')
            print(f"   ✅ Saved: 4_statistical_significance.png")
        
        # 5. Subject significance heatmap
        if len(subject_results) > 0:
            print("   Creating subject significance heatmap...")
            subj_fig = create_subject_significance_heatmap(
                subject_results,
                subject_col='subject_name',
                degradation_col='degradation_percent',
                p_value_col='p_value',
                output_path=plots_dir / '5_subject_significance.png'
            )
            generated_files.append(str(plots_dir / '5_subject_significance.png'))
            plt.close('all')
            print(f"   ✅ Saved: 5_subject_significance.png")
        
    except ImportError as e:
        print(f"   ⚠️ Could not generate plots: {e}")
        print("   Install with: pip install matplotlib seaborn")
    
    # Step 5: Generate Reports
    print("\n" + "-" * 40)
    print("STEP 5: Generating Reports")
    print("-" * 40)
    
    # Statistical report (Markdown)
    print("   Generating statistical report...")
    report_content = generate_statistical_report(
        distortion_results, subject_results, pairwise_results,
        output_path=plots_dir / 'Statistical_Analysis_Report.md',
        project_name=f"{project_name} Analysis"
    )
    generated_files.append(str(plots_dir / 'Statistical_Analysis_Report.md'))
    print(f"   ✅ Saved: Statistical_Analysis_Report.md")
    
    # Summary statistics (JSON)
    print("   Generating summary statistics...")
    summary_stats = generate_summary_statistics(df)
    with open(plots_dir / 'summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    generated_files.append(str(plots_dir / 'summary_statistics.json'))
    print(f"   ✅ Saved: summary_statistics.json")
    
    # Key findings
    key_findings = create_key_findings_summary(distortion_results, subject_results)
    with open(plots_dir / 'key_findings.json', 'w') as f:
        json.dump(key_findings, f, indent=2, default=str)
    generated_files.append(str(plots_dir / 'key_findings.json'))
    print(f"   ✅ Saved: key_findings.json")
    
    # Step 6: Generate Executive Report
    print("\n" + "-" * 40)
    print("STEP 6: Generating Executive Report")
    print("-" * 40)
    
    try:
        from .executive_report import generate_executive_report
        report_path = generate_executive_report(project_name, projects_dir, use_ai_insights=False)
        generated_files.append(report_path)
        print(f"   ✅ Saved: Executive_Report.md")
    except Exception as e:
        print(f"   ⚠️ Could not generate executive report: {e}")
    
    # Step 7: Run Synergy Analysis (Advanced)
    print("\n" + "-" * 40)
    print("STEP 7: Running Synergy Analysis (Advanced)")
    print("-" * 40)
    
    try:
        from .synergy_engine import run_synergy_analysis
        
        # Output to same unified analysis directory
        synergy_dir = analysis_dir
        print(f"   Output: {synergy_dir}")
        
        synergy_result = run_synergy_analysis(
            input_csv=csv_path,
            output_dir=synergy_dir,
            validated_only=False,
            n_bootstrap=500,
            verbose=True
        )
        
        if synergy_result.get("status") == "complete":
            generated_files.extend([str(synergy_dir / f) for f in synergy_result.get("files_generated", [])])
            print(f"   ✅ Synergy analysis complete!")
            print(f"   📄 Executive Summary: {synergy_dir / '11_executive_summary.md'}")
    except Exception as e:
        print(f"   ⚠️ Could not run synergy analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    print("\n" + "═" * 60)
    print("📊 ANALYSIS COMPLETE")
    print("═" * 60)
    
    print(f"\n📁 Output directory: {plots_dir}")
    print(f"\n📈 Generated files ({len(generated_files)} total):")
    
    # Group by type
    csvs = [f for f in generated_files if f.endswith('.csv')]
    pngs = [f for f in generated_files if f.endswith('.png')]
    others = [f for f in generated_files if not f.endswith('.csv') and not f.endswith('.png')]
    
    if csvs:
        print(f"\n   📊 Data files ({len(csvs)}):")
        for f in csvs:
            print(f"      • {Path(f).name}")
    
    if pngs:
        print(f"\n   🖼️ Visualizations ({len(pngs)}):")
        for f in pngs:
            print(f"      • {Path(f).name}")
    
    if others:
        print(f"\n   📝 Reports ({len(others)}):")
        for f in others:
            print(f"      • {Path(f).name}")
    
    # Key statistics
    print("\n📈 Key Statistics:")
    print(f"   Overall accuracy: {overall_acc:.1f}%")
    
    if len(distortion_results) > 0:
        sig_levels = (distortion_results['p_value'] < 0.05).sum()
        print(f"   Significant distortion effects: {sig_levels}/{len(distortion_results)}")
    
    if len(subject_results) > 0:
        sig_subjects = (subject_results['is_significant'] == True).sum()
        print(f"   Subjects with significant degradation: {sig_subjects}/{len(subject_results)}")
        
        if 'degradation_percent' in subject_results.columns:
            most_affected = subject_results.nlargest(3, 'degradation_percent')
            print(f"\n   Most affected subjects:")
            for _, row in most_affected.iterrows():
                print(f"      • {row['subject_name']}: {row['degradation_percent']:.1f}% degradation")
    
    print("\n" + "═" * 60)
    
    return {
        "status": "complete",
        "project": project_name,
        "output_dir": str(plots_dir),
        "files_generated": len(generated_files),
        "overall_accuracy": overall_acc,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Chameleon analysis")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--projects-dir", default="Projects", help="Projects directory")
    
    args = parser.parse_args()
    
    run_full_analysis(args.project, args.projects_dir)


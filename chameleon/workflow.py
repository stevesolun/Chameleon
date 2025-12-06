"""
Chameleon Workflow Orchestrator

Complete end-to-end workflow for:
1. Data preparation (create preliminary CSV)
2. Distortion generation (with validation)
3. Model evaluation (batch API)
4. Analysis and reporting

This is the main entry point for running the complete Chameleon pipeline.
"""

import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

from chameleon.distortion.constants import DEFAULT_MIU_VALUES, DEFAULT_DISTORTIONS_PER_QUESTION
from chameleon.distortion.runner import DistortionRunner, DistortionConfig
from chameleon.distortion.validator import validate_distortion, get_validation_stats


@dataclass
class WorkflowConfig:
    """Configuration for the complete workflow."""
    project_name: str
    project_dir: Path
    miu_values: List[float]
    distortions_per_question: int
    distortion_model: str
    distortion_api_key: str
    target_model: str
    target_api_key: Optional[str]
    skip_distortion: bool = False
    skip_evaluation: bool = False
    skip_analysis: bool = False
    
    @classmethod
    def from_project(cls, project_name: str, projects_dir: str = "Projects") -> "WorkflowConfig":
        """Load configuration from a project."""
        project_dir = Path(projects_dir) / project_name
        config_path = project_dir / "config.yaml"
        env_path = project_dir / ".env"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        load_dotenv(env_path)
        
        # Distortion config
        dist_cfg = cfg.get("distortion", {})
        engine_cfg = dist_cfg.get("engine", {})
        dist_vendor = engine_cfg.get("vendor", "mistral")
        
        # Target model config
        target_cfg = cfg.get("target_model", {})
        target_vendor = target_cfg.get("vendor", "openai")
        
        return cls(
            project_name=project_name,
            project_dir=project_dir,
            miu_values=dist_cfg.get("miu_values", DEFAULT_MIU_VALUES),
            distortions_per_question=dist_cfg.get("distortions_per_question", DEFAULT_DISTORTIONS_PER_QUESTION),
            distortion_model=engine_cfg.get("model_name", "mistral-large-latest"),
            distortion_api_key=os.getenv(f"{dist_vendor.upper()}_API_KEY", ""),
            target_model=target_cfg.get("name", "gpt-5.1"),
            target_api_key=os.getenv(f"{target_vendor.upper()}_API_KEY"),
        )


class ChameleonWorkflow:
    """
    Orchestrates the complete Chameleon evaluation workflow.
    
    Stages:
    1. PREPARE: Create preliminary CSV with question-miu pairs
    2. DISTORT: Generate distorted versions using LLM
    3. VALIDATE: Check distortion quality and repair failures
    4. EVALUATE: Send to target model for answers
    5. ANALYZE: Calculate metrics and generate reports
    """
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.original_data_dir = config.project_dir / "original_data"
        self.distorted_data_dir = config.project_dir / "distorted_data"
        self.results_dir = config.project_dir / "results"
        self.analysis_dir = config.project_dir / "analysis"
        
        # Ensure directories exist
        for d in [self.distorted_data_dir, self.results_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete workflow.
        
        Returns:
            Summary dict with results from each stage
        """
        start_time = time.time()
        results = {
            "project": self.config.project_name,
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }
        
        print("=" * 70)
        print("🦎 CHAMELEON WORKFLOW")
        print("=" * 70)
        print(f"Project: {self.config.project_name}")
        print(f"Target Model: {self.config.target_model}")
        print(f"Distortion Model: {self.config.distortion_model}")
        print(f"Miu Values: {self.config.miu_values}")
        print(f"Distortions/Question: {self.config.distortions_per_question}")
        print("=" * 70)
        
        try:
            # Stage 1: Prepare
            print("\n" + "=" * 70)
            print("📋 STAGE 1: DATA PREPARATION")
            print("=" * 70)
            results["stages"]["prepare"] = self._stage_prepare()
            
            # Stage 2: Distort
            if not self.config.skip_distortion:
                print("\n" + "=" * 70)
                print("🔄 STAGE 2: DISTORTION GENERATION")
                print("=" * 70)
                results["stages"]["distort"] = self._stage_distort()
            else:
                print("\n⏭️ Skipping distortion stage")
                results["stages"]["distort"] = {"skipped": True}
            
            # Stage 3: Validate
            print("\n" + "=" * 70)
            print("✅ STAGE 3: VALIDATION")
            print("=" * 70)
            results["stages"]["validate"] = self._stage_validate()
            
            # Stage 4: Evaluate
            if not self.config.skip_evaluation:
                print("\n" + "=" * 70)
                print("🎯 STAGE 4: MODEL EVALUATION")
                print("=" * 70)
                results["stages"]["evaluate"] = self._stage_evaluate()
            else:
                print("\n⏭️ Skipping evaluation stage")
                results["stages"]["evaluate"] = {"skipped": True}
            
            # Stage 5: Analyze
            if not self.config.skip_analysis:
                print("\n" + "=" * 70)
                print("📊 STAGE 5: ANALYSIS")
                print("=" * 70)
                results["stages"]["analyze"] = self._stage_analyze()
            else:
                print("\n⏭️ Skipping analysis stage")
                results["stages"]["analyze"] = {"skipped": True}
            
            results["status"] = "success"
            
        except Exception as e:
            print(f"\n❌ Workflow failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        results["end_time"] = datetime.now().isoformat()
        
        print("\n" + "=" * 70)
        print("🏁 WORKFLOW COMPLETE")
        print("=" * 70)
        print(f"Status: {results['status']}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print("=" * 70)
        
        return results
    
    def _stage_prepare(self) -> Dict[str, Any]:
        """
        Stage 1: Prepare preliminary CSV.
        
        Creates a CSV with all question-miu pairs ready for distortion.
        """
        preliminary_csv = self.distorted_data_dir / "distortions_in_progress.csv"
        
        # Check if already exists
        if preliminary_csv.exists():
            df = pd.read_csv(preliminary_csv, encoding='utf-8')
            print(f"✓ Preliminary CSV already exists: {len(df)} rows")
            return {"status": "exists", "rows": len(df)}
        
        # Load original data
        print("Loading original data...")
        original_files = list(self.original_data_dir.glob("*.csv"))
        
        if not original_files:
            raise FileNotFoundError(f"No CSV files found in {self.original_data_dir}")
        
        all_dfs = []
        for f in original_files:
            df = pd.read_csv(f, encoding='utf-8')
            print(f"  • {f.name}: {len(df)} questions")
            all_dfs.append(df)
        
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"Total: {len(combined)} questions")
        
        # Standardize column names
        col_map = {
            'Question_Text': 'question_text',
            'question': 'question_text',
            'Options_JSON': 'answer_options',
            'Answer': 'answer',
            'correct_answer': 'answer',
            'Subject': 'subject',
        }
        for old, new in col_map.items():
            if old in combined.columns and new not in combined.columns:
                combined = combined.rename(columns={old: new})
        
        # Ensure question_id exists
        if 'question_id' not in combined.columns:
            combined['question_id'] = [f"q_{i}" for i in range(len(combined))]
        
        # Create preliminary data
        print("\nCreating preliminary CSV...")
        preliminary_data = []
        
        N = self.config.distortions_per_question
        miu_values = sorted(self.config.miu_values)
        
        for _, row in combined.iterrows():
            q_id = row['question_id']
            q_text = row.get('question_text', '')
            answer = row.get('answer', '')
            options = row.get('answer_options', '{}')
            subject = row.get('subject', '')
            
            for miu in miu_values:
                if miu == 0.0:
                    # For miu=0, just one row with original question
                    # composite_key = unique identifier for exact row matching
                    composite_key = f"{q_id}__d0__m0.0"
                    preliminary_data.append({
                        'composite_key': composite_key,
                        'question_id': q_id,
                        'distortion_id': 0,
                        'subject': subject,
                        'miu': miu,
                        'question_text': q_text,
                        'distorted_question': q_text,  # Same as original
                        'answer': answer,
                        'options_json': options,
                        'target_model_name': '',
                        'target_model_answer': '',
                        'is_correct': None,
                    })
                else:
                    # For miu>0, N rows to be filled by distortion
                    for i in range(N):
                        d_id = i + 1  # 1-indexed distortion ID
                        composite_key = f"{q_id}__d{d_id}__m{miu}"
                        preliminary_data.append({
                            'composite_key': composite_key,
                            'question_id': q_id,
                            'distortion_id': d_id,
                            'subject': subject,
                            'miu': miu,
                            'question_text': q_text,
                            'distorted_question': '',  # To be filled
                            'answer': answer,
                            'options_json': options,
                            'target_model_name': '',
                            'target_model_answer': '',
                            'is_correct': None,
                        })
        
        preliminary_df = pd.DataFrame(preliminary_data)
        preliminary_df.to_csv(preliminary_csv, index=False, encoding='utf-8')
        
        print(f"✓ Created preliminary CSV: {len(preliminary_df)} rows")
        return {"status": "created", "rows": len(preliminary_df)}
    
    def _stage_distort(self) -> Dict[str, Any]:
        """
        Stage 2: Generate distortions.
        """
        if not self.config.distortion_api_key:
            raise ValueError("Distortion API key not configured")
        
        runner = DistortionRunner(DistortionConfig(
            project_dir=self.config.project_dir,
            miu_values=self.config.miu_values,
            distortions_per_question=self.config.distortions_per_question,
            model=self.config.distortion_model,
            api_key=self.config.distortion_api_key,
        ))
        
        return runner.run(display_progress=True)
    
    def _stage_validate(self) -> Dict[str, Any]:
        """
        Stage 3: Validate distortions and identify failures.
        """
        csv_path = self.distorted_data_dir / "distortions_in_progress.csv"
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        print("Validating distortions...")
        
        validation_results = []
        invalid_count = 0
        
        for idx, row in df.iterrows():
            if row['miu'] == 0.0:
                continue  # Skip miu=0 (original questions)
            
            original = str(row.get('question_text', ''))
            distorted = str(row.get('distorted_question', ''))
            miu = row['miu']
            
            if pd.isna(row.get('distorted_question')) or distorted == '':
                invalid_count += 1
                continue
            
            result = validate_distortion(original, distorted, miu)
            if not result.is_valid:
                invalid_count += 1
                validation_results.append({
                    'index': idx,
                    'question_id': row['question_id'],
                    'miu': miu,
                    'failures': [f.value for f in result.failures]
                })
        
        valid_count = len(df[df['miu'] > 0]) - invalid_count
        total = len(df[df['miu'] > 0])
        
        print(f"✓ Valid: {valid_count}/{total} ({valid_count*100/total:.1f}%)")
        print(f"✗ Invalid: {invalid_count}/{total}")
        
        if validation_results:
            # Count failure types
            failure_counts = {}
            for r in validation_results:
                for f in r['failures']:
                    failure_counts[f] = failure_counts.get(f, 0) + 1
            
            print("\nFailure breakdown:")
            for f_type, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                print(f"  • {f_type}: {count}")
        
        return {
            "total": total,
            "valid": valid_count,
            "invalid": invalid_count,
            "valid_rate": valid_count / total if total > 0 else 0,
        }
    
    def _stage_evaluate(self) -> Dict[str, Any]:
        """
        Stage 4: Send distorted questions to target model.
        """
        if not self.config.target_api_key:
            print("⚠️ Target model API key not configured. Skipping evaluation.")
            return {"skipped": True, "reason": "no_api_key"}
        
        try:
            from chameleon.evaluation.batch_processor import BatchProcessor, EvaluationConfig
            
            eval_config = EvaluationConfig(
                project_dir=self.config.project_dir,
                model=self.config.target_model,
                api_key=self.config.target_api_key,
            )
            
            processor = BatchProcessor(eval_config)
            
            # Create and submit batches
            batch_files = processor.create_batches()
            
            if not batch_files:
                return {"status": "error", "message": "No batch files created"}
            
            processor.submit_batches(interactive=False)
            
            # Monitor until complete
            print("\n⏳ Waiting for batch completion...")
            while True:
                if processor.monitor():
                    break
                print("⏳ Still processing... checking again in 2 minutes")
                time.sleep(120)
            
            return {"status": "complete"}
            
        except ImportError:
            print("⚠️ OpenAI package not installed. Skipping batch evaluation.")
            return {"skipped": True, "reason": "openai_not_installed"}
    
    def _stage_analyze(self) -> Dict[str, Any]:
        """
        Stage 5: Analyze results and generate reports.
        """
        csv_path = self.distorted_data_dir / "distortions_complete.csv"
        
        if not csv_path.exists():
            csv_path = self.distorted_data_dir / "distortions_in_progress.csv"
        
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        results = {
            "total_questions": len(df),
        }
        
        # Check if we have evaluation results
        if 'is_correct' in df.columns and 'model_answer' in df.columns:
            valid_df = df[df['model_answer'].isin(['A', 'B', 'C', 'D'])]
            
            if len(valid_df) > 0:
                overall_acc = valid_df['is_correct'].mean() * 100
                results["overall_accuracy"] = overall_acc
                print(f"Overall Accuracy: {overall_acc:.1f}%")
                
                # By miu
                if 'miu' in df.columns:
                    print("\nAccuracy by μ:")
                    miu_results = {}
                    
                    for miu in sorted(df['miu'].unique()):
                        subset = valid_df[valid_df['miu'] == miu]
                        if len(subset) > 0:
                            acc = subset['is_correct'].mean() * 100
                            miu_results[miu] = acc
                            print(f"  μ={miu:.1f}: {acc:.1f}% ({len(subset)} questions)")
                    
                    results["accuracy_by_miu"] = miu_results
                
                # By subject
                if 'subject' in df.columns:
                    print("\nAccuracy by Subject:")
                    subject_results = {}
                    
                    for subject in df['subject'].unique():
                        if pd.isna(subject):
                            continue
                        subset = valid_df[valid_df['subject'] == subject]
                        if len(subset) > 0:
                            acc = subset['is_correct'].mean() * 100
                            subject_results[subject] = acc
                            print(f"  {subject}: {acc:.1f}%")
                    
                    results["accuracy_by_subject"] = subject_results
        else:
            print("No evaluation results found. Run evaluation stage first.")
        
        return results


def run_workflow(project_name: str, projects_dir: str = "Projects", **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run the complete workflow.
    
    Args:
        project_name: Name of the project
        projects_dir: Base directory for projects
        **kwargs: Additional options (skip_distortion, skip_evaluation, skip_analysis)
    
    Returns:
        Workflow results
    """
    config = WorkflowConfig.from_project(project_name, projects_dir)
    
    # Apply kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    workflow = ChameleonWorkflow(config)
    return workflow.run()


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Chameleon workflow")
    parser.add_argument("project", help="Project name")
    parser.add_argument("--projects-dir", default="Projects", help="Projects directory")
    parser.add_argument("--skip-distortion", action="store_true", help="Skip distortion stage")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation stage")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis stage")
    
    args = parser.parse_args()
    
    result = run_workflow(
        args.project,
        args.projects_dir,
        skip_distortion=args.skip_distortion,
        skip_evaluation=args.skip_evaluation,
        skip_analysis=args.skip_analysis,
    )
    
    sys.exit(0 if result.get("status") == "success" else 1)


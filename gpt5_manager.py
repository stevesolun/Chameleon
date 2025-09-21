#!/usr/bin/env python3
"""
GPT-5 Batch Manager - Monolithic tool for managing GPT-5 batch processing
Combines all batch operations: create, submit, monitor, cancel, repair, cleanup
"""

import os
import sys
import json
import yaml
import math
import pandas as pd
import openai
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT5BatchManager:
    """Comprehensive GPT-5 batch management system"""
    
    def __init__(self):
        """Initialize the batch manager"""
        self.load_environment()
        self.load_config()
        self.client = openai.OpenAI(api_key=self.api_key)
        self.batch_dir = Path("batches")
        self.jsonl_dir = self.batch_dir / "jsonl"
        self.results_dir = self.batch_dir / "results"
        self.tracking_dir = self.batch_dir / "tracking"
        
        # Ensure directories exist
        for dir_path in [self.batch_dir, self.jsonl_dir, self.results_dir, self.tracking_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.tracking_file = self.tracking_dir / "batch_info.json"
        
        logger.info("GPT-5 Batch Manager initialized")
    
    def load_environment(self):
        """Load environment variables"""
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
    
    def load_config(self):
        """Load configuration from config.yaml"""
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _cleanup_old_batch_files(self, new_split_percentage: float, total_questions: int):
        """Clean up old batch files if configuration has changed"""
        # Check if any batch files exist
        existing_files = list(self.jsonl_dir.glob("gpt5_batch_part_*.jsonl"))
        if not existing_files:
            return
        
        # Calculate expected number of batches with new configuration
        expected_batches = math.ceil(1.0 / new_split_percentage)
        current_batches = len(existing_files)
        
        print(f"🔍 Found {current_batches} existing batch files")
        print(f"📊 New configuration expects {expected_batches} batch files")
        
        # Check if we need to clean up (different number of batches or different split)
        needs_cleanup = False
        
        if current_batches != expected_batches:
            print(f"🧹 Batch count changed: {current_batches} → {expected_batches}")
            needs_cleanup = True
        else:
            # Check if split percentage has changed by examining first file
            if existing_files:
                with open(existing_files[0], 'r') as f:
                    actual_requests = sum(1 for _ in f)
                expected_requests = math.ceil(total_questions * new_split_percentage)
                
                if abs(actual_requests - expected_requests) > 10:  # Allow small variance
                    print(f"🧹 Batch size changed: ~{actual_requests} → ~{expected_requests} requests per batch")
                    needs_cleanup = True
        
        if needs_cleanup:
            print(f"🗑️  Removing {current_batches} old batch files...")
            for file in existing_files:
                file.unlink()
            print(f"✅ Old batch files cleaned up")
        else:
            print(f"✅ Existing batch files match configuration - no cleanup needed")
    
    def create_batches(self):
        """Create GPT-5 batch files from CSV dataset"""
        print("🚀 Creating GPT-5 Batches")
        print("=" * 50)
        
        # Load dataset
        csv_path = Path("distortions/comprehensive_distortion_dataset.csv")
        if not csv_path.exists():
            print(f"❌ CSV file not found: {csv_path}")
            return False
        
        print(f"📊 Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} questions")
        
        # Get split percentage
        split_percentage = self.config.get('settings', {}).get('batch_split_percentage', 0.1)
        print(f"📊 Batch split: {split_percentage*100:.1f}%")
        
        # Clean up old batch files when configuration changes
        self._cleanup_old_batch_files(split_percentage, len(df))
        
        # Create requests
        batch_requests = []
        for idx, row in df.iterrows():
            try:
                answer_options = json.loads(row['answer_options'])
                question_text = row['distorted_question']
                choices_text = "\n".join([f"{k}: {v}" for k, v in answer_options.items()])
                custom_id = f"q_{idx}"
                
                # RLHF-style request
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"EVALUATION TASK: You will receive +1 point for choosing the correct answer and -2 points for any incorrect choice. Your goal is to maximize your score.\n\nINSTRUCTIONS:\n- Read the question carefully\n- Analyze all four options (A, B, C, D)\n- Choose the ONE most correct answer\n- Respond with ONLY the letter (A, B, C, or D)\n- No explanations, no additional text, no reasoning shown\n- Wrong answers result in -2 points, correct answers give +1 point\n\nQUESTION:\n{question_text}\n\nOPTIONS:\n{choices_text}\n\nYour answer (single letter only):"
                            }
                        ],
                        "max_completion_tokens": 2500
                    }
                }
                batch_requests.append(request)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Skipping row {idx}: {e}")
                continue
        
        print(f"✅ Created {len(batch_requests)} requests")
        
        # Split into batches
        batch_files = self._split_requests(batch_requests, split_percentage)
        
        print(f"\n✅ Created {len(batch_files)} batch files in {self.jsonl_dir}")
        for i, batch_file in enumerate(batch_files, 1):
            file_size = batch_file.stat().st_size / 1024 / 1024
            with open(batch_file, 'r') as f:
                num_requests = sum(1 for _ in f)
            print(f"   Part {i}: {batch_file.name} ({file_size:.1f} MB, {num_requests:,} requests)")
        
        return batch_files
    
    def _split_requests(self, requests: List[dict], split_percentage: float) -> List[Path]:
        """Split requests into batch files"""
        total_requests = len(requests)
        num_batches = math.ceil(1.0 / split_percentage)
        requests_per_batch = math.ceil(total_requests * split_percentage)
        
        print(f"🔧 Splitting {total_requests:,} requests into {num_batches} batches")
        print(f"📊 {split_percentage*100:.1f}% per batch = ~{requests_per_batch:,} requests per batch")
        
        batch_files = []
        
        for i in range(num_batches):
            start_idx = i * requests_per_batch
            end_idx = min((i + 1) * requests_per_batch, total_requests)
            
            if start_idx >= total_requests:
                break
            
            batch_requests = requests[start_idx:end_idx]
            batch_filename = self.jsonl_dir / f"gpt5_batch_part_{i+1:02d}.jsonl"
            
            with open(batch_filename, 'w') as f:
                for request in batch_requests:
                    f.write(json.dumps(request) + '\n')
            
            batch_files.append(batch_filename)
            
            actual_percentage = (len(batch_requests) / total_requests) * 100
            print(f"✅ Part {i+1}: {batch_filename.name} ({len(batch_requests):,} requests, {actual_percentage:.1f}%)")
        
        return batch_files
    
    def submit_batches(self, interactive: bool = True):
        """Submit batch files to GPT-5"""
        print("🚀 GPT-5 Batch Submission")
        print("=" * 40)
        
        # Find available batch files
        batch_files = sorted(list(self.jsonl_dir.glob("gpt5_batch_part_*.jsonl")))
        if not batch_files:
            print("❌ No batch files found! Run create_batches first.")
            return False
        
        # Load existing tracking
        submitted_batches = self._load_tracking()
        
        # Show available files
        print(f"📁 Available batch files:")
        available_files = []
        
        for i, batch_file in enumerate(batch_files, 1):
            file_size = batch_file.stat().st_size / 1024 / 1024
            with open(batch_file, 'r') as f:
                num_requests = sum(1 for _ in f)
            
            is_submitted = str(batch_file) in [b['batch_file'] for b in submitted_batches]
            status = "✅ SUBMITTED" if is_submitted else "⏳ PENDING"
            
            print(f"   {i:2d}. {batch_file.name}")
            print(f"       {file_size:.1f} MB, {num_requests:,} requests - {status}")
            
            if not is_submitted:
                available_files.append((i, batch_file, num_requests))
        
        if not available_files:
            print("✅ All batches already submitted!")
            return True
        
        # Interactive selection
        if interactive:
            selected_files = self._interactive_selection(available_files)
            if not selected_files:
                return False
        else:
            selected_files = available_files
        
        # Submit selected batches
        return self._submit_selected_batches(selected_files)
    
    def _interactive_selection(self, available_files):
        """Interactive batch selection"""
        print(f"\n📋 Choose batches to submit:")
        print(f"   • Enter batch numbers (e.g., '1', '1,3,5', '1-3')")
        print(f"   • Enter 'all' to submit all pending batches")
        print(f"   • Enter 'q' to quit")
        
        while True:
            choice = input(f"\nSelect batches to submit: ").strip().lower()
            
            if choice == 'q':
                print("❌ Submission cancelled")
                return None
            
            if choice == 'all':
                return available_files
            
            # Parse selection
            try:
                selected_files = []
                for part in choice.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        for num in range(start, end + 1):
                            for item in available_files:
                                if item[0] == num:
                                    selected_files.append(item)
                    else:
                        num = int(part)
                        for item in available_files:
                            if item[0] == num:
                                selected_files.append(item)
                
                if selected_files:
                    # Remove duplicates
                    selected_files = list({item[1]: item for item in selected_files}.values())
                    return selected_files
                else:
                    print("❌ Invalid selection. Please try again.")
                    
            except ValueError:
                print("❌ Invalid format. Use numbers, ranges (1-3), or 'all'")
    
    def _submit_selected_batches(self, selected_files):
        """Submit the selected batch files"""
        print(f"\n🚀 Submitting {len(selected_files)} batch(es)...")
        
        new_submissions = []
        
        for i, (batch_num, file_path, num_requests) in enumerate(selected_files, 1):
            print(f"\n🚀 Submitting Part {batch_num} ({i}/{len(selected_files)})...")
            
            try:
                # Upload file
                with open(file_path, "rb") as f:
                    batch_input_file = self.client.files.create(file=f, purpose="batch")
                
                # Create batch
                batch = self.client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": f"Chameleon Dataset Part {batch_num} - {num_requests:,} Questions"}
                )
                
                batch_info = {
                    'part': batch_num,
                    'batch_id': batch.id,
                    'batch_file': str(file_path),
                    'num_requests': num_requests,
                    'submitted_at': datetime.now().isoformat(),
                    'status': batch.status
                }
                
                new_submissions.append(batch_info)
                print(f"✅ Part {batch_num} submitted: {batch.id}")
                
            except Exception as e:
                print(f"❌ Failed to submit Part {batch_num}: {e}")
                continue
        
        # Update tracking
        if new_submissions:
            self._update_tracking(new_submissions)
            print(f"\n🎉 Successfully submitted {len(new_submissions)} batch(es)!")
            
            for batch in new_submissions:
                print(f"   Part {batch['part']}: {batch['batch_id']}")
        
        return len(new_submissions) > 0
    
    def monitor_batches(self):
        """Monitor batch progress and handle completion"""
        print("👀 Monitoring GPT-5 Batches")
        print("=" * 40)
        
        tracking_data = self._load_tracking()
        if not tracking_data:
            print("❌ No batches to monitor!")
            return False
        
        print(f"📋 Monitoring {len(tracking_data)} batch(es)")
        
        all_completed = True
        all_statuses = []
        newly_completed = []
        
        for batch_info in tracking_data:
            batch_id = batch_info['batch_id']
            part_num = batch_info['part']
            
            print(f"\n📊 Checking Part {part_num}: {batch_id}")
            
            try:
                batch = self.client.batches.retrieve(batch_id)
                
                print(f"   📊 Status: {batch.status}")
                if batch.request_counts:
                    counts = batch.request_counts
                    total = counts.total
                    completed = counts.completed
                    failed = counts.failed
                    
                    if total > 0:
                        progress = (completed / total) * 100
                        print(f"   📈 Progress: {progress:.1f}% ({completed:,}/{total:,})")
                        if failed > 0:
                            print(f"   ❌ Failed: {failed:,}")
                
                # Calculate completion time if available
                completion_time = None
                if hasattr(batch, 'created_at') and hasattr(batch, 'completed_at'):
                    if batch.created_at and batch.completed_at:
                        created = datetime.fromtimestamp(batch.created_at)
                        completed = datetime.fromtimestamp(batch.completed_at)
                        completion_time = str(completed - created)
                
                status_info = {
                    'batch_id': batch_id,
                    'part': part_num,
                    'status': batch.status,
                    'request_counts': batch.request_counts.__dict__ if batch.request_counts else None,
                    'output_file_id': getattr(batch, 'output_file_id', None),
                    'created_at': getattr(batch, 'created_at', None),
                    'completed_at': getattr(batch, 'completed_at', None),
                    'completion_time': completion_time
                }
                
                all_statuses.append(status_info)
                
                # Show timing info if available
                if completion_time:
                    print(f"   ⏱️  Completion time: {completion_time}")
                
                # Check if this batch just completed and hasn't been downloaded yet
                if batch.status == 'completed' and not batch_info.get('downloaded', False):
                    newly_completed.append(status_info)
                    # Mark as downloaded in tracking
                    batch_info['downloaded'] = True
                    batch_info['download_timestamp'] = datetime.now().isoformat()
                    batch_info['completion_time'] = completion_time
                    batch_info['completed_at'] = getattr(batch, 'completed_at', None)
                
                if batch.status not in ['completed', 'failed', 'cancelled']:
                    all_completed = False
                    
            except Exception as e:
                print(f"   ❌ Error checking status: {e}")
                all_completed = False
        
        # Download newly completed batches immediately
        if newly_completed:
            print(f"\n📥 Downloading {len(newly_completed)} newly completed batch(es)...")
            self._download_completed_batches(newly_completed)
            
            # Update tracking file with download status
            self._save_tracking(tracking_data)
        
        # Summary
        completed_count = sum(1 for s in all_statuses if s['status'] == 'completed')
        failed_count = sum(1 for s in all_statuses if s['status'] == 'failed')
        in_progress_count = len(all_statuses) - completed_count - failed_count
        
        print(f"\n📊 Status Summary:")
        print(f"   ✅ Completed: {completed_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   ⏳ In Progress: {in_progress_count}")
        
        if all_completed:
            print(f"\n🎉 All batches finished! Ready for final CSV update.")
            return True
        else:
            print(f"\n⏳ Still processing... ({completed_count}/{len(tracking_data)} completed)")
            return False
    
    def _download_completed_batches(self, completed_batches):
        """Download results from newly completed batches"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = {}
        
        for status in completed_batches:
            if status['output_file_id']:
                part_num = status['part']
                batch_id = status['batch_id']
                print(f"📥 Downloading Part {part_num} results (batch: {batch_id[:20]}...)...")
                
                try:
                    # Download results
                    content = self.client.files.content(status['output_file_id'])
                    results_file = self.results_dir / f"gpt5_results_part_{part_num}_{timestamp}.jsonl"
                    
                    with open(results_file, 'wb') as f:
                        f.write(content.content)
                    
                    # Create metadata file with batch info
                    metadata_file = self.results_dir / f"gpt5_results_part_{part_num}_{timestamp}.meta.json"
                    metadata = {
                        'part': part_num,
                        'batch_id': batch_id,
                        'download_timestamp': timestamp,
                        'results_file': results_file.name,
                        'request_counts': status.get('request_counts', {}),
                        'completion_time': status.get('completion_time'),
                        'created_at': status.get('created_at'),
                        'completed_at': status.get('completed_at')
                    }
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Parse results
                    batch_results = {}
                    with open(results_file, 'r') as f:
                        for line in f:
                            try:
                                result = json.loads(line)
                                custom_id = result.get('custom_id')
                                if custom_id and result.get('response'):
                                    content = result['response']['body']['choices'][0]['message']['content']
                                    batch_results[custom_id] = content.strip()
                            except (json.JSONDecodeError, KeyError):
                                continue
                    
                    all_results.update(batch_results)
                    print(f"✅ Part {part_num}: {len(batch_results)} results downloaded")
                    print(f"📋 Metadata saved: {metadata_file.name}")
                    
                except Exception as e:
                    print(f"❌ Error downloading Part {part_num}: {e}")
        
        # Update CSV with new results if any were downloaded
        if all_results:
            print(f"\n💾 Updating CSV with {len(all_results)} new results...")
            self._update_csv_with_results(all_results, timestamp)
        
        return len(all_results)

    def _handle_completion(self, all_statuses):
        """Handle batch completion - download and merge results"""
        print(f"\n🎉 All batches completed! Processing results...")
        
        # Download all results
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for status in all_statuses:
            if status['output_file_id']:
                part_num = status['part']
                print(f"📥 Downloading Part {part_num} results...")
                
                try:
                    # Download results
                    content = self.client.files.content(status['output_file_id'])
                    results_file = self.results_dir / f"gpt5_results_part_{part_num}_{timestamp}.jsonl"
                    
                    with open(results_file, 'wb') as f:
                        f.write(content.content)
                    
                    # Parse results
                    with open(results_file, 'r') as f:
                        for line in f:
                            try:
                                result = json.loads(line)
                                custom_id = result.get('custom_id')
                                if custom_id and result.get('response'):
                                    content = result['response']['body']['choices'][0]['message']['content']
                                    all_results[custom_id] = content.strip()
                            except (json.JSONDecodeError, KeyError):
                                continue
                    
                    print(f"✅ Part {part_num}: {len(all_results)} results processed")
                    
                except Exception as e:
                    print(f"❌ Error downloading Part {part_num}: {e}")
        
        # Update CSV with results
        return self._update_csv_with_results(all_results, timestamp)
    
    def _update_csv_with_results(self, all_results, timestamp):
        """Update CSV file with GPT-5 results"""
        csv_path = Path("distortions/comprehensive_distortion_dataset.csv")
        
        print(f"📊 Updating CSV with {len(all_results)} results...")
        
        df = pd.read_csv(csv_path)
        
        matched_count = 0
        valid_answers = 0
        
        for idx, row in df.iterrows():
            custom_id = f"q_{idx}"
            if custom_id in all_results:
                gpt5_answer = all_results[custom_id]
                df.at[idx, 'gpt5_answer'] = gpt5_answer
                matched_count += 1
                
                # Calculate correctness
                if gpt5_answer in ['A', 'B', 'C', 'D']:
                    valid_answers += 1
                    correct_answer = row['correct_answer']
                    is_correct = (gpt5_answer == correct_answer)
                    df.at[idx, 'is_correct'] = is_correct
                else:
                    df.at[idx, 'is_correct'] = False
        
        # Save updated CSV
        output_csv = f"distortions/comprehensive_distortion_dataset_with_gpt5_{timestamp}.csv"
        df.to_csv(output_csv, index=False)
        
        print(f"💾 Results saved: {output_csv}")
        print(f"🔗 Matched: {matched_count}/{len(df)} ({matched_count/len(df)*100:.1f}%)")
        print(f"✅ Valid answers: {valid_answers}/{len(df)} ({valid_answers/len(df)*100:.1f}%)")
        
        # Performance analysis
        if valid_answers > 0:
            valid_df = df[df['gpt5_answer'].isin(['A', 'B', 'C', 'D'])]
            overall_accuracy = valid_df['is_correct'].mean() * 100
            
            print(f"\n📈 GPT-5 Performance:")
            print(f"   Overall accuracy: {overall_accuracy:.1f}%")
            
            # Accuracy by μ value
            print(f"\n📊 Accuracy by μ value:")
            miu_accuracy = valid_df.groupby('miu')['is_correct'].agg(['mean', 'count']).round(3)
            for miu, (accuracy, count) in miu_accuracy.iterrows():
                print(f"   μ={miu:.1f}: {accuracy*100:.1f}% ({count} questions)")
        
        # Cleanup
        self.tracking_file.unlink(missing_ok=True)
        print(f"\n🧹 Cleaned up tracking file")
        print(f"🎉 GPT-5 evaluation complete!")
        
        return True
    
    def cancel_batches(self, batch_ids: List[str] = None):
        """Cancel specific batches or show interactive cancellation"""
        print("🛑 Batch Cancellation")
        print("=" * 30)
        
        tracking_data = self._load_tracking()
        if not tracking_data:
            print("❌ No batches to cancel!")
            return False
        
        # Check which batches can be cancelled
        cancellable_batches = []
        
        for batch_info in tracking_data:
            batch_id = batch_info['batch_id']
            part_num = batch_info['part']
            
            try:
                batch = self.client.batches.retrieve(batch_id)
                if batch.status in ['validating', 'in_progress']:
                    cancellable_batches.append({
                        'part': part_num,
                        'batch_id': batch_id,
                        'status': batch.status,
                        'num_requests': batch_info.get('num_requests', 0)
                    })
                    print(f"⚠️  Part {part_num}: {batch.status} (can cancel)")
                else:
                    print(f"✅ Part {part_num}: {batch.status} (cannot cancel)")
                    
            except Exception as e:
                print(f"❌ Part {part_num}: Error checking status")
        
        if not cancellable_batches:
            print("✅ No batches can be cancelled!")
            return True
        
        # Interactive cancellation
        if not batch_ids:
            print(f"\n🛑 Select batches to cancel:")
            for i, batch in enumerate(cancellable_batches, 1):
                print(f"   {i}. Part {batch['part']}: {batch['status']} ({batch['num_requests']} requests)")
            
            choice = input(f"\nEnter part numbers to cancel (e.g., '1,3') or 'all': ").strip().lower()
            
            if choice == 'all':
                selected_batches = cancellable_batches
            else:
                try:
                    parts = [int(p.strip()) for p in choice.split(',')]
                    selected_batches = [b for b in cancellable_batches if b['part'] in parts]
                except ValueError:
                    print("❌ Invalid selection")
                    return False
        else:
            selected_batches = [b for b in cancellable_batches if b['batch_id'] in batch_ids]
        
        # Cancel selected batches
        cancelled_count = 0
        for batch in selected_batches:
            try:
                self.client.batches.cancel(batch['batch_id'])
                print(f"✅ Cancelled Part {batch['part']}: {batch['batch_id']}")
                cancelled_count += 1
            except Exception as e:
                print(f"❌ Failed to cancel Part {batch['part']}: {e}")
        
        print(f"\n🎉 Cancelled {cancelled_count} batch(es)")
        return cancelled_count > 0
    
    def cleanup_tracking(self, auto_cleanup: bool = False, cleanup_failed_only: bool = False):
        """Clean up tracking data"""
        print("🧹 Cleaning Up Tracking Data")
        print("=" * 35)
        
        tracking_data = self._load_tracking()
        if not tracking_data:
            print("✅ No tracking data to clean")
            return True
        
        if auto_cleanup or cleanup_failed_only:
            # Auto-remove failed/cancelled batches while keeping completed ones
            active_batches = []
            removed_count = 0
            cancelled_count = 0
            
            print("🔍 Analyzing batch statuses...")
            
            for batch_info in tracking_data:
                try:
                    batch = self.client.batches.retrieve(batch_info['batch_id'])
                    part_num = batch_info['part']
                    
                    if batch.status == 'completed':
                        # Always keep completed batches (even with some failed requests)
                        active_batches.append(batch_info)
                        print(f"✅ Keeping Part {part_num}: completed")
                        
                    elif batch.status in ['validating', 'in_progress', 'finalizing']:
                        # Keep active batches
                        active_batches.append(batch_info)
                        print(f"⏳ Keeping Part {part_num}: {batch.status}")
                        
                    elif batch.status == 'failed':
                        # Remove failed batches (they can't be resubmitted)
                        removed_count += 1
                        print(f"🗑️  Removed Part {part_num}: failed (token_limit_exceeded)")
                        
                    elif batch.status in ['cancelled', 'expired']:
                        # Remove cancelled/expired batches
                        removed_count += 1
                        print(f"🗑️  Removed Part {part_num}: {batch.status}")
                        
                    else:
                        # Unknown status - remove to be safe
                        removed_count += 1
                        print(f"🗑️  Removed Part {part_num}: unknown status ({batch.status})")
                        
                except Exception as e:
                    # Error retrieving batch - remove from tracking
                    removed_count += 1
                    print(f"🗑️  Removed Part {batch_info['part']}: error retrieving ({str(e)[:50]}...)")
            
            # Save updated tracking
            self._save_tracking(active_batches)
            
            print(f"\n📊 Cleanup Summary:")
            print(f"   ✅ Kept: {len(active_batches)} batches")
            print(f"   🗑️  Removed: {removed_count} batches")
            
            if cleanup_failed_only:
                print(f"\n💡 Next steps:")
                print(f"   1. Create new batches for missing parts")
                print(f"   2. Submit missing parts with proven ultra-safe config")
                print(f"   3. Monitor progress and collect any failures")
            
        else:
            # Interactive cleanup
            print("Select cleanup option:")
            print("1. Remove specific part")
            print("2. Auto-cleanup failed batches only (keep completed)")
            print("3. Auto-cleanup all inactive batches")
            print("4. Clear all tracking")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == '1':
                part = int(input("Enter part number to remove: "))
                updated_data = [b for b in tracking_data if b['part'] != part]
                self._save_tracking(updated_data)
                print(f"✅ Removed Part {part}")
                
            elif choice == '2':
                return self.cleanup_tracking(cleanup_failed_only=True)
                
            elif choice == '3':
                return self.cleanup_tracking(auto_cleanup=True)
                
            elif choice == '4':
                self._save_tracking([])
                print("✅ Cleared all tracking data")
        
        return True
    
    def collect_failed_requests(self):
        """Collect failed requests from existing result files"""
        print("🔍 Collecting Failed Requests")
        print("=" * 35)
        
        results_files = list(self.results_dir.glob("gpt5_results_part_*.jsonl"))
        if not results_files:
            print("❌ No result files found!")
            return []
        
        failed_custom_ids = []
        
        for results_file in results_files:
            print(f"📄 Checking {results_file.name}...")
            
            try:
                with open(results_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            result = json.loads(line)
                            
                            # Check for actual errors or empty responses
                            custom_id = result.get('custom_id')
                            
                            # Case 1: Actual API errors
                            if result.get('error'):
                                if custom_id:
                                    failed_custom_ids.append(custom_id)
                                    print(f"   ❌ API Error: {custom_id} - {result['error'].get('message', 'Unknown error')}")
                            
                            # Case 2: Empty or invalid responses
                            elif result.get('response') and custom_id:
                                try:
                                    response_body = result['response']['body']
                                    content = response_body['choices'][0]['message']['content'].strip()
                                    
                                    # Check for empty content or invalid answers
                                    if not content or content not in ['A', 'B', 'C', 'D']:
                                        failed_custom_ids.append(custom_id)
                                        finish_reason = response_body['choices'][0].get('finish_reason', 'unknown')
                                        print(f"   ❌ Invalid Response: {custom_id} - Content: '{content}' (finish_reason: {finish_reason})")
                                        
                                except (KeyError, IndexError, TypeError):
                                    if custom_id:
                                        failed_custom_ids.append(custom_id)
                                        print(f"   ❌ Malformed Response: {custom_id} - Cannot parse response structure")
                            
                        except json.JSONDecodeError:
                            print(f"   ⚠️  Skipping malformed line {line_num}")
                            continue
                            
            except Exception as e:
                print(f"   ❌ Error reading {results_file.name}: {e}")
                continue
        
        print(f"\n📊 Summary: Found {len(failed_custom_ids)} failed requests")
        
        if failed_custom_ids:
            print("🔧 Failed request IDs:")
            for i, custom_id in enumerate(failed_custom_ids[:10], 1):  # Show first 10
                print(f"   {i:2d}. {custom_id}")
            if len(failed_custom_ids) > 10:
                print(f"   ... and {len(failed_custom_ids) - 10} more")
        
        return failed_custom_ids
    
    def create_repair_batch(self, failed_custom_ids: List[str] = None):
        """Create repair batch for failed requests"""
        print("🔧 Creating Repair Batch")
        print("=" * 30)
        
        # Auto-collect failures if not provided
        if failed_custom_ids is None:
            failed_custom_ids = self.collect_failed_requests()
        
        if not failed_custom_ids:
            print("❌ No failed custom IDs found!")
            return None
        
        # Load original dataset
        csv_path = Path("distortions/comprehensive_distortion_dataset.csv")
        df = pd.read_csv(csv_path)
        
        repair_requests = []
        
        for custom_id in failed_custom_ids:
            try:
                row_idx = int(custom_id.split('_')[1])
                if row_idx >= len(df):
                    continue
                
                row = df.iloc[row_idx]
                answer_options = json.loads(row['answer_options'])
                question_text = row['distorted_question']
                choices_text = "\n".join([f"{k}: {v}" for k, v in answer_options.items()])
                
                # Create repair request with RLHF format
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"EVALUATION TASK: You will receive +1 point for choosing the correct answer and -2 points for any incorrect choice. Your goal is to maximize your score.\n\nINSTRUCTIONS:\n- Read the question carefully\n- Analyze all four options (A, B, C, D)\n- Choose the ONE most correct answer\n- Respond with ONLY the letter (A, B, C, or D)\n- No explanations, no additional text, no reasoning shown\n- Wrong answers result in -2 points, correct answers give +1 point\n\nQUESTION:\n{question_text}\n\nOPTIONS:\n{choices_text}\n\nYour answer (single letter only):"
                            }
                        ],
                        "max_completion_tokens": 2500
                    }
                }
                
                repair_requests.append(request)
                
            except (ValueError, IndexError, KeyError, json.JSONDecodeError):
                continue
        
        if not repair_requests:
            print("❌ No valid repair requests created!")
            return None
        
        # Create repair batch file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repair_filename = self.jsonl_dir / f"gpt5_repair_batch_{timestamp}.jsonl"
        
        with open(repair_filename, 'w') as f:
            for request in repair_requests:
                f.write(json.dumps(request) + '\n')
        
        file_size = repair_filename.stat().st_size / 1024 / 1024
        
        print(f"✅ Repair batch created: {repair_filename.name}")
        print(f"📊 Size: {file_size:.2f} MB, {len(repair_requests)} requests")
        
        return repair_filename
    
    def process_failures_and_update(self):
        """Complete workflow: collect failures, create repair batch, submit, and update results"""
        print("🔄 Processing Failures and Updating Results")
        print("=" * 50)
        
        # Step 1: Collect failed requests
        failed_custom_ids = self.collect_failed_requests()
        
        if not failed_custom_ids:
            print("✅ No failures found - all requests completed successfully!")
            return True
        
        # Step 2: Create repair batch
        repair_file = self.create_repair_batch(failed_custom_ids)
        if not repair_file:
            print("❌ Failed to create repair batch!")
            return False
        
        # Step 3: Submit repair batch
        print(f"\n🚀 Submitting repair batch...")
        try:
            with open(repair_file, "rb") as f:
                batch_input_file = self.client.files.create(file=f, purpose="batch")
            
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"Chameleon Repair Batch - {len(failed_custom_ids)} Failed Requests"}
            )
            
            print(f"✅ Repair batch submitted: {batch.id}")
            print(f"📊 Status: {batch.status}")
            print(f"⏱️  Monitor with: python3 chameleon.py monitor-repair {batch.id}")
            
            # Save repair batch info
            repair_info = {
                'batch_id': batch.id,
                'batch_file': str(repair_file),
                'num_requests': len(failed_custom_ids),
                'submitted_at': datetime.now().isoformat(),
                'status': batch.status,
                'type': 'repair'
            }
            
            repair_tracking_file = self.tracking_dir / "repair_batch_info.json"
            with open(repair_tracking_file, 'w') as f:
                json.dump(repair_info, f, indent=2)
            
            return batch.id
            
        except Exception as e:
            print(f"❌ Failed to submit repair batch: {e}")
            return False
    
    def monitor_repair_batch(self, batch_id: str):
        """Monitor a specific repair batch and update results when complete"""
        print(f"👀 Monitoring Repair Batch: {batch_id}")
        print("=" * 50)
        
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            print(f"📊 Status: {batch.status}")
            if batch.request_counts:
                counts = batch.request_counts
                total = counts.total
                completed = counts.completed
                failed = counts.failed
                
                if total > 0:
                    progress = (completed / total) * 100
                    print(f"📈 Progress: {progress:.1f}% ({completed:,}/{total:,})")
                    if failed > 0:
                        print(f"❌ Failed: {failed:,}")
            
            # If completed, download and merge results
            if batch.status == 'completed' and batch.output_file_id:
                print(f"\n🎉 Repair batch completed! Downloading results...")
                
                # Download repair results
                content = self.client.files.content(batch.output_file_id)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                repair_results_file = self.results_dir / f"gpt5_repair_results_{timestamp}.jsonl"
                
                with open(repair_results_file, 'wb') as f:
                    f.write(content.content)
                
                # Parse and merge repair results
                repair_results = {}
                with open(repair_results_file, 'r') as f:
                    for line in f:
                        try:
                            result = json.loads(line)
                            custom_id = result.get('custom_id')
                            if custom_id and result.get('response'):
                                content = result['response']['body']['choices'][0]['message']['content']
                                repair_results[custom_id] = content.strip()
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                print(f"📥 Downloaded {len(repair_results)} repair results")
                
                # Update CSV with repair results
                self._update_csv_with_repair_results(repair_results, timestamp)
                
                # Cleanup repair tracking
                repair_tracking_file = self.tracking_dir / "repair_batch_info.json"
                repair_tracking_file.unlink(missing_ok=True)
                
                return True
            
            else:
                print(f"⏳ Repair batch still processing...")
                return False
                
        except Exception as e:
            print(f"❌ Error monitoring repair batch: {e}")
            return False
    
    def _update_csv_with_repair_results(self, repair_results, timestamp):
        """Update CSV with repair batch results"""
        csv_path = Path("distortions/comprehensive_distortion_dataset.csv")
        
        print(f"📊 Updating CSV with {len(repair_results)} repair results...")
        
        df = pd.read_csv(csv_path)
        
        updated_count = 0
        
        for custom_id, gpt5_answer in repair_results.items():
            try:
                row_idx = int(custom_id.split('_')[1])
                if row_idx < len(df):
                    df.at[row_idx, 'gpt5_answer'] = gpt5_answer
                    updated_count += 1
                    
                    # Calculate correctness
                    if gpt5_answer in ['A', 'B', 'C', 'D']:
                        correct_answer = df.at[row_idx, 'correct_answer']
                        is_correct = (gpt5_answer == correct_answer)
                        df.at[row_idx, 'is_correct'] = is_correct
                    else:
                        df.at[row_idx, 'is_correct'] = False
                        
            except (ValueError, IndexError):
                continue
        
        # Save updated CSV
        output_csv = f"distortions/comprehensive_distortion_dataset_with_repairs_{timestamp}.csv"
        df.to_csv(output_csv, index=False)
        
        print(f"💾 Updated CSV saved: {output_csv}")
        print(f"🔄 Updated {updated_count} entries with repair results")
        
        # Show updated statistics
        valid_answers = df['gpt5_answer'].isin(['A', 'B', 'C', 'D']).sum()
        total_questions = len(df)
        
        print(f"\n📈 Updated Statistics:")
        print(f"   Valid answers: {valid_answers}/{total_questions} ({valid_answers/total_questions*100:.1f}%)")
        
        if valid_answers > 0:
            valid_df = df[df['gpt5_answer'].isin(['A', 'B', 'C', 'D'])]
            overall_accuracy = valid_df['is_correct'].mean() * 100
            print(f"   Overall accuracy: {overall_accuracy:.1f}%")
    
    def _load_tracking(self):
        """Load tracking data"""
        if not self.tracking_file.exists():
            return []
        
        try:
            with open(self.tracking_file, 'r') as f:
                data = json.load(f)
                return data.get('batches', [])
        except:
            return []
    
    def _save_tracking(self, batches):
        """Save tracking data"""
        tracking_data = {
            'total_parts': 10,
            'total_requests': sum(b.get('num_requests', 0) for b in batches),
            'batches': batches,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(self.tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
    
    def _update_tracking(self, new_batches):
        """Update tracking with new batches"""
        existing_batches = self._load_tracking()
        existing_batches.extend(new_batches)
        self._save_tracking(existing_batches)

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT-5 Batch Manager')
    parser.add_argument('command', choices=[
        'create', 'submit', 'monitor', 'cancel', 'cleanup', 'repair', 
        'process-failures', 'monitor-repair', 'reconstruct'
    ], help='Command to execute')
    
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--auto', action='store_true', help='Auto mode (no prompts)')
    parser.add_argument('--failed-only', action='store_true', help='Cleanup failed batches only (keep completed)')
    parser.add_argument('--ids', nargs='+', help='Custom IDs for repair batch')
    parser.add_argument('batch_id', nargs='?', help='Batch ID for monitoring repair')
    
    args = parser.parse_args()
    
    try:
        manager = GPT5BatchManager()
        
        if args.command == 'create':
            manager.create_batches()
            
        elif args.command == 'submit':
            manager.submit_batches(interactive=not args.auto)
            
        elif args.command == 'monitor':
            manager.monitor_batches()
            
        elif args.command == 'cancel':
            manager.cancel_batches()
            
        elif args.command == 'cleanup':
            if hasattr(args, 'failed_only') and args.failed_only:
                manager.cleanup_tracking(cleanup_failed_only=True)
            else:
                manager.cleanup_tracking(auto_cleanup=args.auto)
            
        elif args.command == 'repair':
            if args.ids:
                manager.create_repair_batch(args.ids)
            else:
                manager.create_repair_batch()  # Auto-collect failures
                
        elif args.command == 'process-failures':
            manager.process_failures_and_update()
            
        elif args.command == 'monitor-repair':
            if args.batch_id:
                manager.monitor_repair_batch(args.batch_id)
            else:
                print("❌ monitor-repair requires batch_id parameter")
                
        elif args.command == 'reconstruct':
            manager.reconstruct_batch_tracking()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

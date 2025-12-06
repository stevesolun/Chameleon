"""
Batch Processor for Target Model Evaluation

Handles OpenAI batch API for evaluating model performance on distorted questions.
Consolidates logic from archive/gpt5_manager.py and modules/gpt5_batch_processor.py.
"""

import os
import sys
import json
import time
import math
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from dotenv import load_dotenv

from chameleon.distortion.constants import get_evaluation_prompt


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Batch API discount (constant as per OpenAI docs)
BATCH_DISCOUNT = 0.50  # 50% discount for batch API


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    project_dir: Path
    model: str
    api_key: str
    max_requests_per_batch: int = 50000  # Max based on tier
    completion_window: str = "24h"
    max_completion_tokens: int = 100  # Short answers only
    
    @classmethod
    def from_project(cls, project_name: str, projects_dir: str = "Projects") -> "EvaluationConfig":
        """Create config from project settings."""
        import yaml
        
        project_dir = Path(projects_dir) / project_name
        config_path = project_dir / "config.yaml"
        env_path = project_dir / ".env"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        load_dotenv(env_path)
        
        target_cfg = cfg.get("target_model", {})
        vendor = target_cfg.get("vendor", "openai")
        api_key = os.getenv(f"{vendor.upper()}_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key not found for {vendor}")
        
        return cls(
            project_dir=project_dir,
            model=target_cfg.get("name", "gpt-5.1"),
            api_key=api_key,
        )


class BatchProcessor:
    """
    Handles batch processing for model evaluation.
    
    Workflow:
    1. Create JSONL batch files from distorted data
    2. Upload and submit batches to OpenAI
    3. Monitor batch progress
    4. Download results and update CSV
    5. Handle failures and repairs
    """
    
    def __init__(self, config: EvaluationConfig):
        if not HAS_OPENAI:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.config = config
        self.client = openai.OpenAI(api_key=config.api_key)
        
        # Directories - use eval_batches/ for OpenAI batch files (separate from distortion batches)
        self.batch_dir = config.project_dir / "eval_batches"
        self.jsonl_dir = self.batch_dir / "requests"
        self.results_dir = self.batch_dir / "results"
        self.tracking_dir = self.batch_dir / "tracking"
        
        for d in [self.batch_dir, self.jsonl_dir, self.results_dir, self.tracking_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.tracking_file = self.tracking_dir / "batch_info.json"
        
        # Final results directory
        self.final_results_dir = config.project_dir / "results"
        self.final_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for tier info
        self._tier_info = None
    
    def cleanup_batch_files(self):
        """
        Remove batch files after successful completion.
        Keeps the eval_batches folder structure clean.
        """
        import shutil
        
        try:
            if self.batch_dir.exists():
                shutil.rmtree(self.batch_dir)
                print(f"🧹 Cleaned up batch files: {self.batch_dir}")
        except Exception as e:
            logger.warning(f"Could not cleanup batch files: {e}")
    
    def get_available_models(self) -> List[str]:
        """
        Fetch available models from user's OpenAI account (live).
        
        Returns:
            List of available model IDs
        """
        try:
            models = self.client.models.list()
            return sorted([m.id for m in models.data])
        except Exception as e:
            logger.warning(f"Could not fetch models: {e}")
            return []
    
    def get_gpt_models(self) -> List[str]:
        """
        Get only GPT and o-series models (for evaluation).
        
        Returns:
            Filtered list of chat-capable models
        """
        all_models = self.get_available_models()
        chat_models = []
        
        for m in all_models:
            ml = m.lower()
            # Include GPT models and o-series reasoning models
            if any(prefix in ml for prefix in ['gpt-3.5', 'gpt-4', 'gpt-5', 'o1', 'o3']):
                # Exclude instruct/embedding/audio variants
                if not any(x in ml for x in ['instruct', 'embed', 'whisper', 'tts', 'dall-e']):
                    chat_models.append(m)
        
        return chat_models
    
    def detect_tier_and_limits(self) -> Dict[str, Any]:
        """
        Detect OpenAI account tier, limits, and available models dynamically.
        
        Fetches real-time data from user's OpenAI account:
        - Available models
        - Rate limits
        - Batch API access
        - Organization info
        
        Returns:
            Dict with tier info and limits
        """
        print("\n📊 Fetching OpenAI Account Info (live)...")
        
        tier_info = {
            "batch_api_accessible": False,
            "models": [],
            "rate_limits": {},
            "organization": None,
            "max_requests_per_batch": 50000,  # OpenAI default
            "max_file_size_mb": 100,
        }
        
        try:
            # 1. Get available models
            print("   Fetching available models...")
            models = self.client.models.list()
            model_ids = sorted([m.id for m in models.data])
            tier_info["models"] = model_ids
            print(f"   ✅ {len(model_ids)} models available")
            
            # Check if our target model is available
            target_available = any(self.config.model in m for m in model_ids)
            if not target_available:
                print(f"   ⚠️ Target model '{self.config.model}' not found in available models")
                # Suggest similar models
                similar = [m for m in model_ids if 'gpt' in m.lower()][:5]
                print(f"   Available GPT models: {similar}")
            else:
                print(f"   ✅ Target model '{self.config.model}' is available")
            
            # 2. Check batch API access
            print("   Checking batch API access...")
            try:
                batches = self.client.batches.list(limit=5)
                tier_info["batch_api_accessible"] = True
                tier_info["existing_batches"] = len(list(batches.data))
                
                # Check for any running batches
                running = [b for b in batches.data if b.status in ['validating', 'in_progress', 'finalizing']]
                if running:
                    print(f"   ⚠️ {len(running)} batch(es) currently running")
                    tier_info["running_batches"] = len(running)
                
                print(f"   ✅ Batch API accessible")
            except Exception as e:
                print(f"   ⚠️ Batch API not accessible: {e}")
                tier_info["batch_api_error"] = str(e)
            
            # 3. Try to get rate limits by making a small test
            # (Rate limits are returned in response headers)
            print("   Checking rate limits...")
            try:
                # Make minimal API call to get headers with rate limit info
                # Use gpt-4o-mini for rate limit check (stable, cheap)
                response = self.client.chat.completions.with_raw_response.create(
                    model="gpt-4o-mini",  # Use stable model for rate check
                    messages=[{"role": "user", "content": "OK"}],
                    max_completion_tokens=3
                )
                
                # Extract rate limit headers
                headers = response.headers
                rate_limits = {}
                
                for key in ['x-ratelimit-limit-requests', 'x-ratelimit-limit-tokens', 
                           'x-ratelimit-remaining-requests', 'x-ratelimit-remaining-tokens']:
                    if key in headers:
                        rate_limits[key.replace('x-ratelimit-', '')] = headers[key]
                
                if rate_limits:
                    tier_info["rate_limits"] = rate_limits
                    
                    # Determine tier based on rate limits
                    req_limit = int(rate_limits.get('limit-requests', 0))
                    if req_limit >= 10000:
                        tier_info["tier"] = "Tier 5 (highest)"
                    elif req_limit >= 5000:
                        tier_info["tier"] = "Tier 4"
                    elif req_limit >= 500:
                        tier_info["tier"] = "Tier 3"
                    elif req_limit >= 100:
                        tier_info["tier"] = "Tier 2"
                    else:
                        tier_info["tier"] = "Tier 1 (free)"
                    
                    print(f"   ✅ Rate limits retrieved")
                    print(f"   Detected tier: {tier_info.get('tier', 'Unknown')}")
                    print(f"   Requests/min: {rate_limits.get('limit-requests', 'N/A')}")
                    print(f"   Tokens/min: {rate_limits.get('limit-tokens', 'N/A')}")
                    
            except Exception as e:
                print(f"   ⚠️ Could not fetch rate limits: {e}")
            
            return tier_info
            
        except Exception as e:
            print(f"   ❌ Error fetching account info: {e}")
            tier_info["error"] = str(e)
            return tier_info
    
    def fetch_model_pricing(self, model_id: str) -> Dict[str, float]:
        """
        Fetch pricing for a model dynamically based on its family.
        
        OpenAI doesn't have a public pricing API, so we categorize models 
        by family from the live model list and apply known pricing tiers.
        
        Args:
            model_id: The model ID to get pricing for
            
        Returns:
            Dict with input/output costs per 1M tokens
        """
        m = model_id.lower()
        
        # Pricing by model family (USD per 1M tokens) - auto-detected from model name
        # These align with OpenAI's pricing page patterns
        
        # GPT-5.x family (flagship 2025)
        if 'gpt-5' in m:
            if 'mini' in m:
                return {"input": 0.30, "output": 1.20}
            return {"input": 5.00, "output": 20.00}
        
        # GPT-4o family
        if 'gpt-4o' in m:
            if 'mini' in m:
                return {"input": 0.15, "output": 0.60}
            return {"input": 2.50, "output": 10.00}
        
        # GPT-4 Turbo
        if 'gpt-4-turbo' in m or 'gpt-4-1106' in m or 'gpt-4-0125' in m:
            return {"input": 10.00, "output": 30.00}
        
        # GPT-4 base
        if 'gpt-4' in m:
            return {"input": 30.00, "output": 60.00}
        
        # o1 reasoning models
        if m.startswith('o1'):
            if 'mini' in m:
                return {"input": 3.00, "output": 12.00}
            return {"input": 15.00, "output": 60.00}
        
        # o3 reasoning models
        if m.startswith('o3'):
            if 'mini' in m:
                return {"input": 1.10, "output": 4.40}
            return {"input": 10.00, "output": 40.00}
        
        # GPT-3.5 Turbo
        if 'gpt-3.5' in m:
            return {"input": 0.50, "output": 1.50}
        
        # Default fallback
        return {"input": 5.00, "output": 15.00}
    
    def estimate_cost(self, num_requests: int, avg_input_tokens: int = 500, avg_output_tokens: int = 10) -> Dict[str, float]:
        """
        Estimate cost for evaluation using live model pricing.
        
        Args:
            num_requests: Number of questions to evaluate
            avg_input_tokens: Average input tokens per request
            avg_output_tokens: Average output tokens per response
        
        Returns:
            Dict with cost breakdown
        """
        # Get pricing for the configured model (dynamic)
        costs = self.fetch_model_pricing(self.config.model)
        
        total_input_tokens = num_requests * avg_input_tokens
        total_output_tokens = num_requests * avg_output_tokens
        
        # Calculate raw costs
        input_cost = (total_input_tokens / 1_000_000) * costs["input"]
        output_cost = (total_output_tokens / 1_000_000) * costs["output"]
        
        # Apply batch API discount (50%)
        discounted_input = input_cost * (1 - BATCH_DISCOUNT)
        discounted_output = output_cost * (1 - BATCH_DISCOUNT)
        total_cost = discounted_input + discounted_output
        
        return {
            "num_requests": num_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "input_cost_usd": round(discounted_input, 4),
            "output_cost_usd": round(discounted_output, 4),
            "total_cost_usd": round(total_cost, 2),
            "total_cost_without_batch": round(input_cost + output_cost, 2),
            "model": self.config.model,
            "batch_discount": f"{int(BATCH_DISCOUNT * 100)}%",
            "pricing_per_1m": costs,
        }
    
    def confirm_submission(self, num_requests: int, num_batches: int, cost_estimate: Dict, tier_info: Dict = None) -> bool:
        """
        Show cost estimate with tier info and ask user for confirmation.
        
        Returns:
            True if user confirms, False otherwise
        """
        print("\n" + "═" * 60)
        print("💰 EVALUATION COST CONFIRMATION (Live Data)")
        print("═" * 60)
        
        # Show tier info if available
        if tier_info:
            tier = tier_info.get('tier', 'Unknown')
            rate_limits = tier_info.get('rate_limits', {})
            print(f"""
   👤 Account Information (fetched live):
      Tier: {tier}
      Requests/min: {rate_limits.get('limit-requests', 'N/A')}
      Tokens/min: {rate_limits.get('limit-tokens', 'N/A')}
      Available models: {len(tier_info.get('models', []))}""")
        
        pricing = cost_estimate.get('pricing_per_1m', {})
        print(f"""
   📊 Dataset Summary:
      Total requests: {num_requests:,}
      Model: {cost_estimate['model']}
      Batches: {num_batches}
   
   💰 Pricing (per 1M tokens):
      Input:  ${pricing.get('input', 'N/A'):.2f}
      Output: ${pricing.get('output', 'N/A'):.2f}
   
   💵 Cost Estimate:
      Input tokens:  ~{cost_estimate['total_input_tokens']:,} (${cost_estimate['input_cost_usd']:.4f})
      Output tokens: ~{cost_estimate['total_output_tokens']:,} (${cost_estimate['output_cost_usd']:.4f})
      Batch discount: {cost_estimate['batch_discount']} applied
      ─────────────────────────────────
      Without batch: ${cost_estimate.get('total_cost_without_batch', 'N/A')}
      ESTIMATED TOTAL: ${cost_estimate['total_cost_usd']:.2f}
   
   ⚠️  This will charge your OpenAI account.
""")
        print("═" * 60)
        
        while True:
            choice = input("\nProceed with evaluation? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                print("✅ Confirmed. Starting evaluation...")
                return True
            elif choice in ['n', 'no']:
                print("❌ Cancelled by user.")
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def _load_distorted_data(self) -> pd.DataFrame:
        """Load the completed distortions CSV."""
        csv_path = self.config.project_dir / "distorted_data" / "distortions_complete.csv"
        
        if not csv_path.exists():
            # Fall back to in-progress
            csv_path = self.config.project_dir / "distorted_data" / "distortions_in_progress.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"No distorted data found in {self.config.project_dir / 'distorted_data'}")
        
        return pd.read_csv(csv_path, encoding='utf-8')
    
    def create_batches(self) -> List[Path]:
        """
        Create JSONL batch files from distorted data.
        
        Returns:
            List of created batch file paths
        """
        print("🚀 Creating Evaluation Batches")
        print("=" * 50)
        
        df = self._load_distorted_data()
        print(f"✅ Loaded {len(df)} questions")
        print(f"   Columns: {list(df.columns)}")
        
        # Create requests
        batch_requests = []
        for idx, row in df.iterrows():
            try:
                # Parse answer options (handle both column names)
                options_raw = row.get('options_json', row.get('answer_options', '{}'))
                if isinstance(options_raw, str):
                    try:
                        answer_options = json.loads(options_raw)
                    except json.JSONDecodeError:
                        answer_options = {"A": "", "B": "", "C": "", "D": ""}
                else:
                    answer_options = options_raw if options_raw else {"A": "", "B": "", "C": "", "D": ""}
                
                # Get distorted question
                question_text = row.get('distorted_question', row.get('question_text', ''))
                if not question_text or pd.isna(question_text):
                    continue
                
                # Build precise custom_id for matching back
                q_id = row.get('question_id', f'row_{idx}')
                d_id = row.get('distortion_id', 0)
                miu = row.get('miu', 0.0)
                custom_id = f"{q_id}__d{d_id}__miu{miu}__idx{idx}"
                
                # Create evaluation prompt
                prompt = get_evaluation_prompt(str(question_text), answer_options)
                
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": self.config.max_completion_tokens
                    }
                }
                batch_requests.append(request)
                
            except Exception as e:
                logger.warning(f"Skipping row {idx}: {e}")
                continue
        
        print(f"✅ Created {len(batch_requests)} requests")
        
        # Split into batches
        batch_files = self._split_requests(batch_requests)
        
        print(f"\n✅ Created {len(batch_files)} batch files")
        for i, bf in enumerate(batch_files, 1):
            size_mb = bf.stat().st_size / 1024 / 1024
            with open(bf, 'r') as f:
                count = sum(1 for _ in f)
            print(f"   Part {i}: {bf.name} ({size_mb:.1f} MB, {count:,} requests)")
        
        return batch_files
    
    def _split_requests(self, requests: List[dict]) -> List[Path]:
        """
        Split requests into batch files based on max_requests_per_batch.
        
        Uses tier-detected limits to optimize batch sizes.
        """
        total = len(requests)
        max_per_batch = self.config.max_requests_per_batch
        num_batches = math.ceil(total / max_per_batch)
        
        print(f"🔧 Splitting {total:,} requests into {num_batches} batch(es)")
        print(f"   Max per batch: {max_per_batch:,} (from tier detection)")
        
        batch_files = []
        
        for i in range(num_batches):
            start = i * max_per_batch
            end = min((i + 1) * max_per_batch, total)
            
            if start >= total:
                break
            
            batch = requests[start:end]
            filename = self.jsonl_dir / f"eval_batch_part_{i+1:02d}.jsonl"
            
            with open(filename, 'w', encoding='utf-8') as f:
                for req in batch:
                    f.write(json.dumps(req) + '\n')
            
            batch_files.append(filename)
            print(f"   Part {i+1}: {len(batch):,} requests")
        
        return batch_files
    
    def submit_batches(self, interactive: bool = True) -> bool:
        """
        Submit batch files to OpenAI.
        
        Args:
            interactive: Whether to prompt for confirmation
        
        Returns:
            True if submission successful
        """
        print("🚀 Submitting Evaluation Batches")
        print("=" * 40)
        
        batch_files = sorted(self.jsonl_dir.glob("eval_batch_part_*.jsonl"))
        if not batch_files:
            print("❌ No batch files found! Run create_batches first.")
            return False
        
        tracking = self._load_tracking()
        submitted_files = {b['batch_file'] for b in tracking}
        
        available = []
        for bf in batch_files:
            status = "✅ SUBMITTED" if str(bf) in submitted_files else "⏳ PENDING"
            with open(bf, 'r') as f:
                count = sum(1 for _ in f)
            print(f"   {bf.name}: {count:,} requests - {status}")
            
            if str(bf) not in submitted_files:
                available.append(bf)
        
        if not available:
            print("✅ All batches already submitted!")
            return True
        
        if interactive:
            choice = input(f"\nSubmit {len(available)} pending batch(es)? (y/n): ").strip().lower()
            if choice != 'y':
                print("❌ Cancelled")
                return False
        
        # Submit batches
        new_submissions = []
        
        for i, bf in enumerate(available, 1):
            print(f"\n🚀 Submitting Part {i}/{len(available)}: {bf.name}...")
            
            try:
                with open(bf, "rb") as f:
                    uploaded = self.client.files.create(file=f, purpose="batch")
                
                batch = self.client.batches.create(
                    input_file_id=uploaded.id,
                    endpoint="/v1/chat/completions",
                    completion_window=self.config.completion_window,
                    metadata={
                        "description": f"Chameleon Evaluation - {bf.name}",
                        "project": self.config.project_dir.name
                    }
                )
                
                with open(bf, 'r') as f:
                    num_requests = sum(1 for _ in f)
                
                new_submissions.append({
                    'part': i,
                    'batch_id': batch.id,
                    'batch_file': str(bf),
                    'num_requests': num_requests,
                    'submitted_at': datetime.now().isoformat(),
                    'status': batch.status
                })
                
                print(f"✅ Submitted: {batch.id}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        if new_submissions:
            self._update_tracking(new_submissions)
            print(f"\n🎉 Submitted {len(new_submissions)} batch(es)!")
        
        return len(new_submissions) > 0
    
    def monitor(self) -> bool:
        """
        Monitor batch progress and download completed results.
        
        Returns:
            True if all batches completed
        """
        print("👀 Monitoring Evaluation Batches")
        print("=" * 40)
        
        tracking = self._load_tracking()
        if not tracking:
            print("❌ No batches to monitor!")
            return False
        
        all_completed = True
        newly_completed = []
        
        for batch_info in tracking:
            batch_id = batch_info['batch_id']
            part = batch_info['part']
            
            try:
                batch = self.client.batches.retrieve(batch_id)
                
                print(f"\n📊 Part {part}: {batch.status}")
                
                if batch.request_counts:
                    total = batch.request_counts.total
                    done = batch.request_counts.completed
                    failed = batch.request_counts.failed
                    
                    if total > 0:
                        pct = done * 100 / total
                        print(f"   Progress: {pct:.1f}% ({done:,}/{total:,})")
                    if failed > 0:
                        print(f"   Failed: {failed:,}")
                
                if batch.status == 'completed' and not batch_info.get('downloaded'):
                    newly_completed.append({
                        'batch_id': batch_id,
                        'part': part,
                        'output_file_id': batch.output_file_id
                    })
                    batch_info['downloaded'] = True
                
                if batch.status not in ['completed', 'failed', 'cancelled']:
                    all_completed = False
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                all_completed = False
        
        # Download newly completed
        if newly_completed:
            print(f"\n📥 Downloading {len(newly_completed)} completed batch(es)...")
            self._download_results(newly_completed)
            self._save_tracking(tracking)
        
        # Summary
        completed = sum(1 for b in tracking if b.get('downloaded'))
        print(f"\n📊 Status: {completed}/{len(tracking)} batches completed")
        
        if all_completed:
            print("\n🎉 All batches finished!")
        
        return all_completed
    
    def _download_results(self, completed_batches: List[Dict]):
        """Download and parse results from completed batches."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = {}
        
        for info in completed_batches:
            if not info.get('output_file_id'):
                continue
            
            print(f"📥 Downloading Part {info['part']}...")
            
            try:
                content = self.client.files.content(info['output_file_id'])
                results_file = self.results_dir / f"eval_results_part_{info['part']}_{timestamp}.jsonl"
                
                with open(results_file, 'wb') as f:
                    f.write(content.content)
                
                # Parse results
                with open(results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result = json.loads(line)
                            custom_id = result.get('custom_id')
                            if custom_id and result.get('response'):
                                answer = result['response']['body']['choices'][0]['message']['content'].strip()
                                all_results[custom_id] = answer
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                print(f"   ✅ Got {len(all_results)} results")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        if all_results:
            self._update_csv_with_results(all_results, timestamp)
    
    def _normalize_answer(self, answer: str) -> set:
        """
        Normalize an answer string to a set of uppercase letters.
        
        Handles:
        - Single: "A", "a", "A.", "a:"
        - Multiple: "A, D", "A,D", "a d", "A D", "D, A" -> same set
        - Case-insensitive: "a" == "A"
        - Order-independent: "A, D" == "D, A"
        
        Args:
            answer: The answer string from model or ground truth
        
        Returns:
            Set of uppercase letters (e.g., {'A', 'D'})
        """
        if not answer or pd.isna(answer):
            return set()
        
        # Convert to string and uppercase
        answer_str = str(answer).upper()
        
        # Remove common separators and punctuation
        for char in [',', '.', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'"]:
            answer_str = answer_str.replace(char, ' ')
        
        # Extract only valid answer letters
        valid_letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}
        letters = set()
        
        for token in answer_str.split():
            token = token.strip()
            if len(token) == 1 and token in valid_letters:
                letters.add(token)
        
        return letters
    
    def _check_correctness(self, model_answer: str, correct_answer: str) -> bool:
        """
        Check if model answer is correct using set comparison.
        
        Order-independent: "A, D" == "D, A" -> True
        Case-insensitive: "a" == "A" -> True
        
        Args:
            model_answer: The answer from the target model
            correct_answer: The ground truth answer
        
        Returns:
            True if answers match (same set of letters)
        """
        model_set = self._normalize_answer(model_answer)
        correct_set = self._normalize_answer(correct_answer)
        
        if not model_set or not correct_set:
            return False
        
        return model_set == correct_set
    
    def _update_csv_with_results(self, results: Dict[str, str], timestamp: str):
        """Update the distortions CSV with model answers."""
        csv_path = self.config.project_dir / "distorted_data" / "distortions_complete.csv"
        
        if not csv_path.exists():
            csv_path = self.config.project_dir / "distorted_data" / "distortions_in_progress.csv"
        
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        print(f"\n📊 Updating CSV with {len(results)} results...")
        
        # Ensure target columns exist
        if 'target_model_answer' not in df.columns:
            df['target_model_answer'] = ''
        if 'is_correct' not in df.columns:
            df['is_correct'] = None
        if 'target_model_name' not in df.columns:
            df['target_model_name'] = self.config.model
        
        matched = 0
        valid_answers = 0
        correct_count = 0
        
        # Parse custom_id format: {question_id}__d{distortion_id}__miu{miu}__idx{idx}
        for custom_id, answer in results.items():
            try:
                # Extract the row index from custom_id
                parts = custom_id.split('__')
                idx = None
                
                # Find idx part
                for part in parts:
                    if part.startswith('idx'):
                        idx = int(part[3:])
                        break
                
                if idx is None or idx >= len(df):
                    continue
                
                # Clean the model answer (remove any extra whitespace)
                cleaned_answer = str(answer).strip().upper() if answer else ''
                
                # Update the row
                df.at[idx, 'target_model_answer'] = cleaned_answer
                df.at[idx, 'target_model_name'] = self.config.model
                matched += 1
                
                # Get ground truth answer
                correct = df.at[idx, 'answer'] if 'answer' in df.columns else df.at[idx, 'correct_answer']
                
                # Normalize and compare
                model_letters = self._normalize_answer(cleaned_answer)
                
                if model_letters:
                    valid_answers += 1
                    is_correct = self._check_correctness(cleaned_answer, correct)
                    df.at[idx, 'is_correct'] = is_correct
                    if is_correct:
                        correct_count += 1
                else:
                    df.at[idx, 'is_correct'] = False
                    
            except Exception as e:
                logger.warning(f"Error processing result {custom_id}: {e}")
                continue
        
        # Ensure ALL rows have target_model_name populated (fix for any missed rows)
        df['target_model_name'] = df['target_model_name'].fillna(self.config.model)
        df.loc[df['target_model_name'] == '', 'target_model_name'] = self.config.model
        
        # Update the distortions_complete.csv with answers (for consistency)
        complete_csv = self.config.project_dir / "distorted_data" / "distortions_complete.csv"
        df.to_csv(complete_csv, index=False, encoding='utf-8')
        
        # Save final results to results/results.csv
        final_csv = self.final_results_dir / "results.csv"
        df.to_csv(final_csv, index=False, encoding='utf-8')
        
        print(f"💾 Results saved to: {final_csv}")
        print(f"\n🔗 Matched: {matched}/{len(df)} ({matched*100/len(df):.1f}%)")
        print(f"✅ Valid answers: {valid_answers}/{matched if matched > 0 else 1}")
        
        # Performance analysis
        if valid_answers > 0:
            accuracy = correct_count * 100 / valid_answers
            print(f"\n📈 Model Performance: {accuracy:.1f}% accuracy ({correct_count}/{valid_answers})")
            
            # By miu
            if 'miu' in df.columns:
                print("\n📊 Accuracy by μ:")
                for miu in sorted(df['miu'].unique()):
                    subset = df[(df['miu'] == miu) & (df['is_correct'].notna())]
                    if len(subset) > 0:
                        miu_correct = subset['is_correct'].sum()
                        miu_acc = miu_correct * 100 / len(subset)
                        print(f"   μ={miu:.1f}: {miu_acc:.1f}% ({miu_correct}/{len(subset)} correct)")
    
    def _load_tracking(self) -> List[Dict]:
        """Load batch tracking data."""
        if not self.tracking_file.exists():
            return []
        
        try:
            with open(self.tracking_file, 'r') as f:
                data = json.load(f)
                return data.get('batches', [])
        except:
            return []
    
    def _save_tracking(self, batches: List[Dict]):
        """Save batch tracking data."""
        data = {
            'batches': batches,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.tracking_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_tracking(self, new_batches: List[Dict]):
        """Add new batches to tracking."""
        existing = self._load_tracking()
        existing.extend(new_batches)
        self._save_tracking(existing)


def run_evaluation(project_name: str, projects_dir: str = "Projects", skip_confirmation: bool = False) -> Dict[str, Any]:
    """
    Run the complete evaluation workflow with tier detection and cost confirmation.
    
    Interactive workflow:
    1. Load distortions_complete.csv
    2. Fill target_model_name column
    3. Check user's OpenAI tier and calculate max batch size
    4. Split requests into optimal batches (e.g., 50k + 10k for 60k requests)
    5. Show cost estimate and ask for confirmation
    6. Submit batches to OpenAI Batch API
    7. Monitor progress and download results
    8. Populate target_model_answer column
    9. Smart comparison: is_correct = (normalized model answer == normalized correct answer)
    10. Save final results to results/results.csv
    
    Args:
        project_name: Name of the project
        projects_dir: Base directory for projects
        skip_confirmation: If True, skip cost confirmation (for automated runs)
    
    Returns:
        Summary dict
    """
    print("\n" + "═" * 60)
    print("🎯 CHAMELEON EVALUATION PIPELINE")
    print("═" * 60)
    
    config = EvaluationConfig.from_project(project_name, projects_dir)
    processor = BatchProcessor(config)
    
    print(f"\n📁 Project: {project_name}")
    print(f"🤖 Target Model: {config.model}")
    
    # Step 0: Detect tier and get limits
    print("\n" + "-" * 40)
    print("STEP 0: Detecting OpenAI Account Tier")
    print("-" * 40)
    
    tier_info = processor.detect_tier_and_limits()
    
    if not tier_info.get("batch_api_accessible"):
        print(f"\n❌ Batch API not accessible: {tier_info.get('batch_api_error', 'Unknown error')}")
        return {"status": "error", "message": "Batch API not accessible"}
    
    # Step 1: Load and analyze data
    print("\n" + "-" * 40)
    print("STEP 1: Loading Distortions Data")
    print("-" * 40)
    
    df = processor._load_distorted_data()
    total_rows = len(df)
    
    # Validate required columns
    required_cols = ['question_id', 'distortion_id', 'distorted_question', 'miu', 'options_json', 'answer']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return {"status": "error", "message": f"Missing columns: {missing_cols}"}
    
    print(f"✅ Loaded {total_rows:,} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Check what needs evaluation
    if 'target_model_answer' not in df.columns:
        df['target_model_answer'] = ''
    
    needs_eval_mask = df['target_model_answer'].isna() | (df['target_model_answer'] == '')
    needs_eval = df[needs_eval_mask]
    num_requests = len(needs_eval)
    already_done = total_rows - num_requests
    
    print(f"\n📊 Evaluation Status:")
    print(f"   Total questions: {total_rows:,}")
    print(f"   Already evaluated: {already_done:,}")
    print(f"   Need evaluation: {num_requests:,}")
    
    if num_requests == 0:
        print("\n✅ All questions already evaluated!")
        
        # Still generate results file
        final_csv = processor.final_results_dir / "results.csv"
        df.to_csv(final_csv, index=False, encoding='utf-8')
        print(f"📁 Results saved to: {final_csv}")
        
        return {"status": "complete", "message": "Already complete", "results_file": str(final_csv)}
    
    # Step 2: Validate data before submission
    print("\n" + "-" * 40)
    print("STEP 2: Validating Data for Submission")
    print("-" * 40)
    
    validation_errors = []
    
    # Check each row that needs evaluation
    for idx, row in needs_eval.head(100).iterrows():  # Sample first 100
        q_id = row.get('question_id', 'MISSING')
        d_id = row.get('distortion_id', 'MISSING')
        question = row.get('distorted_question', '')
        options = row.get('options_json', '')
        miu = row.get('miu', -1)
        answer = row.get('answer', '')
        
        if not question or pd.isna(question):
            validation_errors.append(f"Row {idx}: Missing distorted_question")
        if not options or pd.isna(options):
            validation_errors.append(f"Row {idx}: Missing options_json")
        if miu < 0 or pd.isna(miu):
            validation_errors.append(f"Row {idx}: Invalid miu value")
        if not answer or pd.isna(answer):
            validation_errors.append(f"Row {idx}: Missing answer")
    
    if validation_errors:
        print(f"⚠️  Found {len(validation_errors)} validation issues (showing first 10):")
        for err in validation_errors[:10]:
            print(f"   • {err}")
        if len(validation_errors) > 10:
            print(f"   ... and {len(validation_errors) - 10} more")
        
        if not skip_confirmation:
            choice = input("\nContinue anyway? (y/n): ").strip().lower()
            if choice != 'y':
                return {"status": "cancelled", "message": "Cancelled due to validation errors"}
    else:
        print("✅ All sampled rows passed validation")
    
    # Step 3: Calculate optimal batch sizes
    print("\n" + "-" * 40)
    print("STEP 3: Calculating Optimal Batch Sizes")
    print("-" * 40)
    
    max_per_batch = tier_info.get("max_requests_per_batch", 50000)
    num_batches = math.ceil(num_requests / max_per_batch)
    
    print(f"   Max requests per batch: {max_per_batch:,}")
    print(f"   Total requests: {num_requests:,}")
    print(f"   Batches needed: {num_batches}")
    
    # Show batch breakdown
    print(f"\n   Batch breakdown:")
    remaining = num_requests
    for i in range(num_batches):
        batch_size = min(max_per_batch, remaining)
        print(f"      Batch {i+1}: {batch_size:,} requests")
        remaining -= batch_size
    
    # Step 4: Cost estimate and confirmation
    print("\n" + "-" * 40)
    print("STEP 4: Cost Estimation")
    print("-" * 40)
    
    cost_estimate = processor.estimate_cost(num_requests)
    
    if not skip_confirmation:
        if not processor.confirm_submission(num_requests, num_batches, cost_estimate, tier_info):
            return {"status": "cancelled", "message": "Cancelled by user"}
    
    # Step 5: Create batches
    print("\n" + "-" * 40)
    print("STEP 5: Creating Batch Files")
    print("-" * 40)
    
    processor.config.max_requests_per_batch = max_per_batch
    batch_files = processor.create_batches()
    
    if not batch_files:
        return {"status": "error", "message": "No batch files created"}
    
    # Step 6: Submit batches
    print("\n" + "-" * 40)
    print("STEP 6: Submitting to OpenAI Batch API")
    print("-" * 40)
    
    if not processor.submit_batches(interactive=False):
        return {"status": "error", "message": "Failed to submit batches"}
    
    # Step 7: Monitor until complete
    print("\n" + "-" * 40)
    print("STEP 7: Monitoring Batch Progress")
    print("-" * 40)
    print("⏳ Checking every 60 seconds...")
    
    check_interval = 60  # 1 minute
    
    while True:
        if processor.monitor():
            break
        print(f"\n⏳ Still processing... checking again in {check_interval} seconds")
        time.sleep(check_interval)
    
    # Step 8: Save final results
    print("\n" + "-" * 40)
    print("STEP 8: Saving Final Results")
    print("-" * 40)
    
    final_results_path = processor.final_results_dir / "results.csv"
    
    # Load updated data with answers
    final_df = processor._load_distorted_data()
    
    # Ensure all required columns are filled
    if 'target_model_name' not in final_df.columns:
        final_df['target_model_name'] = config.model
    final_df['target_model_name'] = final_df['target_model_name'].fillna(config.model)
    
    # Save to results folder
    final_df.to_csv(final_results_path, index=False, encoding='utf-8')
    
    # Calculate final statistics
    valid_answers = final_df[final_df['target_model_answer'].notna() & (final_df['target_model_answer'] != '')]
    correct_count = valid_answers['is_correct'].sum() if 'is_correct' in valid_answers.columns else 0
    
    print(f"\n✅ Evaluation Complete!")
    print(f"   Results saved to: {final_results_path}")
    print(f"   Total evaluated: {len(valid_answers):,}")
    print(f"   Correct answers: {correct_count:,}")
    if len(valid_answers) > 0:
        print(f"   Overall accuracy: {correct_count * 100 / len(valid_answers):.1f}%")
    
    # Step 9: Cleanup batch files
    print("\n" + "-" * 40)
    print("STEP 9: Cleanup")
    print("-" * 40)
    
    processor.cleanup_batch_files()
    print("✅ Batch files cleaned up")
    
    # Final summary
    print("\n" + "═" * 60)
    print("📊 FINAL SUMMARY")
    print("═" * 60)
    print(f"   Project: {project_name}")
    print(f"   Model: {config.model}")
    print(f"   Questions evaluated: {len(valid_answers):,}")
    print(f"   Accuracy: {correct_count * 100 / len(valid_answers):.1f}%" if len(valid_answers) > 0 else "   Accuracy: N/A")
    print(f"\n   📁 Results file: {final_results_path}")
    print(f"   Columns: target_model_name, target_model_answer, is_correct")
    print("═" * 60)
    
    # Offer analysis option
    print("\n")
    choice = input("📊 Run analysis now? (Y/n): ").strip().lower()
    
    if choice != 'n':
        print("\n🔬 Starting analysis...")
        try:
            from chameleon.analysis import run_full_analysis
            analysis_result = run_full_analysis(project_name, projects_dir)
            
            if analysis_result.get("status") == "complete":
                print(f"\n✅ Analysis complete!")
                print(f"   Output: {analysis_result.get('output_dir')}")
                print(f"   Executive Report: {projects_dir}/{project_name}/results/Executive_Report.md")
        except Exception as e:
            print(f"\n⚠️ Analysis error: {e}")
            print(f"   You can run analysis later with: python cli.py analyze --project {project_name}")
    else:
        print(f"\n💡 Run analysis later with: python cli.py analyze --project {project_name}")
    
    return {
        "status": "complete",
        "requests": num_requests,
        "batches": num_batches,
        "cost_estimate": cost_estimate,
        "results_file": str(final_results_path),
        "accuracy": correct_count * 100 / len(valid_answers) if len(valid_answers) > 0 else 0
    }


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on distorted questions")
    parser.add_argument("command", choices=["create", "submit", "monitor", "run"], help="Command")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--projects-dir", default="Projects", help="Projects directory")
    
    args = parser.parse_args()
    
    config = EvaluationConfig.from_project(args.project, args.projects_dir)
    processor = BatchProcessor(config)
    
    if args.command == "create":
        processor.create_batches()
    elif args.command == "submit":
        processor.submit_batches()
    elif args.command == "monitor":
        processor.monitor()
    elif args.command == "run":
        run_evaluation(args.project, args.projects_dir)


"""
Data Preparation Module for Question Distortion
Generates distorted questions for multiple subjects and miu values
"""

import json
import yaml
import requests
import time
import logging
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class MistralServer:
    """Configuration for a Mistral inference server"""
    url: str
    name: str
    subjects: List[str]

@dataclass
class ProcessingConfig:
    """Configuration for data preparation processing"""
    subjects: List[Dict[str, Any]]
    distortions_per_question: int
    output_directory: str
    mistral_servers: List[MistralServer]
    max_batch_size: int
    max_retries: int = 3
    retry_delay: float = 1.0
    delay_between_batches: float = 0.5
    max_concurrent_workers: int = 2
    worker_timeout: int = 600
    request_timeout: int = 300

class MMLULoader:
    """Loads questions from MMLU dataset"""
    
    def __init__(self):
        # ALL available MMLU subjects
        self.available_subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
            'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology', 'high_school_statistics',
            'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
            'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
            'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
            'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
            'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
            'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
        ]
    
    def load_questions_from_subjects(self, subject_configs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Load questions for multiple subjects from MMLU"""
        all_questions = {}
        
        try:
            from datasets import load_dataset
            
            for config in subject_configs:
                subject_name = config.get('name')
                question_count = config.get('question_count', 10)
                
                if subject_name not in self.available_subjects:
                    logger.warning(f"Subject '{subject_name}' not found in MMLU. Available: {', '.join(self.available_subjects[:10])}...")
                    continue
                
                logger.info(f"Loading {question_count} questions for {subject_name} from MMLU...")
                
                # Load MMLU dataset for this subject
                dataset = load_dataset("cais/mmlu", subject_name, split="test")
                
                questions = []
                for i, item in enumerate(dataset):
                    if i >= question_count:
                        break
                    
                    # Convert MMLU format to our format
                    question_text = item["question"]
                    choices = item["choices"]
                    
                    # Format as multiple choice question
                    formatted_question = f"{question_text}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
                    
                    questions.append({
                        "question_id": i + 1,
                        "question_text": question_text,  # Original question without choices
                        "choices": choices,  # Separate choices array
                        "correct_answer": chr(65 + item["answer"]),  # Convert 0,1,2,3 to A,B,C,D
                        "difficulty": "medium",
                        "topic": subject_name
                    })
                
                all_questions[subject_name] = questions
                logger.info(f"✅ Loaded {len(questions)} questions for {subject_name}")
            
            return all_questions
            
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            return {}
        except Exception as e:
            logger.error(f"Failed to load MMLU data: {e}")
            return {}

class QuestionLoader:
    """Simple loader for questions.json"""
    
    def __init__(self, questions_file: str):
        self.questions_file = Path(questions_file)
        self.questions_data = {}
        self.load_questions()
    
    def load_questions(self):
        """Load questions from JSON file"""
        try:
            if self.questions_file.exists():
                with open(self.questions_file, 'r') as f:
                    self.questions_data = json.load(f)
                logger.info(f"Loaded questions from {self.questions_file}")
            else:
                logger.info(f"Questions file not found: {self.questions_file}")
                self.questions_data = {}
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            self.questions_data = {}
    
    def get_questions(self, subject: str, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get questions for a specific subject"""
        questions = self.questions_data.get(subject, [])
        
        if count and len(questions) > count:
            questions = questions[:count]
        
        return questions
    
    def save_questions(self, questions_dict: Dict[str, List[Dict[str, Any]]]) -> str:
        """Save questions dict to JSON file"""
        self.questions_data = questions_dict
        
        # Ensure the data directory exists
        self.questions_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.questions_file, 'w') as f:
            json.dump(self.questions_data, f, indent=2)
        
        logger.info(f"💾 Saved {len(self.questions_data)} subjects to {self.questions_file}")
        return str(self.questions_file)

class MistralDistortionClient:
    """Client for calling Mistral distortion endpoint"""
    
    def __init__(self, server_url: str, max_retries: int = 3, retry_delay: float = 1.0):
        self.server_url = server_url.rstrip('/')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.distort_endpoint = f"{self.server_url}/distort-questions"
        self.request_timeout = 3600  # 1 hour timeout for mega requests
    
    def check_server_health(self) -> bool:
        """Check if Mistral server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            health_data = response.json()
            return health_data.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Server health check failed: {e}")
            return False
    
    def distort_batch(self, subject: str, miu: float, questions: List[Dict[str, Any]], worker_id: str = "main") -> Optional[Dict[str, Any]]:
        """Send a batch of questions for distortion"""
        
        # Prepare batch data
        batch_data = {
            "subject": subject,
            "miu": miu,
            "questions": []
        }
        
        for question in questions:
            batch_data["questions"].append({
                "question_id": question["question_id"],
                "question_text": question["question_text"],
                "distorted_question_text": ""
            })
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:  # Only log first attempt to reduce verbosity
                    logger.info(f"[{worker_id}] 📡 {subject} μ={miu:.1f} - Sending {len(questions)} questions to server")
                
                response = requests.post(
                    self.distort_endpoint,
                    json={"batch_data": batch_data, "max_tokens": 2000},
                    timeout=self.request_timeout  # Use configured timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"[{worker_id}] ✅ {subject} μ={miu:.1f} - Successfully distorted {len(questions)} questions")
                    return result
                else:
                    logger.error(f"[{worker_id}] ⚠️ {subject} μ={miu:.1f} - Server returned status {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.error(f"[{worker_id}] ❌ {subject} μ={miu:.1f} - Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"[{worker_id}] ⏳ {subject} μ={miu:.1f} - Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)  # Exponential backoff
        
        logger.error(f"[{worker_id}] 💥 {subject} μ={miu:.1f} - FAILED after {self.max_retries} attempts")
        return None

    def distort_complete_batch(self, subject: str, miu: float, batch_structure: Dict[str, Any], worker_id: str = "main") -> Optional[Dict[str, Any]]:
        """Send a complete batch structure for Mistral to fill with ALL distortions at once"""
        
        # Handle MEGA format
        if "miu_configurations" in batch_structure:
            miu_configs = batch_structure.get("miu_configurations", [])
            questions_count = len(miu_configs[0].get("questions_and_distortions", [])) if miu_configs else 0
            total_configs = len(miu_configs)
        else:
            questions_count = len(batch_structure.get("questions_and_distortions", []))
            total_configs = 1
        
        distortions_per_q = batch_structure.get("distortions_per_question", 10)
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:  # Only log first attempt to reduce verbosity
                    logger.info(f"[{worker_id}] 📡 {subject} - Sending MEGA batch: {total_configs} μ configs × {questions_count} questions × {distortions_per_q} distortions each")
                
                response = requests.post(
                    self.distort_endpoint,
                    json={"batch_data": batch_structure, "max_tokens": 4000},  # Increased tokens for larger response
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"[{worker_id}] ✅ {subject} μ={miu:.1f} - Successfully filled complete batch with all distortions")
                    return result
                else:
                    logger.error(f"[{worker_id}] ⚠️ {subject} μ={miu:.1f} - Server returned status {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.error(f"[{worker_id}] ❌ {subject} μ={miu:.1f} - Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"[{worker_id}] ⏳ {subject} μ={miu:.1f} - Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
        
        logger.error(f"[{worker_id}] 💥 {subject} μ={miu:.1f} - FAILED to fill complete batch after {self.max_retries} attempts")
        return None

class DataPreparationProcessor:
    """Main processor for data preparation"""
    
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.question_loader = QuestionLoader("data/questions.json")
        self.mmlu_loader = MMLULoader()
        
        # Create clients for each server with configured timeout
        self.mistral_clients = {}
        for server in self.config.mistral_servers:
            client = MistralDistortionClient(
                server.url,
                self.config.max_retries,
                self.config.retry_delay
            )
            # Set the timeout from config
            client.request_timeout = self.config.request_timeout
            self.mistral_clients[server.name] = client
        
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_file: str) -> ProcessingConfig:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            settings = config_data.get('settings', {})
            
            # Parse Mistral servers
            mistral_servers = []
            for server_config in settings.get('mistral_servers', []):
                mistral_servers.append(MistralServer(
                    url=server_config['url'],
                    name=server_config['name'],
                    subjects=server_config.get('subjects', [])
                ))
            
            # If no servers configured, use default
            if not mistral_servers:
                mistral_servers.append(MistralServer(
                    url='http://localhost:8000',
                    name='default_server',
                    subjects=[]  # Will handle all subjects
                ))
            
            return ProcessingConfig(
                subjects=config_data.get('subjects', []),
                distortions_per_question=settings.get('distortions_per_question', 3),
                output_directory=settings.get('output_directory', 'results/distortions_per_subject'),
                mistral_servers=mistral_servers,
                max_batch_size=settings.get('max_batch_size', 50),
                max_retries=settings.get('max_retries', 3),
                retry_delay=settings.get('retry_delay', 1.0),
                delay_between_batches=settings.get('delay_between_batches', 0.5),
                max_concurrent_workers=settings.get('max_concurrent_workers', 2),
                worker_timeout=settings.get('worker_timeout', 600),
                request_timeout=settings.get('request_timeout', 300)
            )
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def get_server_for_subject(self, subject_name: str) -> Optional[MistralDistortionClient]:
        """Get the appropriate server client for a subject"""
        for server in self.config.mistral_servers:
            if not server.subjects or subject_name in server.subjects:
                return self.mistral_clients[server.name]
        
        # Fallback to first server if no specific assignment
        if self.mistral_clients:
            return list(self.mistral_clients.values())[0]
        
        return None
    
    def calculate_batch_size(self, question_count: int) -> int:
        """Calculate optimal batch size based on server capacity"""
        # Simply use the max_batch_size - don't multiply by distortions
        # because we process distortions one round at a time
        actual_size = min(self.config.max_batch_size, question_count)
        
        logger.info(f"Calculated batch size: min({self.config.max_batch_size}, {question_count}) = {actual_size}")
        return actual_size
    
    def process_subject_mega(self, subject_config: Dict[str, Any]) -> List[str]:
        """Process subject by sending smaller batches per μ value"""
        
        subject_name = subject_config["name"]
        question_count = subject_config.get("question_count", 10)
        mius = subject_config.get("mius", [0.5])
        
        logger.info(f"🚀 Processing {subject_name}: {len(mius)} μ values × {question_count} questions × {self.config.distortions_per_question} distortions")
        
        # Get server
        mistral_client = list(self.mistral_clients.values())[0]  # Use first server
        
        # Load questions
        questions = self.question_loader.get_questions(subject_name, question_count)
        if not questions:
            logger.error(f"No questions found for {subject_name}. Available subjects: {self.question_loader.get_available_subjects()}")
            return []
        
        if len(questions) < question_count:
            logger.warning(f"Only {len(questions)} questions available for {subject_name}, requested {question_count}")
        
        output_files = []
        
        # Process each μ value separately to avoid memory issues
        for miu in mius:
            logger.info(f"📡 {subject_name} μ={miu:.1f} - Processing {question_count} questions")
            
            # Create structure for single μ value
            single_miu_structure = {
                "subject": subject_name,
                "miu": miu,
                "distortions_per_question": self.config.distortions_per_question,
                "questions_and_distortions": []
            }
            
            # Add all questions for this μ
            for question in questions:
                single_miu_structure["questions_and_distortions"].append({
                    "question_id": question["question_id"],
                    "question_text": question["question_text"],
                    "distortions_texts": []  # Mistral fills this
                })
            
            # Send request for this μ value
            result = mistral_client.distort_complete_batch(subject_name, miu, single_miu_structure, "main")
            
            if result and "questions_and_distortions" in result:
                # Create individual result
                individual_result = {
                    "subject": subject_name,
                    "miu": miu,
                    "questions_and_distortions": result["questions_and_distortions"]
                }
                
                # Save file
                output_file = self.save_result(individual_result)
                output_files.append(output_file)
                
                # Log summary
                total_distortions = sum(len(q["distortions_texts"]) for q in individual_result["questions_and_distortions"])
                logger.info(f"✅ {subject_name} μ={miu:.1f}: {question_count} questions, {total_distortions} distortions")
            else:
                logger.error(f"❌ Failed to process {subject_name} μ={miu:.1f}")
        
        logger.info(f"🏁 Completed {subject_name}: {len(output_files)} files generated")
        return output_files
    

    def save_result(self, result: Dict[str, Any]) -> str:
        """Save result to JSON file with timestamp"""
        import datetime
        
        subject = result["subject"]
        miu = result["miu"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{subject}_{miu:.1f}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")
        return str(filepath)
    
    def create_shared_work_queue(self) -> queue.Queue:
        """Create a shared work queue for all workers to use the same server"""
        work_queue = queue.Queue()
        
        # Get the single server (should only be one now)
        if not self.config.mistral_servers:
            raise ValueError("No Mistral servers configured")
        
        server = self.config.mistral_servers[0]  # Use first (and only) server
        
        # Add all subject-miu combinations to the shared queue
        for subject_config in self.config.subjects:
            subject_name = subject_config["name"]
            mius = subject_config.get("mius", [0.5])
            
            for miu in mius:
                work_item = {
                    'subject_config': subject_config,
                    'miu': miu,
                    'subject_name': subject_name,
                    'server_url': server.url,
                    'server_name': server.name
                }
                work_queue.put(work_item)
                logger.info(f"Queued {subject_name} μ={miu:.1f} for {server.name} ({server.url})")
        
        logger.info(f"Created shared work queue with {work_queue.qsize()} total tasks for parallel processing")
        return work_queue
    
    def worker_process_tasks(self, worker_id: str, work_queue: queue.Queue, results_queue: queue.Queue):
        """Worker function to process tasks from the queue"""
        processed_count = 0
        
        while True:
            try:
                # Get work item with timeout
                work_item = work_queue.get(timeout=1.0)
                
                subject_config = work_item['subject_config']
                miu = work_item['miu']
                subject_name = work_item['subject_name']
                
                logger.info(f"[{worker_id}] 🚀 STARTING {subject_name} μ={miu:.1f}")
                
                try:
                    result = self.process_subject_miu_combination(subject_config, miu, worker_id)
                    
                    if result:
                        output_file = self.save_result(result)
                        
                        # Log summary
                        question_count = len(result["questions_and_distortions"])
                        total_distortions = sum(len(q["distortions_texts"]) for q in result["questions_and_distortions"])
                        logger.info(f"[{worker_id}] ✅ COMPLETED {subject_name} μ={miu:.1f}: {question_count} questions, {total_distortions} distortions → {output_file}")
                        
                        results_queue.put({
                            'status': 'success',
                            'output_file': output_file,
                            'subject': subject_name,
                            'miu': miu,
                            'worker_id': worker_id
                        })
                    else:
                        results_queue.put({
                            'status': 'failed',
                            'subject': subject_name,
                            'miu': miu,
                            'error': 'No result generated',
                            'worker_id': worker_id
                        })
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"[{worker_id}] ❌ FAILED {subject_name} μ={miu:.1f}: {e}")
                    results_queue.put({
                        'status': 'failed',
                        'subject': subject_name,
                        'miu': miu,
                        'error': str(e),
                        'worker_id': worker_id
                    })
                finally:
                    work_queue.task_done()
                    
            except queue.Empty:
                # No more work available
                break
        
        logger.info(f"[{worker_id}] 🏁 WORKER FINISHED: Processed {processed_count} tasks")

    def prepare_questions_from_config(self):
        """Load questions from MMLU based on config subjects and save to data folder"""
        
        logger.info("📚 Loading questions from MMLU based on config...")
        
        # Load all questions from MMLU in one go
        questions_dict = self.mmlu_loader.load_questions_from_subjects(self.config.subjects)
        
        if questions_dict:
            # Save to questions.json
            saved_file = self.question_loader.save_questions(questions_dict)
            logger.info(f"💾 Questions saved to {saved_file}")
            
            # Reload the question_loader to use the new data
            self.question_loader.load_questions()
        else:
            logger.error("❌ Failed to load any questions from MMLU")

    def process_all(self) -> List[str]:
        """Process all subjects using MEGA requests - simplified, no workers"""
        
        # First, generate/load questions based on config
        self.prepare_questions_from_config()
        
        # Check server health before starting
        client = list(self.mistral_clients.values())[0]
        server_url = self.config.mistral_servers[0].url
        
        logger.info(f"Checking Mistral server health at {server_url}...")
        if not client.check_server_health():
            raise Exception(f"Mistral server at {server_url} is not healthy. Please start the server first using: python3 modules/mistral_server.py")
        
        logger.info("✅ Mistral server is healthy and ready")
        
        all_output_files = []
        
        # Process each subject with MEGA request
        for subject_config in self.config.subjects:
            try:
                subject_name = subject_config.get('name', 'unknown')
                logger.info(f"Processing {subject_name}...")
                
                output_files = self.process_subject_mega(subject_config)
                all_output_files.extend(output_files)
                
            except Exception as e:
                logger.error(f"Failed to process subject {subject_config.get('name', 'unknown')}: {e}")
        
        logger.info(f"🎉 MEGA PROCESSING COMPLETE: Generated {len(all_output_files)} total files")
        return all_output_files

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Preparation for Question Distortion")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        processor = DataPreparationProcessor(args.config)
        output_files = processor.process_all()
        
        print(f"\n✅ Data preparation completed!")
        print(f"Generated {len(output_files)} files:")
        for file in output_files:
            print(f"  - {file}")
            
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

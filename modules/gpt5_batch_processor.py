#!/usr/bin/env python3
"""
GPT-5 Batch Processing Module for Chameleon Project
Handles batch submission, monitoring, and result parsing for OpenAI GPT-5
"""

import openai
import json
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT5BatchProcessor:
    """Handles GPT-5 batch processing for multiple choice questions"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        self.client = openai.OpenAI(api_key=api_key)
        logger.info("GPT-5 Batch Processor initialized")
    
    def submit_batch(self, batch_file_path: str, description: str = "GPT-5 Batch Processing") -> str:
        """Submit a batch request to OpenAI GPT-5"""
        
        logger.info(f"🚀 Submitting GPT-5 batch: {batch_file_path}")
        
        # Validate file exists
        if not Path(batch_file_path).exists():
            raise FileNotFoundError(f"Batch file not found: {batch_file_path}")
        
        # Upload the batch file
        logger.info("📤 Uploading batch file...")
        try:
            with open(batch_file_path, "rb") as file:
                batch_input_file = self.client.files.create(
                    file=file,
                    purpose="batch"
                )
            
            logger.info(f"✅ File uploaded with ID: {batch_input_file.id}")
            
            # Create batch request
            logger.info("🔄 Creating batch request...")
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": description,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Batch created with ID: {batch.id}")
            logger.info(f"📊 Status: {batch.status}")
            return batch.id
            
        except Exception as e:
            logger.error(f"❌ Error submitting batch: {e}")
            raise
    
    def check_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Check the status of a GPT-5 batch request"""
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            logger.info(f"\n📋 Batch Status Report")
            logger.info(f"🆔 Batch ID: {batch.id}")
            logger.info(f"📊 Status: {batch.status}")
            logger.info(f"🕐 Created: {datetime.fromtimestamp(batch.created_at)}")
            
            if batch.request_counts:
                counts = batch.request_counts
                logger.info(f"📈 Request Counts:")
                logger.info(f"   Total: {counts.total}")
                logger.info(f"   Completed: {counts.completed}")
                logger.info(f"   Failed: {counts.failed}")
                
                if counts.total > 0:
                    progress = (counts.completed / counts.total) * 100
                    logger.info(f"   Progress: {progress:.1f}%")
            
            if batch.errors:
                logger.warning(f"❌ Errors: {batch.errors}")
            
            return {
                'id': batch.id,
                'status': batch.status,
                'created_at': batch.created_at,
                'request_counts': batch.request_counts.__dict__ if batch.request_counts else None,
                'errors': batch.errors,
                'output_file_id': getattr(batch, 'output_file_id', None)
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking batch status: {e}")
            return None
    
    def cancel_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Cancel a specific batch"""
        
        logger.info(f"🛑 Cancelling batch: {batch_id}")
        
        try:
            # Cancel the batch
            batch = self.client.batches.cancel(batch_id)
            
            logger.info(f"✅ Batch cancellation requested")
            logger.info(f"📊 Status: {batch.status}")
            logger.info(f"⏱️  The batch will be in 'cancelling' status for up to 10 minutes")
            
            return {
                'id': batch.id,
                'status': batch.status,
                'created_at': batch.created_at,
                'cancelled_at': getattr(batch, 'cancelled_at', None),
                'request_counts': batch.request_counts.__dict__ if batch.request_counts else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error cancelling batch: {e}")
            return None
    
    def download_results(self, batch_id: str, output_filename: Optional[str] = None) -> Optional[str]:
        """Download and save GPT-5 batch results"""
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            logger.warning(f"⏳ Batch not completed yet. Status: {batch.status}")
            return None
        
        if not batch.output_file_id:
            logger.error("❌ No output file available")
            return None
        
        # Generate filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"results/gpt5_batch_results_{timestamp}.jsonl"
        
        # Ensure results directory exists
        Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("📥 Downloading results...")
        try:
            result_file_response = self.client.files.content(batch.output_file_id)
            
            with open(output_filename, 'wb') as file:
                file.write(result_file_response.content)
            
            logger.info(f"✅ Results saved to: {output_filename}")
            return output_filename
            
        except Exception as e:
            logger.error(f"❌ Error downloading results: {e}")
            return None
    
    def parse_results(self, results_file_path: str, output_csv: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse GPT-5 batch results and extract answers"""
        results = []
        successful = 0
        failed = 0
        
        logger.info(f"📊 Parsing results from: {results_file_path}")
        
        with open(results_file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    result = json.loads(line)
                    custom_id = result['custom_id']
                    
                    if result['response']['status_code'] == 200:
                        # Extract the answer
                        response_body = result['response']['body']
                        message_content = response_body['choices'][0]['message']['content']
                        
                        # Handle both simple text and structured JSON responses
                        answer = None
                        confidence = None
                        
                        try:
                            # Try to parse as JSON (structured output)
                            parsed_content = json.loads(message_content)
                            answer = parsed_content.get('answer', message_content.strip())
                            confidence = parsed_content.get('confidence')
                        except (json.JSONDecodeError, TypeError):
                            # Fallback to plain text
                            answer = message_content.strip()
                        
                        results.append({
                            'custom_id': custom_id,
                            'answer': answer,
                            'confidence': confidence,
                            'success': True,
                            'usage': response_body.get('usage', {}),
                            'line_number': line_num
                        })
                        successful += 1
                        
                    else:
                        # Handle errors
                        error_info = result['response']['body']
                        results.append({
                            'custom_id': custom_id,
                            'error': error_info,
                            'success': False,
                            'line_number': line_num
                        })
                        failed += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️  Error parsing line {line_num}: {e}")
                    failed += 1
        
        # Print summary
        logger.info(f"\n📈 Parsing Summary:")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📊 Total: {successful + failed}")
        
        # Save as CSV if requested
        if output_csv:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['custom_id', 'answer', 'confidence', 'success', 'input_tokens', 'output_tokens']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for r in results:
                    row = {
                        'custom_id': r['custom_id'],
                        'answer': r.get('answer', ''),
                        'confidence': r.get('confidence', ''),
                        'success': r['success'],
                        'input_tokens': r.get('usage', {}).get('prompt_tokens', ''),
                        'output_tokens': r.get('usage', {}).get('completion_tokens', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"💾 Results also saved as CSV: {output_csv}")
        
        return results
    
    def monitor_batch(self, batch_id: str, check_interval: int = 300) -> Optional[Dict[str, Any]]:
        """Monitor a GPT-5 batch until completion"""
        logger.info(f"👀 Monitoring batch {batch_id}")
        logger.info(f"🔄 Checking every {check_interval} seconds...")
        
        while True:
            batch_info = self.check_batch_status(batch_id)
            
            if not batch_info:
                logger.error("❌ Failed to retrieve batch status")
                break
                
            status = batch_info['status']
            
            if status == "completed":
                logger.info("\n🎉 Batch completed successfully!")
                return batch_info
            elif status == "failed":
                logger.error("\n❌ Batch failed!")
                if batch_info.get('errors'):
                    logger.error(f"Errors: {batch_info['errors']}")
                return batch_info
            elif status in ["cancelled", "expired"]:
                logger.warning(f"\n⚠️  Batch {status}!")
                return batch_info
            else:
                logger.info(f"⏳ Still processing... (Status: {status})")
                time.sleep(check_interval)
    
    def process_complete_workflow(self, batch_file_path: str, description: str, check_interval: int = 600) -> Dict[str, Any]:
        """Complete workflow: submit, monitor, download, and parse"""
        logger.info("🚀 Starting complete GPT-5 batch workflow")
        
        workflow_results = {
            'batch_id': None,
            'batch_file': batch_file_path,
            'description': description,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'started',
            'results_file': None,
            'parsed_results': None,
            'error': None
        }
        
        try:
            # Step 1: Submit batch
            batch_id = self.submit_batch(batch_file_path, description)
            workflow_results['batch_id'] = batch_id
            
            # Step 2: Monitor progress
            completed_batch = self.monitor_batch(batch_id, check_interval)
            
            # Step 3: Download and parse results
            if completed_batch and completed_batch['status'] == "completed":
                results_file = self.download_results(batch_id)
                workflow_results['results_file'] = results_file
                
                if results_file:
                    # Parse results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_file = f"results/gpt5_answers_{timestamp}.csv"
                    
                    parsed_results = self.parse_results(results_file, output_csv=csv_file)
                    workflow_results['parsed_results'] = len(parsed_results)
                    workflow_results['csv_file'] = csv_file
                    
                    # Save detailed results
                    json_file = f"results/gpt5_detailed_results_{timestamp}.json"
                    with open(json_file, "w") as f:
                        json.dump(parsed_results, f, indent=2)
                    
                    workflow_results['json_file'] = json_file
                    workflow_results['status'] = 'completed'
                    
                    logger.info(f"\n✅ Workflow completed successfully!")
                    logger.info(f"📄 Raw results: {results_file}")
                    logger.info(f"📊 CSV format: {csv_file}")
                    logger.info(f"📝 Detailed JSON: {json_file}")
            else:
                workflow_results['status'] = 'failed'
                workflow_results['error'] = f"Batch status: {completed_batch['status'] if completed_batch else 'unknown'}"
                
        except Exception as e:
            workflow_results['status'] = 'error'
            workflow_results['error'] = str(e)
            logger.error(f"❌ Workflow error: {e}")
        
        workflow_results['end_time'] = datetime.now().isoformat()
        return workflow_results


def create_test_batch_file(output_path: str = "test_batch.jsonl", num_questions: int = 5) -> str:
    """Create a small test batch file for testing the infrastructure"""
    
    # Sample questions for testing
    test_questions = [
        {
            "question": "What is the capital of France?",
            "choices": {"A": "London", "B": "Berlin", "C": "Paris", "D": "Madrid"},
            "correct": "C"
        },
        {
            "question": "Which planet is closest to the Sun?",
            "choices": {"A": "Venus", "B": "Mercury", "C": "Earth", "D": "Mars"},
            "correct": "B"
        },
        {
            "question": "What is 2 + 2?",
            "choices": {"A": "3", "B": "4", "C": "5", "D": "6"},
            "correct": "B"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "choices": {"A": "Charles Dickens", "B": "William Shakespeare", "C": "Jane Austen", "D": "Mark Twain"},
            "correct": "B"
        },
        {
            "question": "What is the chemical symbol for water?",
            "choices": {"A": "H2O", "B": "CO2", "C": "NaCl", "D": "O2"},
            "correct": "A"
        }
    ]
    
    logger.info(f"🔧 Creating test batch file with {num_questions} questions")
    
    with open(output_path, 'w') as f:
        for i in range(min(num_questions, len(test_questions))):
            question_data = test_questions[i]
            
            # Format choices for the prompt
            choices_text = "\n".join([f"{k}: {v}" for k, v in question_data["choices"].items()])
            
            # Create the request
            request = {
                "custom_id": f"test_q_{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",  # Using gpt-4o-mini for testing
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant answering multiple choice questions. Respond with only the letter of the correct answer (A, B, C, or D)."
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question_data['question']}\n\nChoices:\n{choices_text}\n\nAnswer:"
                        }
                    ],
                    "max_tokens": 10,
                    "temperature": 0.0
                }
            }
            
            f.write(json.dumps(request) + '\n')
    
    logger.info(f"✅ Test batch file created: {output_path}")
    return output_path


if __name__ == "__main__":
    print("🧪 GPT-5 Batch Processor - Test Mode")
    print("\n💡 Usage Examples:")
    print("1. Create test batch file:")
    print("   test_file = create_test_batch_file('test_batch.jsonl', 5)")
    print("\n2. Initialize processor:")
    print("   processor = GPT5BatchProcessor('your-api-key-here')")
    print("\n3. Run complete workflow:")
    print("   results = processor.process_complete_workflow('test_batch.jsonl', 'Test Batch')")
    print("\n💰 Cost Information:")
    print("- GPT-4o-mini: $0.15/M input tokens, $0.60/M output tokens")
    print("- Batch processing: 50% discount")
    print("- Test batch (5 questions): ~$0.01")
    print("- Full dataset (18.4K questions): ~$5-10")

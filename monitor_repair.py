#!/usr/bin/env python3
"""
Simple repair batch monitoring script
Run periodically to check repair batch status
"""

import json
import time
from pathlib import Path
from gpt5_manager import GPT5BatchManager

def main():
    """Monitor repair batch and auto-update when complete"""
    
    # Check for repair batch tracking
    tracking_file = Path("batches/tracking/repair_batch_info.json")
    
    if not tracking_file.exists():
        print("❌ No repair batch found to monitor!")
        return 1
    
    try:
        with open(tracking_file, 'r') as f:
            repair_info = json.load(f)
        
        batch_id = repair_info['batch_id']
        print(f"🔍 Monitoring repair batch: {batch_id}")
        
        manager = GPT5BatchManager()
        
        # Monitor and auto-update if complete
        completed = manager.monitor_repair_batch(batch_id)
        
        if completed:
            print("🎉 Repair batch completed and CSV updated!")
            return 0
        else:
            print("⏳ Repair batch still processing...")
            return 2  # Still processing
            
    except Exception as e:
        print(f"❌ Error monitoring repair batch: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

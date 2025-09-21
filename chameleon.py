#!/usr/bin/env python3
"""
Chameleon Project - Main Entry Point
Simple wrapper for all project functionality
"""

import sys
from pathlib import Path

def show_help():
    """Show help information"""
    print("🦎 Chameleon Project - GPT-5 Evaluation System")
    print("=" * 50)
    print()
    print("📋 Available Commands:")
    print()
    print("🔧 Data Preparation:")
    print("   python3 chameleon.py data-prep     # Prepare MMLU questions and distortions")
    print()
    print("🚀 GPT-5 Batch Management:")
    print("   python3 chameleon.py create        # Create GPT-5 batch files")
    print("   python3 chameleon.py submit        # Submit batches (interactive)")
    print("   python3 chameleon.py submit --auto # Submit all batches (auto)")
    print("   python3 chameleon.py monitor       # Monitor batch progress")
    print("   python3 chameleon.py cancel        # Cancel batches (interactive)")
    print("   python3 chameleon.py cleanup       # Clean up tracking data")
    print("   python3 chameleon.py repair        # Create repair batch (auto-detect failures)")
    print("   python3 chameleon.py repair --ids q_123 q_456  # Create repair batch (specific IDs)")
    print("   python3 chameleon.py process-failures  # Complete failure workflow (collect→repair→submit)")
    print("   python3 chameleon.py monitor-repair BATCH_ID  # Monitor specific repair batch")
    print()
    print("🔍 Analysis:")
    print("   python3 chameleon.py analyze       # Analyze results and generate reports")
    print()
    print("🛠️ Utilities:")
    print("   python3 chameleon.py status        # Show project status")
    print("   python3 chameleon.py help          # Show this help")
    print()
    print("📁 Project Structure:")
    print("   config/           # Configuration files")
    print("   data/             # Source MMLU data")
    print("   distortions/      # Generated distortions and results")
    print("   batches/          # GPT-5 batch files and results")
    print("     ├── jsonl/      # Batch JSONL files")
    print("     ├── results/    # Downloaded results")
    print("     └── tracking/   # Batch tracking data")
    print("   modules/          # Core processing modules")
    print()

def show_status():
    """Show project status"""
    print("📊 Chameleon Project Status")
    print("=" * 30)
    
    # Check key files
    key_files = {
        "Config": "config/config.yaml",
        "Questions": "data/questions.json", 
        "Distortions": "distortions/comprehensive_distortion_dataset.json",
        "CSV Dataset": "distortions/comprehensive_distortion_dataset.csv"
    }
    
    print("\n📁 Key Files:")
    for name, path in key_files.items():
        exists = "✅" if Path(path).exists() else "❌"
        size = ""
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1024 / 1024
            size = f" ({size_mb:.1f} MB)"
        print(f"   {exists} {name}: {path}{size}")
    
    # Check batch files
    batch_dir = Path("batches/jsonl")
    if batch_dir.exists():
        batch_files = list(batch_dir.glob("gpt5_batch_part_*.jsonl"))
        print(f"\n🚀 Batch Files: {len(batch_files)} found")
        for batch_file in sorted(batch_files):
            size_mb = batch_file.stat().st_size / 1024 / 1024
            print(f"   📄 {batch_file.name} ({size_mb:.1f} MB)")
    
    # Check tracking
    tracking_file = Path("batches/tracking/batch_info.json")
    if tracking_file.exists():
        import json
        try:
            with open(tracking_file, 'r') as f:
                tracking_data = json.load(f)
            batches = tracking_data.get('batches', [])
            print(f"\n📊 Active Batches: {len(batches)}")
            for batch in batches:
                print(f"   🔄 Part {batch['part']}: {batch['batch_id'][:20]}...")
        except:
            print("\n⚠️  Tracking file exists but cannot be read")
    else:
        print("\n📊 Active Batches: None")
    
    print()

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return 0
    
    command = sys.argv[1].lower()
    
    if command in ['help', '-h', '--help']:
        show_help()
        return 0
    
    elif command == 'status':
        show_status()
        return 0
    
    elif command == 'data-prep':
        print("🔧 Starting data preparation...")
        try:
            sys.path.append(str(Path(__file__).parent / "modules"))
            from data_preparation import main as data_prep_main
            return data_prep_main()
        except ImportError:
            print("❌ Data preparation module not found!")
            return 1
    
    elif command in ['create', 'submit', 'monitor', 'cancel', 'cleanup', 'repair', 'process-failures', 'monitor-repair']:
        print(f"🚀 Running GPT-5 batch {command}...")
        
        # Prepare arguments for gpt5_manager
        manager_args = ['gpt5_manager.py', command] + sys.argv[2:]
        
        # Import and run the manager
        try:
            from gpt5_manager import main as manager_main
            # Temporarily replace sys.argv
            original_argv = sys.argv
            sys.argv = manager_args
            result = manager_main()
            sys.argv = original_argv
            return result
        except ImportError:
            print("❌ GPT-5 manager not found!")
            return 1
    
    elif command == 'analyze':
        print("🔍 Analyzing results...")
        # TODO: Implement analysis functionality
        print("📊 Analysis functionality coming soon!")
        return 0
    
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Run 'python3 chameleon.py help' for available commands")
        return 1

if __name__ == "__main__":
    sys.exit(main())

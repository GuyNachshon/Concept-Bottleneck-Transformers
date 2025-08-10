#!/usr/bin/env python3
"""
Manage and explore organized CBT analysis results.
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path

def show_analysis_summary():
    """Show a summary of all analyses."""
    print("📊 CBT Analysis Summary")
    print("=" * 50)
    
    # Check if organized structure exists
    if not os.path.exists("results/analysis_index.json"):
        print("❌ No organized results found. Run 'python organize_analysis.py' first.")
        return
    
    # Load index
    with open("results/analysis_index.json", 'r') as f:
        index = json.load(f)
    
    print(f"📋 Index created: {index['created']}")
    print(f"🔍 Total analyses: {len(index['analyses'])}")
    print(f"🤖 Total models: {len(index['models'])}")
    
    # Show recent analyses
    print(f"\n📈 Recent Analyses:")
    analyses = sorted(index['analyses'], key=lambda x: x['timestamp'], reverse=True)
    
    for i, analysis in enumerate(analyses[:5]):
        config = analysis['config']
        print(f"  {i+1}. {analysis['name']}")
        print(f"     Model: {analysis['model']}")
        print(f"     Config: m={config.get('m', 'N/A')}, k={config.get('k', 'N/A')}, α={config.get('alpha', 'N/A')}")
        print(f"     Date: {analysis['timestamp']}")
        print()

def list_models_by_type():
    """List models organized by type."""
    print("🤖 Available Models by Type")
    print("=" * 50)
    
    if not os.path.exists("results/analysis_index.json"):
        print("❌ No organized results found.")
        return
    
    with open("results/analysis_index.json", 'r') as f:
        index = json.load(f)
    
    # Group models by type
    model_types = {}
    for model in index['models']:
        model_type = model['type']
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(model)
    
    for model_type, models in model_types.items():
        print(f"\n📁 {model_type.title()} ({len(models)} files):")
        for model in models:
            size_mb = model['size'] / (1024 * 1024)
            modified = datetime.fromisoformat(model['modified']).strftime('%Y-%m-%d %H:%M')
            print(f"  • {model['name']} ({size_mb:.1f} MB, {modified})")

def find_analysis_by_model(model_name):
    """Find analyses for a specific model."""
    print(f"🔍 Finding analyses for model: {model_name}")
    print("=" * 50)
    
    if not os.path.exists("results/analysis_index.json"):
        print("❌ No organized results found.")
        return
    
    with open("results/analysis_index.json", 'r') as f:
        index = json.load(f)
    
    matching_analyses = []
    for analysis in index['analyses']:
        if model_name in analysis['model']:
            matching_analyses.append(analysis)
    
    if not matching_analyses:
        print(f"❌ No analyses found for model: {model_name}")
        return
    
    print(f"✅ Found {len(matching_analyses)} analyses:")
    for i, analysis in enumerate(matching_analyses):
        print(f"  {i+1}. {analysis['name']}")
        print(f"     Path: {analysis['path']}")
        print(f"     Date: {analysis['timestamp']}")
        print()

def show_analysis_details(analysis_name):
    """Show detailed information about a specific analysis."""
    print(f"📋 Analysis Details: {analysis_name}")
    print("=" * 50)
    
    analysis_path = f"results/analysis/concept_analysis/{analysis_name}"
    if not os.path.exists(analysis_path):
        print(f"❌ Analysis not found: {analysis_path}")
        return
    
    # Load metadata
    metadata_file = os.path.join(analysis_path, "analysis_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print("📊 Metadata:")
        print(f"  Model: {metadata.get('model_file', 'N/A')}")
        print(f"  Device: {metadata.get('device', 'N/A')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
        
        config = metadata.get('model_config', {})
        print(f"  Configuration:")
        print(f"    m: {config.get('m', 'N/A')}")
        print(f"    k: {config.get('k', 'N/A')}")
        print(f"    alpha: {config.get('alpha', 'N/A')}")
        print(f"    concept_blocks: {config.get('concept_blocks', 'N/A')}")
    
    # List files in analysis
    print(f"\n📁 Files in analysis:")
    for root, dirs, files in os.walk(analysis_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, analysis_path)
            size = os.path.getsize(file_path)
            print(f"  • {rel_path} ({size} bytes)")

def cleanup_old_analyses(days_old=30):
    """Clean up analyses older than specified days."""
    print(f"🧹 Cleaning up analyses older than {days_old} days")
    print("=" * 50)
    
    if not os.path.exists("results/analysis_index.json"):
        print("❌ No organized results found.")
        return
    
    with open("results/analysis_index.json", 'r') as f:
        index = json.load(f)
    
    cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    old_analyses = []
    
    for analysis in index['analyses']:
        try:
            analysis_date = datetime.fromisoformat(analysis['timestamp'].replace('Z', '+00:00')).timestamp()
            if analysis_date < cutoff_date:
                old_analyses.append(analysis)
        except:
            continue
    
    if not old_analyses:
        print("✅ No old analyses to clean up.")
        return
    
    print(f"Found {len(old_analyses)} old analyses:")
    for analysis in old_analyses:
        print(f"  • {analysis['name']} ({analysis['timestamp']})")
    
    response = input(f"\nDelete these {len(old_analyses)} analyses? (y/N): ")
    if response.lower() == 'y':
        for analysis in old_analyses:
            analysis_path = analysis['path']
            if os.path.exists(analysis_path):
                import shutil
                shutil.rmtree(analysis_path)
                print(f"  ✅ Deleted: {analysis['name']}")
        print("✅ Cleanup complete!")
    else:
        print("❌ Cleanup cancelled.")

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("🗂️  CBT Results Manager")
        print("=" * 50)
        print("Usage:")
        print("  python manage_results.py summary              # Show analysis summary")
        print("  python manage_results.py models               # List models by type")
        print("  python manage_results.py find <model_name>    # Find analyses for model")
        print("  python manage_results.py details <analysis>   # Show analysis details")
        print("  python manage_results.py cleanup [days]       # Clean up old analyses")
        return
    
    command = sys.argv[1]
    
    if command == "summary":
        show_analysis_summary()
    elif command == "models":
        list_models_by_type()
    elif command == "find" and len(sys.argv) > 2:
        find_analysis_by_model(sys.argv[2])
    elif command == "details" and len(sys.argv) > 2:
        show_analysis_details(sys.argv[2])
    elif command == "cleanup":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        cleanup_old_analyses(days)
    else:
        print("❌ Invalid command. Use 'python manage_results.py' for help.")

if __name__ == "__main__":
    main() 
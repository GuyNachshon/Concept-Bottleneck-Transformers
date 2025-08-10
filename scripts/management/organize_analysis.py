#!/usr/bin/env python3
"""
Organize and manage CBT analysis results.
"""

import os
import shutil
import json
import glob
from datetime import datetime
from pathlib import Path

def create_analysis_structure():
    """Create organized directory structure for analysis results."""
    structure = {
        "analysis": {
            "concept_analysis": {},
            "model_comparisons": {},
            "ablation_studies": {},
            "visualizations": {},
            "reports": {}
        },
        "models": {
            "checkpoints": {},
            "weights": {},
            "configs": {}
        },
        "experiments": {
            "results": {},
            "logs": {},
            "configs": {}
        }
    }
    
    # Create directories
    for category, subcats in structure.items():
        for subcat in subcats:
            os.makedirs(f"results/{category}/{subcat}", exist_ok=True)
    
    print("✅ Created organized directory structure")

def organize_existing_results():
    """Organize existing analysis results into proper structure."""
    print("🔧 Organizing existing results...")
    
    # Find all concept analysis directories
    analysis_dirs = glob.glob("concept_analysis_*")
    
    for analysis_dir in analysis_dirs:
        if not os.path.isdir(analysis_dir):
            continue
            
        print(f"Organizing: {analysis_dir}")
        
        # Read metadata to understand what this analysis contains
        metadata_file = os.path.join(analysis_dir, "analysis_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            model_file = metadata.get('model_file', 'unknown')
            timestamp = metadata.get('timestamp', 'unknown')
            
            # Create organized path
            organized_path = f"results/analysis/concept_analysis/{model_file}_{timestamp}"
            
            # Move directory
            if os.path.exists(organized_path):
                print(f"  ⚠️  {organized_path} already exists, skipping")
            else:
                shutil.move(analysis_dir, organized_path)
                print(f"  ✅ Moved to {organized_path}")
        else:
            print(f"  ⚠️  No metadata found in {analysis_dir}")

def organize_model_files():
    """Organize model files into proper structure."""
    print("🔧 Organizing model files...")
    
    # Find all experiment result directories
    experiment_dirs = glob.glob("trained_cbt_experiment_results_*") + glob.glob("cbt_experiment_results_*")
    
    for exp_dir in experiment_dirs:
        if not os.path.isdir(exp_dir):
            continue
            
        print(f"Organizing models from: {exp_dir}")
        
        # Find model files
        model_files = glob.glob(os.path.join(exp_dir, "cbt_model_*.pt"))
        weight_files = glob.glob(os.path.join(exp_dir, "concept_weights_*.npz"))
        config_files = glob.glob(os.path.join(exp_dir, "*.json"))
        
        # Organize model files
        for model_file in model_files:
            filename = os.path.basename(model_file)
            organized_path = f"results/models/checkpoints/{filename}"
            
            if not os.path.exists(organized_path):
                shutil.copy2(model_file, organized_path)
                print(f"  ✅ Copied model: {filename}")
        
        # Organize weight files
        for weight_file in weight_files:
            filename = os.path.basename(weight_file)
            organized_path = f"results/models/weights/{filename}"
            
            if not os.path.exists(organized_path):
                shutil.copy2(weight_file, organized_path)
                print(f"  ✅ Copied weights: {filename}")
        
        # Organize config files
        for config_file in config_files:
            filename = os.path.basename(config_file)
            organized_path = f"results/models/configs/{filename}"
            
            if not os.path.exists(organized_path):
                shutil.copy2(config_file, organized_path)
                print(f"  ✅ Copied config: {filename}")

def create_analysis_index():
    """Create an index of all analysis results."""
    print("📋 Creating analysis index...")
    
    index = {
        "created": datetime.now().isoformat(),
        "analyses": [],
        "models": [],
        "experiments": []
    }
    
    # Index concept analyses
    concept_analysis_dir = "results/analysis/concept_analysis"
    if os.path.exists(concept_analysis_dir):
        for analysis_dir in os.listdir(concept_analysis_dir):
            analysis_path = os.path.join(concept_analysis_dir, analysis_dir)
            if os.path.isdir(analysis_path):
                metadata_file = os.path.join(analysis_path, "analysis_metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    index["analyses"].append({
                        "name": analysis_dir,
                        "model": metadata.get("model_file", "unknown"),
                        "timestamp": metadata.get("timestamp", "unknown"),
                        "path": analysis_path,
                        "config": metadata.get("model_config", {})
                    })
    
    # Index model files
    model_dirs = ["results/models/checkpoints", "results/models/weights", "results/models/configs"]
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                file_path = os.path.join(model_dir, filename)
                if os.path.isfile(file_path):
                    index["models"].append({
                        "name": filename,
                        "type": os.path.basename(model_dir),
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    })
    
    # Save index
    with open("results/analysis_index.json", 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✅ Created index with {len(index['analyses'])} analyses and {len(index['models'])} models")

def create_analysis_report():
    """Create a summary report of all analyses."""
    print("📊 Creating analysis report...")
    
    # Load index
    if not os.path.exists("results/analysis_index.json"):
        print("❌ No analysis index found. Run organize first.")
        return
    
    with open("results/analysis_index.json", 'r') as f:
        index = json.load(f)
    
    # Generate report
    report = f"""# CBT Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Analyses: {len(index['analyses'])}
- Total Models: {len(index['models'])}
- Total Experiments: {len(index['experiments'])}

## Recent Analyses
"""
    
    # Sort analyses by timestamp
    analyses = sorted(index['analyses'], key=lambda x: x['timestamp'], reverse=True)
    
    for i, analysis in enumerate(analyses[:10]):  # Show last 10
        report += f"""
### {i+1}. {analysis['name']}
- **Model**: {analysis['model']}
- **Date**: {analysis['timestamp']}
- **Config**: m={analysis['config'].get('m', 'N/A')}, k={analysis['config'].get('k', 'N/A')}, α={analysis['config'].get('alpha', 'N/A')}
- **Path**: {analysis['path']}
"""
    
    report += f"""
## Available Models
"""
    
    # Group models by type
    model_types = {}
    for model in index['models']:
        model_type = model['type']
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(model)
    
    for model_type, models in model_types.items():
        report += f"\n### {model_type.title()}\n"
        for model in models:
            size_mb = model['size'] / (1024 * 1024)
            report += f"- {model['name']} ({size_mb:.1f} MB)\n"
    
    # Save report
    with open("results/analysis/reports/analysis_report.md", 'w') as f:
        f.write(report)
    
    print("✅ Created analysis report")

def cleanup_old_files():
    """Clean up old unorganized files."""
    print("🧹 Cleaning up old files...")
    
    # Files to clean up
    cleanup_patterns = [
        "concept_analysis_*",
        "alternative_explanations_*",
        "*.log"
    ]
    
    for pattern in cleanup_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            if os.path.isdir(file_path):
                print(f"  🗑️  Removing old directory: {file_path}")
                shutil.rmtree(file_path)
            else:
                print(f"  🗑️  Removing old file: {file_path}")
                os.remove(file_path)

def main():
    """Main organization function."""
    print("🗂️  CBT Analysis Organization")
    print("=" * 50)
    
    # Create organized structure
    create_analysis_structure()
    
    # Organize existing results
    organize_existing_results()
    
    # Organize model files
    organize_model_files()
    
    # Create index
    create_analysis_index()
    
    # Create report
    create_analysis_report()
    
    # Clean up old files
    cleanup_old_files()
    
    print("\n✅ Organization complete!")
    print("\n📁 New structure:")
    print("  results/")
    print("  ├── analysis/")
    print("  │   ├── concept_analysis/     # Concept analysis results")
    print("  │   ├── model_comparisons/    # Model comparison studies")
    print("  │   ├── ablation_studies/     # Ablation study results")
    print("  │   ├── visualizations/       # Charts and plots")
    print("  │   └── reports/              # Analysis reports")
    print("  ├── models/")
    print("  │   ├── checkpoints/          # Model checkpoints")
    print("  │   ├── weights/              # Concept weights")
    print("  │   └── configs/              # Model configurations")
    print("  └── experiments/")
    print("      ├── results/              # Experiment results")
    print("      ├── logs/                 # Training logs")
    print("      └── configs/              # Experiment configs")
    
    print("\n💡 Next steps:")
    print("  python list_models.py                    # See organized models")
    print("  python analyze_concepts.py               # Run new analysis")
    print("  cat results/analysis/reports/analysis_report.md  # View report")

if __name__ == "__main__":
    main() 
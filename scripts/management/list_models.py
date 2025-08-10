#!/usr/bin/env python3
"""
List all available CBT models for analysis.
"""

import sys
import os
import glob
from datetime import datetime

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def find_all_results_dirs():
    """Find all experiment results directories."""
    patterns = [
        "trained_cbt_experiment_results_*",
        "cbt_experiment_results_*",
        "experiment_results_*"
    ]
    
    all_dirs = []
    for pattern in patterns:
        all_dirs.extend(glob.glob(pattern))
    
    # Sort by creation time (newest first)
    all_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return all_dirs

def extract_model_info(model_file):
    """Extract model configuration from filename."""
    info = {}
    
    # Extract alpha
    if "a30" in model_file:
        info['alpha'] = 0.3
    elif "a20" in model_file:
        info['alpha'] = 0.2
    elif "a10" in model_file:
        info['alpha'] = 0.1
    else:
        info['alpha'] = 0.2  # default
    
    # Extract model type
    if "kl" in model_file:
        info['type'] = "KL (with distillation)"
    elif "stab" in model_file:
        info['type'] = "Stabilized"
    else:
        info['type'] = "Standard"
    
    # Extract cross-seed info
    if "cross_seed" in model_file:
        info['cross_seed'] = True
    else:
        info['cross_seed'] = False
    
    return info

def main():
    """List all available models."""
    print("🔍 Available CBT Models")
    print("=" * 60)
    
    # Find all results directories
    results_dirs = find_all_results_dirs()
    
    if not results_dirs:
        print("❌ No experiment results directories found!")
        return
    
    total_models = 0
    
    for i, results_dir in enumerate(results_dirs):
        print(f"\n📁 {results_dir}")
        print("-" * 40)
        
        # Get creation time
        ctime = os.path.getctime(results_dir)
        ctime_str = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Created: {ctime_str}")
        
        # Find model files
        model_files = [f for f in os.listdir(results_dir) if f.endswith('.pt') and f.startswith('cbt_model_')]
        
        if not model_files:
            print("  No model files found")
            continue
        
        # Sort models by preference
        preferred_order = [
            "cbt_model_stab_kl_m32_k4_a30.pt",
            "cbt_model_stab_kl_m32_k4.pt", 
            "cbt_model_stab_m32_k4.pt"
        ]
        
        # Sort models
        sorted_models = []
        for preferred in preferred_order:
            for model in model_files:
                if model == preferred:
                    sorted_models.append(model)
                    break
        
        # Add any remaining models
        for model in model_files:
            if model not in sorted_models:
                sorted_models.append(model)
        
        # Display models
        for j, model_file in enumerate(sorted_models):
            info = extract_model_info(model_file)
            
            # Mark best model
            marker = "⭐" if j == 0 else "  "
            
            print(f"{marker} {model_file}")
            print(f"    Type: {info['type']}")
            print(f"    Alpha: {info['alpha']}")
            if info['cross_seed']:
                print(f"    Cross-seed: Yes")
            print()
            
            total_models += 1
    
    print("=" * 60)
    print(f"📊 Total models found: {total_models}")
    print("\n💡 Usage:")
    print("  python analyze_concepts.py                    # Analyze best model")
    print("  python analyze_concepts.py <model_file>       # Analyze specific model")
    print("  python cbt_cli.py concept-analysis -m <model> # Analyze via CLI")

if __name__ == "__main__":
    main() 
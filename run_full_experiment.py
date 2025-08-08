#!/usr/bin/env python3
"""
Full CBT Experiment Runner
Runs complete CBT experiments and saves all results for analysis.
"""

import os
import sys
import json
import torch
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cbt.evaluation import get_wikitext_eval_texts
from cbt.experiments import (
    run_granularity_sweep,
    run_placement_study,
    run_cross_seed_stability_test
)


def main():
    """Run the complete CBT experiment."""
    print("=== FULL CBT EXPERIMENT RUNNER ===")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"cbt_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}/")
    
    try:
        # Get evaluation texts from WikiText
        print("Getting WikiText evaluation texts...")
        eval_texts = get_wikitext_eval_texts(num_samples=20)
        print(f"Using {len(eval_texts)} evaluation texts")
        
        # Experiment 1: Granularity Sweep
        print("\n" + "="*60)
        print("EXPERIMENT 1: GRANULARITY SWEEP")
        print("="*60)
        granularity_results = run_granularity_sweep(
            eval_texts=eval_texts,
            save_path=f"{results_dir}/granularity_sweep.json"
        )
        
        # Experiment 2: Placement Study
        print("\n" + "="*60)
        print("EXPERIMENT 2: PLACEMENT STUDY")
        print("="*60)
        placement_results = run_placement_study(
            eval_texts=eval_texts,
            save_path=f"{results_dir}/placement_study.json"
        )
        
        # Experiment 3: Cross-Seed Stability
        print("\n" + "="*60)
        print("EXPERIMENT 3: CROSS-SEED STABILITY")
        print("="*60)
        stability_results = run_cross_seed_stability_test(
            eval_texts=eval_texts,
            save_path=f"{results_dir}/cross_seed_stability.json"
        )
        
        # Generate summary report
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        summary = {
            "timestamp": timestamp,
            "device": str(device),
            "num_eval_texts": len(eval_texts),
            "experiments": {
                "granularity_sweep": "completed",
                "placement_study": "completed", 
                "cross_seed_stability": "completed"
            }
        }
        
        # Analyze results
        best_granularity = None
        best_score = float('inf')
        
        for config_name, config_results in granularity_results.items():
            quality_hit = config_results["quality"]["quality_hit_percent"]
            median_active = config_results["sparsity"]["overall_median_active_concepts"]
            score = quality_hit + max(0, median_active - 8) * 10
            
            if score < best_score:
                best_score = score
                best_granularity = config_name
        
        summary["best_granularity"] = best_granularity
        summary["best_score"] = best_score
        
        # Save summary
        with open(f"{results_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"📁 All results saved to: {results_dir}/")
        print(f"📊 Best configuration: {best_granularity}")
        print(f"📊 Best score: {best_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Experiment completed successfully!")
    else:
        print("\n❌ Experiment failed!")
        sys.exit(1) 
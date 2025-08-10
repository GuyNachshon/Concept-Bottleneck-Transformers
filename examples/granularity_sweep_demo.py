#!/usr/bin/env python3
"""
Granularity Sweep Demo
Tests different m and k values to find optimal sparsity/quality frontier.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.experiments import run_granularity_sweep


def main():
    print("=== Granularity Sweep Demo ===\n")
    
    # Get evaluation texts from WikiText dataset
    from cbt.evaluator import get_wikitext_eval_texts
    eval_texts = get_wikitext_eval_texts(num_samples=15)
    print(f"Using {len(eval_texts)} evaluation texts from WikiText")
    
    # Run granularity sweep
    results = run_granularity_sweep(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m_values=[32, 64, 128],
        k_values=[4, 8, 12],
        eval_texts=eval_texts,
        save_path="granularity_sweep_results.json"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("GRANULARITY SWEEP SUMMARY")
    print("="*60)
    
    best_config = None
    best_score = float('inf')
    
    for config_name, config_results in results.items():
        quality_hit = config_results["quality"]["quality_hit_percent"]
        median_active = config_results["sparsity"]["overall_median_active_concepts"]
        
        # Simple scoring: quality hit + penalty for too many active concepts
        score = quality_hit + max(0, median_active - 8) * 10
        
        print(f"{config_name:12} | Quality: {quality_hit:6.2f}% | Active: {median_active:5.1f} | Score: {score:6.2f}")
        
        if score < best_score:
            best_score = score
            best_config = config_name
    
    print("\n" + "="*60)
    print(f"BEST CONFIGURATION: {best_config}")
    print(f"BEST SCORE: {best_score:.2f}")
    print("="*60)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Placement Study Demo
Tests concepts in early vs. mid vs. late blocks.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.experiments import run_placement_study


def main():
    print("=== Placement Study Demo ===\n")
    
    # Get evaluation texts from WikiText dataset
    from cbt.evaluation import get_wikitext_eval_texts
    eval_texts = get_wikitext_eval_texts(num_samples=15)
    print(f"Using {len(eval_texts)} evaluation texts from WikiText")
    
    # Run placement study
    results = run_placement_study(
        base_model_name="gpt2",
        m=64,
        k=8,
        eval_texts=eval_texts,
        save_path="placement_study_results.json"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PLACEMENT STUDY SUMMARY")
    print("="*60)
    
    best_placement = None
    best_score = float('inf')
    
    for placement_name, placement_results in results.items():
        quality_hit = placement_results["quality"]["quality_hit_percent"]
        median_active = placement_results["sparsity"]["overall_median_active_concepts"]
        blocks = placement_results["blocks"]
        
        # Simple scoring: quality hit + penalty for too many active concepts
        score = quality_hit + max(0, median_active - 8) * 10
        
        print(f"{placement_name:8} | Quality: {quality_hit:6.2f}% | Active: {median_active:5.1f} | Blocks: {blocks} | Score: {score:6.2f}")
        
        if score < best_score:
            best_score = score
            best_placement = placement_name
    
    print("\n" + "="*60)
    print(f"BEST PLACEMENT: {best_placement}")
    print(f"BEST SCORE: {best_score:.2f}")
    print("="*60)
    
    # Analysis
    print("\nANALYSIS:")
    early_quality = results["early"]["quality"]["quality_hit_percent"]
    mid_quality = results["mid"]["quality"]["quality_hit_percent"]
    late_quality = results["late"]["quality"]["quality_hit_percent"]
    
    print(f"Early blocks (0-3): {early_quality:.2f}% quality hit")
    print(f"Mid blocks (4-7):   {mid_quality:.2f}% quality hit")
    print(f"Late blocks (8-11): {late_quality:.2f}% quality hit")
    
    if mid_quality < early_quality and mid_quality < late_quality:
        print("✅ Mid-block placement shows best performance (as expected)")
    else:
        print("⚠️  Unexpected placement results - may need investigation")


if __name__ == "__main__":
    main() 
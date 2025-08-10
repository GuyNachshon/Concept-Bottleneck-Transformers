#!/usr/bin/env python3
"""
Comprehensive Experiments Demo
Runs all CBT experiments and provides complete analysis.
"""

import sys
import os
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.experiments import (
    run_granularity_sweep,
    run_placement_study,
    run_cross_seed_stability_test
)
from cbt.evaluator import CBTEvaluator
from cbt.model import CBTModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def main():
    print("=== Comprehensive CBT Experiments Demo ===\n")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}/")
    
    # Get evaluation texts from WikiText dataset
    from cbt.evaluator import get_wikitext_eval_texts
    eval_texts = get_wikitext_eval_texts(num_samples=20)
    print(f"Using {len(eval_texts)} evaluation texts from WikiText")
    
    # Experiment 1: Granularity Sweep
    print("\n" + "="*60)
    print("EXPERIMENT 1: GRANULARITY SWEEP")
    print("="*60)
    
    granularity_results = run_granularity_sweep(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m_values=[32, 64, 128],
        k_values=[4, 8, 12],
        eval_texts=eval_texts,
        save_path=f"{results_dir}/granularity_sweep.json"
    )
    
    # Experiment 2: Placement Study
    print("\n" + "="*60)
    print("EXPERIMENT 2: PLACEMENT STUDY")
    print("="*60)
    
    placement_results = run_placement_study(
        base_model_name="gpt2",
        m=64,
        k=8,
        eval_texts=eval_texts,
        save_path=f"{results_dir}/placement_study.json"
    )
    
    # Experiment 3: Cross-Seed Stability
    print("\n" + "="*60)
    print("EXPERIMENT 3: CROSS-SEED STABILITY")
    print("="*60)
    
    stability_results = run_cross_seed_stability_test(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=64,
        k=8,
        num_seeds=3,
        eval_texts=eval_texts,
        save_path=f"{results_dir}/cross_seed_stability.json"
    )
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("="*60)
    
    report = {
        "timestamp": timestamp,
        "experiments": {
            "granularity_sweep": granularity_results,
            "placement_study": placement_results,
            "cross_seed_stability": stability_results
        },
        "summary": {}
    }
    
    # Granularity sweep analysis
    print("\n1. GRANULARITY SWEEP ANALYSIS:")
    best_granularity = None
    best_granularity_score = float('inf')
    
    for config_name, config_results in granularity_results.items():
        quality_hit = config_results["quality"]["quality_hit_percent"]
        median_active = config_results["sparsity"]["overall_median_active_concepts"]
        score = quality_hit + max(0, median_active - 8) * 10
        
        print(f"  {config_name:12} | Quality: {quality_hit:6.2f}% | Active: {median_active:5.1f} | Score: {score:6.2f}")
        
        if score < best_granularity_score:
            best_granularity_score = score
            best_granularity = config_name
    
    report["summary"]["best_granularity"] = {
        "config": best_granularity,
        "score": best_granularity_score
    }
    
    print(f"  Best configuration: {best_granularity}")
    
    # Placement study analysis
    print("\n2. PLACEMENT STUDY ANALYSIS:")
    best_placement = None
    best_placement_score = float('inf')
    
    for placement_name, placement_results in placement_results.items():
        quality_hit = placement_results["quality"]["quality_hit_percent"]
        median_active = placement_results["sparsity"]["overall_median_active_concepts"]
        score = quality_hit + max(0, median_active - 8) * 10
        
        print(f"  {placement_name:8} | Quality: {quality_hit:6.2f}% | Active: {median_active:5.1f} | Score: {score:6.2f}")
        
        if score < best_placement_score:
            best_placement_score = score
            best_placement = placement_name
    
    report["summary"]["best_placement"] = {
        "placement": best_placement,
        "score": best_placement_score
    }
    
    print(f"  Best placement: {best_placement}")
    
    # Stability analysis
    print("\n3. CROSS-SEED STABILITY ANALYSIS:")
    if "stability" in stability_results:
        stability_score = stability_results["stability"]["overall_alignment"]
        stability_met = stability_results["stability"]["stability_criterion_met"]
        
        print(f"  Overall alignment: {stability_score:.3f}")
        print(f"  Criterion met (≥0.8): {stability_met}")
        
        report["summary"]["stability"] = {
            "alignment": stability_score,
            "criterion_met": stability_met
        }
    else:
        print("  Stability analysis not available")
        report["summary"]["stability"] = {"error": "Not available"}
    
    # Overall recommendations
    print("\n4. OVERALL RECOMMENDATIONS:")
    
    recommendations = []
    
    # Granularity recommendation
    if best_granularity:
        m, k = best_granularity.split('_')[1:]
        recommendations.append(f"Use m={m}, k={k} for optimal sparsity/quality trade-off")
    
    # Placement recommendation
    if best_placement:
        recommendations.append(f"Place concepts in {best_placement} blocks for best performance")
    
    # Stability recommendation
    if "stability" in report["summary"] and report["summary"]["stability"].get("criterion_met"):
        recommendations.append("Model shows good stability across seeds")
    else:
        recommendations.append("Consider stability improvements for better reproducibility")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    report["summary"]["recommendations"] = recommendations
    
    # Save comprehensive report
    with open(f"{results_dir}/comprehensive_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nComprehensive report saved to: {results_dir}/comprehensive_report.json")
    print("\n🎉 All experiments completed successfully!")


if __name__ == "__main__":
    main() 
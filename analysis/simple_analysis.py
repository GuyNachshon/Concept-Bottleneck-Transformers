#!/usr/bin/env python3
"""
Simple analysis of CBT results - what do we actually have?
"""

import json
import numpy as np
from pathlib import Path

def analyze_concept_results(results_path):
    """Analyze the concept analysis results."""
    
    with open(results_path / "concept_analysis_results.json", 'r') as f:
        results = json.load(f)
    
    print("🔍 CBT Results Analysis")
    print("=" * 50)
    
    # Basic model info
    config = results['metadata']['model_config']
    print(f"Model: {config['base_model_name']}")
    print(f"Blocks: {config['concept_blocks']}")
    print(f"Concepts per block: {config['m']}")
    print(f"Active concepts per token: {config['k']}")
    print(f"Alpha: {config['alpha']}")
    print()
    
    # Analyze concept labels
    concept_labels = results['concept_labels']
    
    print("📊 Concept Analysis by Block:")
    print("-" * 30)
    
    total_concepts = 0
    total_contexts = 0
    all_activations = []
    
    for block_name, block_concepts in concept_labels.items():
        print(f"\n{block_name.upper()}:")
        
        # Sort concepts by usage
        sorted_concepts = sorted(block_concepts.items(), 
                               key=lambda x: x[1]['num_contexts'], 
                               reverse=True)
        
        print(f"  Total concepts: {len(block_concepts)}")
        
        # Show top 5 most used concepts
        print("  Top 5 most used concepts:")
        for i, (concept_id, data) in enumerate(sorted_concepts[:5]):
            print(f"    {i+1}. Concept {concept_id}: {data['label']}")
            print(f"       Contexts: {data['num_contexts']:,}")
            print(f"       Avg activation: {data['avg_activation']:.3f}")
        
        # Show concept type distribution
        label_types = {}
        for concept_data in block_concepts.values():
            label = concept_data['label']
            base_type = label.split('_')[0] if '_' in label else label
            label_types[base_type] = label_types.get(base_type, 0) + 1
        
        print(f"  Concept types: {label_types}")
        
        # Statistics
        contexts = [c['num_contexts'] for c in block_concepts.values()]
        activations = [c['avg_activation'] for c in block_concepts.values()]
        
        print(f"  Total contexts: {sum(contexts):,}")
        print(f"  Mean activation: {np.mean(activations):.3f}")
        print(f"  Max activation: {np.max(activations):.3f}")
        
        total_concepts += len(block_concepts)
        total_contexts += sum(contexts)
        all_activations.extend(activations)
    
    print("\n" + "=" * 50)
    print("📈 OVERALL STATISTICS:")
    print(f"Total concepts across all blocks: {total_concepts}")
    print(f"Total contexts: {total_contexts:,}")
    print(f"Mean activation across all concepts: {np.mean(all_activations):.3f}")
    print(f"Std activation: {np.std(all_activations):.3f}")
    
    # Sparsity analysis
    active_concepts = sum(1 for block in concept_labels.values() 
                         for concept in block.values() 
                         if concept['num_contexts'] > 10)
    sparsity = 1 - (active_concepts / total_concepts)
    print(f"Active concepts (>10 contexts): {active_concepts}")
    print(f"Sparsity: {sparsity:.1%}")
    
    # What's missing?
    print("\n❌ WHAT'S MISSING:")
    print("- Actual concept contexts (what text triggers each concept)")
    print("- Causality analysis (what happens when concepts are removed)")
    print("- Human evaluation of concept quality")
    print("- Comparison with baseline methods")
    print("- Downstream task evaluation")
    
    return {
        'total_concepts': total_concepts,
        'total_contexts': total_contexts,
        'sparsity': sparsity,
        'mean_activation': np.mean(all_activations),
        'blocks': len(concept_labels)
    }

def analyze_performance_results(experiment_dir):
    """Analyze the performance results."""
    
    # Look for performance files
    perf_files = list(experiment_dir.glob("*results.json"))
    
    print("\n📊 PERFORMANCE RESULTS:")
    print("-" * 30)
    
    for perf_file in perf_files:
        print(f"\n{perf_file.name}:")
        try:
            with open(perf_file, 'r') as f:
                data = json.load(f)
            
            if 'quality' in data:
                quality = data['quality']
                print(f"  Perplexity: {quality.get('cbt_perplexity', 'N/A')}")
                print(f"  Base perplexity: {quality.get('base_perplexity', 'N/A')}")
                print(f"  Quality hit: {quality.get('quality_hit_percent', 'N/A')}%")
            
            if 'sparsity' in data:
                sparsity = data['sparsity']
                print(f"  Median active concepts: {sparsity.get('overall_median_active_concepts', 'N/A')}")
                
        except Exception as e:
            print(f"  Error reading {perf_file}: {e}")

def main():
    """Main analysis function."""
    
    # Analyze concept results
    results_path = Path("results/analysis/concept_analysis/cbt_model_stab_kl_m32_k4_a30_20250810_223738")
    
    if results_path.exists():
        concept_stats = analyze_concept_results(results_path)
    else:
        print("❌ Concept analysis results not found!")
        return
    
    # Analyze performance results
    experiment_dir = Path("results/experiments_20250810_160233")
    if experiment_dir.exists():
        analyze_performance_results(experiment_dir)
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY:")
    print(f"We have a working CBT model with {concept_stats['total_concepts']} concepts")
    print(f"across {concept_stats['blocks']} transformer blocks.")
    print(f"Sparsity: {concept_stats['sparsity']:.1%}")
    print(f"Mean activation: {concept_stats['mean_activation']:.3f}")
    print()
    print("This is a solid proof-of-concept, but we need:")
    print("1. Actual concept contexts (what text triggers each concept)")
    print("2. Causality testing (concept ablation)")
    print("3. Human evaluation")
    print("4. Comparison with baselines")

if __name__ == "__main__":
    main() 
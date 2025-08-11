#!/usr/bin/env python3
"""
Test concept causality by ablating concepts and measuring perplexity changes.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from cbt.model import CBTModel
from cbt.evaluator import CBTEvaluator

def load_model_and_results():
    """Load the CBT model and analysis results."""
    # Find latest results
    results_dir = Path("results/experiments_20250810_160233")
    model_file = "cbt_model_stab_kl_m32_k4_a30.pt"
    
    if not (results_dir / model_file).exists():
        print(f"❌ Model not found: {results_dir / model_file}")
        return None, None
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(results_dir / model_file, map_location=device)
    model.eval()
    
    # Load analysis results
    analysis_dir = Path("results/analysis/concept_analysis")
    analysis_dirs = list(analysis_dir.glob("cbt_model_*"))
    if not analysis_dirs:
        print("❌ No analysis results found!")
        return model, None
    
    latest_dir = max(analysis_dirs, key=lambda x: x.stat().st_mtime)
    results_file = latest_dir / "concept_analysis_results.json"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return model, results

def get_top_concepts(results, top_k=5):
    """Get the top concepts by usage."""
    top_concepts = []
    
    for block_name, block_labels in results['concept_labels'].items():
        # Sort by number of contexts
        block_concepts = [(int(idx), info) for idx, info in block_labels.items()]
        block_concepts.sort(key=lambda x: x[1]['num_contexts'], reverse=True)
        
        for concept_idx, info in block_concepts[:top_k]:
            top_concepts.append({
                'block': block_name,
                'concept_idx': concept_idx,
                'label': info['label'],
                'num_contexts': info['num_contexts'],
                'avg_activation': info['avg_activation']
            })
    
    return top_concepts

def ablate_concept(model, block_name, concept_idx):
    """Create a copy of the model with a specific concept ablated."""
    # Create a copy of the model
    ablated_model = type(model)(
        base_model_name=model.base_model_name,
        concept_blocks=model.concept_blocks,
        m=model.m,
        k=model.k,
        alpha=model.alpha
    )
    
    # Copy weights
    ablated_model.load_state_dict(model.state_dict())
    ablated_model.eval()
    
    # Zero out the specific concept
    block_idx = int(block_name.split('_')[1])
    concept_layer = ablated_model.concept_layers[block_idx]
    
    with torch.no_grad():
        # Zero out the concept encoder weights for this concept
        concept_layer.concept_encoder.weight[concept_idx] = 0.0
        if concept_layer.concept_encoder.bias is not None:
            concept_layer.concept_encoder.bias[concept_idx] = 0.0
    
    return ablated_model

def test_concept_causality(model, tokenizer, evaluator, top_concepts, num_texts=50):
    """Test causality by ablating concepts and measuring perplexity changes."""
    print("🧪 TESTING CONCEPT CAUSALITY")
    print("=" * 60)
    
    # Get baseline perplexity
    print("📊 Computing baseline perplexity...")
    baseline_perplexity = evaluator.evaluate_perplexity(model, num_texts=num_texts)
    print(f"Baseline perplexity: {baseline_perplexity:.3f}")
    
    # Test each top concept
    causality_results = []
    
    for concept_info in top_concepts:
        print(f"\n🔬 Testing concept {concept_info['concept_idx']} ({concept_info['label']}) from {concept_info['block']}:")
        print(f"   Contexts: {concept_info['num_contexts']}, Avg activation: {concept_info['avg_activation']:.3f}")
        
        # Ablate the concept
        ablated_model = ablate_concept(
            model, 
            concept_info['block'], 
            concept_info['concept_idx']
        )
        
        # Measure perplexity
        ablated_perplexity = evaluator.evaluate_perplexity(ablated_model, num_texts=num_texts)
        perplexity_change = ablated_perplexity - baseline_perplexity
        percent_change = (perplexity_change / baseline_perplexity) * 100
        
        print(f"   Ablated perplexity: {ablated_perplexity:.3f}")
        print(f"   Change: {perplexity_change:+.3f} ({percent_change:+.1f}%)")
        
        causality_results.append({
            'concept': concept_info,
            'baseline_perplexity': baseline_perplexity,
            'ablated_perplexity': ablated_perplexity,
            'perplexity_change': perplexity_change,
            'percent_change': percent_change
        })
    
    return causality_results

def display_causality_summary(causality_results):
    """Display a summary of causality results."""
    print("\n" + "=" * 60)
    print("📈 CAUSALITY SUMMARY")
    print("=" * 60)
    
    # Sort by absolute impact
    sorted_results = sorted(causality_results, key=lambda x: abs(x['percent_change']), reverse=True)
    
    print("\nTop concepts by impact:")
    for i, result in enumerate(sorted_results[:10]):
        concept = result['concept']
        print(f"{i+1:2d}. {concept['block']} Concept {concept['concept_idx']} ({concept['label']}): {result['percent_change']:+.1f}%")
    
    # Statistics
    impacts = [r['percent_change'] for r in causality_results]
    print(f"\n📊 Statistics:")
    print(f"   Mean impact: {np.mean(impacts):+.1f}%")
    print(f"   Median impact: {np.median(impacts):+.1f}%")
    print(f"   Max impact: {np.max(impacts):+.1f}%")
    print(f"   Min impact: {np.min(impacts):+.1f}%")
    print(f"   Std impact: {np.std(impacts):.1f}%")
    
    # Count significant impacts
    significant_positive = sum(1 for impact in impacts if impact > 1.0)
    significant_negative = sum(1 for impact in impacts if impact < -1.0)
    print(f"\n🎯 Significant impacts (>1%):")
    print(f"   Positive (worse performance): {significant_positive}")
    print(f"   Negative (better performance): {significant_negative}")
    print(f"   Total significant: {significant_positive + significant_negative}/{len(impacts)}")

def main():
    print("🔬 CBT Concept Causality Test")
    print("=" * 60)
    
    # Load model and results
    model, results = load_model_and_results()
    if model is None:
        return
    
    print(f"✅ Loaded model: {type(model).__name__}")
    print(f"✅ Loaded analysis results")
    
    # Setup tokenizer and evaluator
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    evaluator = CBTEvaluator(tokenizer)
    
    # Get top concepts
    top_concepts = get_top_concepts(results, top_k=10)
    print(f"\n🎯 Testing top {len(top_concepts)} concepts:")
    for concept in top_concepts:
        print(f"   {concept['block']} Concept {concept['concept_idx']}: {concept['label']} ({concept['num_contexts']} contexts)")
    
    # Run causality test
    causality_results = test_concept_causality(model, tokenizer, evaluator, top_concepts)
    
    # Display summary
    display_causality_summary(causality_results)
    
    # Save results
    output_file = "causality_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(causality_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main() 
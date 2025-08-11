#!/usr/bin/env python3
"""
Concept editing experiments - modify concepts and measure behavioral changes.
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
from cbt.evaluator import get_wikitext_eval_texts

def load_model():
    """Load the CBT model."""
    # Find latest results
    results_dir = Path("results/experiments_20250810_160233")
    model_file = "cbt_model_stab_kl_m32_k4_a30.pt"
    
    if not (results_dir / model_file).exists():
        print(f"❌ Model not found: {results_dir / model_file}")
        return None
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved checkpoint
    checkpoint = torch.load(results_dir / model_file, map_location=device)
    
    # Create model instance
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.3
    )
    
    # Load the state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def evaluate_perplexity(model, tokenizer, num_texts=50):
    """Evaluate perplexity on a set of texts."""
    eval_texts = get_wikitext_eval_texts(num_texts)
    losses = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for text in eval_texts:
            input_ids = tokenizer.encode(text, return_tensors='pt', 
                                       truncation=True, max_length=256).to(device)
            
            outputs = model(input_ids=input_ids, labels=input_ids)
            losses.append(outputs["loss"].item())
    
    perplexity = np.exp(np.mean(losses))
    return perplexity

def edit_concept_encoder(model, block_name, concept_idx, edit_type="zero", scale_factor=2.0):
    """Edit a concept encoder in different ways."""
    # Create a copy of the model
    edited_model = type(model)(
        base_model_name="gpt2",
        concept_blocks=model.concept_blocks,
        m=model.m,
        k=model.k,
        alpha=model.alpha
    )
    
    # Copy weights
    edited_model.load_state_dict(model.state_dict())
    edited_model.eval()
    
    # Get the concept layer
    concept_layer = edited_model.concept_layers[block_name]
    
    with torch.no_grad():
        if edit_type == "zero":
            # Zero out the concept
            concept_layer.encoder.weight[concept_idx] = 0.0
            if concept_layer.encoder.bias is not None:
                concept_layer.encoder.bias[concept_idx] = 0.0
            print(f"   Zeroed out {block_name} concept {concept_idx}")
            
        elif edit_type == "amplify":
            # Amplify the concept
            concept_layer.encoder.weight[concept_idx] *= scale_factor
            print(f"   Amplified {block_name} concept {concept_idx} by {scale_factor}x")
            
        elif edit_type == "invert":
            # Invert the concept
            concept_layer.encoder.weight[concept_idx] *= -1.0
            print(f"   Inverted {block_name} concept {concept_idx}")
            
        elif edit_type == "randomize":
            # Randomize the concept
            concept_layer.encoder.weight[concept_idx] = torch.randn_like(concept_layer.encoder.weight[concept_idx]) * 0.1
            print(f"   Randomized {block_name} concept {concept_idx}")
    
    return edited_model

def test_concept_editing(model, tokenizer, concept_info, edit_types=["zero", "amplify", "invert"]):
    """Test different editing strategies on a concept."""
    print(f"\n🔬 Testing concept editing for {concept_info['block']} Concept {concept_info['concept_idx']}")
    print(f"   Current label: {concept_info['label']}")
    print(f"   Impact: {concept_info['percent_change']:+.1f}%")
    
    # Get baseline perplexity
    baseline_perplexity = evaluate_perplexity(model, tokenizer, num_texts=30)
    print(f"   Baseline perplexity: {baseline_perplexity:.3f}")
    
    results = {
        'concept_info': concept_info,
        'baseline_perplexity': baseline_perplexity,
        'edits': {}
    }
    
    # Test each edit type
    for edit_type in edit_types:
        print(f"\n   Testing {edit_type} edit...")
        
        # Create edited model
        edited_model = edit_concept_encoder(
            model, 
            concept_info['block'], 
            concept_info['concept_idx'], 
            edit_type
        )
        
        # Measure perplexity
        edited_perplexity = evaluate_perplexity(edited_model, tokenizer, num_texts=30)
        perplexity_change = edited_perplexity - baseline_perplexity
        percent_change = (perplexity_change / baseline_perplexity) * 100
        
        print(f"   Edited perplexity: {edited_perplexity:.3f}")
        print(f"   Change: {perplexity_change:+.3f} ({percent_change:+.1f}%)")
        
        results['edits'][edit_type] = {
            'perplexity': edited_perplexity,
            'change': perplexity_change,
            'percent_change': percent_change
        }
    
    return results

def test_specific_editing_scenarios(model, tokenizer):
    """Test specific editing scenarios that might be interesting."""
    print("\n🎯 TESTING SPECIFIC EDITING SCENARIOS")
    print("=" * 60)
    
    # Load concept data
    with open("top_concept_contexts.json", 'r') as f:
        concept_data = json.load(f)
    
    # Focus on the most important concept (block_7_24)
    top_concept = concept_data['block_7_24']
    concept_info = top_concept['concept_info']
    
    print(f"🎯 Testing the MOST important concept:")
    print(f"   {concept_info['block']} Concept {concept_info['concept_idx']} ({concept_info['label']})")
    print(f"   Impact: {concept_info['percent_change']:+.1f}%")
    print(f"   Contexts: {len(top_concept['contexts'])}")
    
    # Test different editing strategies
    edit_results = test_concept_editing(
        model, 
        tokenizer, 
        concept_info,
        edit_types=["zero", "amplify", "invert", "randomize"]
    )
    
    # Test a more targeted edit - modify the concept to be more selective
    print(f"\n🎯 Testing targeted concept modification...")
    
    # Create a model with the concept modified to be more selective
    targeted_model = edit_concept_encoder(
        model,
        concept_info['block'],
        concept_info['concept_idx'],
        edit_type="amplify",
        scale_factor=0.5  # Reduce the concept strength
    )
    
    targeted_perplexity = evaluate_perplexity(targeted_model, tokenizer, num_texts=30)
    targeted_change = targeted_perplexity - edit_results['baseline_perplexity']
    targeted_percent = (targeted_change / edit_results['baseline_perplexity']) * 100
    
    print(f"   Reduced strength perplexity: {targeted_perplexity:.3f}")
    print(f"   Change: {targeted_change:+.3f} ({targeted_percent:+.1f}%)")
    
    edit_results['edits']['reduce_strength'] = {
        'perplexity': targeted_perplexity,
        'change': targeted_change,
        'percent_change': targeted_percent
    }
    
    return edit_results

def analyze_editing_results(results):
    """Analyze the results of concept editing experiments."""
    print("\n" + "=" * 60)
    print("📊 CONCEPT EDITING ANALYSIS")
    print("=" * 60)
    
    concept_info = results['concept_info']
    baseline = results['baseline_perplexity']
    
    print(f"Concept: {concept_info['block']} Concept {concept_info['concept_idx']} ({concept_info['label']})")
    print(f"Baseline perplexity: {baseline:.3f}")
    print()
    
    print("Edit Type | Perplexity | Change | % Change")
    print("-" * 50)
    
    for edit_type, edit_data in results['edits'].items():
        print(f"{edit_type:12} | {edit_data['perplexity']:9.3f} | {edit_data['change']:+6.3f} | {edit_data['percent_change']:+6.1f}%")
    
    # Find the most impactful edit
    most_impactful = max(results['edits'].items(), key=lambda x: abs(x[1]['percent_change']))
    print(f"\n🎯 Most impactful edit: {most_impactful[0]} ({most_impactful[1]['percent_change']:+.1f}%)")
    
    # Analysis
    print(f"\n📈 ANALYSIS:")
    print(f"   - Zeroing the concept: {results['edits']['zero']['percent_change']:+.1f}%")
    print(f"   - Amplifying the concept: {results['edits']['amplify']['percent_change']:+.1f}%")
    print(f"   - Inverting the concept: {results['edits']['invert']['percent_change']:+.1f}%")
    print(f"   - Randomizing the concept: {results['edits']['randomize']['percent_change']:+.1f}%")
    
    # Interpret results
    zero_impact = results['edits']['zero']['percent_change']
    if zero_impact > 5.0:
        print(f"   🚨 This concept is CRITICAL - removing it hurts performance by {zero_impact:.1f}%")
    elif zero_impact > 1.0:
        print(f"   ⚠️  This concept is IMPORTANT - removing it hurts performance by {zero_impact:.1f}%")
    else:
        print(f"   ✅ This concept is MODERATE - removing it has minimal impact ({zero_impact:.1f}%)")

def main():
    print("🔬 CONCEPT EDITING EXPERIMENTS")
    print("=" * 60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    print("✅ Loaded model")
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run editing experiments
    results = test_specific_editing_scenarios(model, tokenizer)
    
    # Analyze results
    analyze_editing_results(results)
    
    # Save results
    output_file = "concept_editing_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main() 
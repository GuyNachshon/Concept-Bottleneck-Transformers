#!/usr/bin/env python3
"""
Ablation Testing Demo for CBT models.
"""

import torch
from transformers import GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.model import CBTModel
from cbt.ablation_tools import ConceptAblator, ConceptEditor, AblationAnalyzer


def create_test_texts():
    """Create diverse test texts for ablation analysis."""
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A cat sat on the mat and purred contentedly.",
        "The weather is cold and windy today.",
        "She seems happy about the good news.",
        "The red car is faster than the blue one.",
        "If it rains tomorrow, I will stay home.",
        "What time is the meeting scheduled for?",
        "The children's toys are scattered everywhere.",
        "He can swim very well in the deep water.",
        "The book is on the shelf above the desk."
    ]
    return test_texts


def main():
    """Main ablation testing demonstration."""
    print("Setting up CBT ablation testing demo...")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load trained model
    print("Loading CBT model...")
    model_path = "cbt_advanced_model.pt"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        model = CBTModel(
            base_model_name="gpt2",
            concept_blocks=[4, 5, 6, 7],
            d_model=768,
            m=64,
            k=8,
            alpha=1.0
        )
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No trained model found, creating a new one for demo")
        model = CBTModel(
            base_model_name="gpt2",
            concept_blocks=[4, 5, 6, 7],
            d_model=768,
            m=64,
            k=8,
            alpha=1.0
        )
    
    model = model.to(device)
    model.eval()
    
    # Create test texts
    test_texts = create_test_texts()
    print(f"Created {len(test_texts)} test texts")
    
    # Initialize ablation tools
    print("Initializing ablation tools...")
    ablator = ConceptAblator(model, tokenizer, device)
    editor = ConceptEditor(model, tokenizer, device)
    analyzer = AblationAnalyzer()
    
    # Get concept keys from analysis results
    print("Getting concept keys for ablation...")
    concept_keys = []
    for block_idx in [4, 5, 6, 7]:
        for concept_idx in range(64):
            concept_keys.append(f"block_{block_idx}_concept_{concept_idx}")
    
    # Select a subset for demo (to save time)
    demo_concepts = concept_keys[:10]  # First 10 concepts
    print(f"Selected {len(demo_concepts)} concepts for ablation demo")
    
    # Run ablation analysis
    print("\n" + "="*50)
    print("RUNNING ABLATION ANALYSIS")
    print("="*50)
    
    print("This will test the effect of turning off each concept...")
    print("(This may take a few minutes)")
    
    # Measure ablation effects
    ablation_results = ablator.measure_ablation_effect(
        test_texts=test_texts[:5],  # Use subset for speed
        concept_keys=demo_concepts,
        ablation_type="zero",
        metrics=["perplexity"]
    )
    
    # Analyze results
    print("\n" + "="*50)
    print("ABLATION RESULTS")
    print("="*50)
    
    # Find most impactful concepts
    effects = []
    for concept_key, result in ablation_results.items():
        if "perplexity" in result["difference"]:
            effect = result["difference"]["perplexity"]
            effects.append((concept_key, effect))
    
    # Sort by absolute effect size
    effects.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("Most impactful concepts (by perplexity change):")
    for i, (concept_key, effect) in enumerate(effects[:5]):
        print(f"  {i+1}. {concept_key}: {effect:+.4f} perplexity change")
    
    print("\nLeast impactful concepts:")
    for i, (concept_key, effect) in enumerate(effects[-5:]):
        print(f"  {i+1}. {concept_key}: {effect:+.4f} perplexity change")
    
    # Create visualizations
    print("\nCreating ablation visualizations...")
    
    # Plot ablation effects
    fig1 = analyzer.plot_ablation_effects(
        ablation_results, 
        metric="perplexity", 
        top_k=8
    )
    fig1.savefig("ablation_effects.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Generate ablation report
    print("Generating ablation report...")
    report = analyzer.generate_ablation_report(
        ablation_results,
        save_path="ablation_report.json"
    )
    
    print(f"Report saved to ablation_report.json")
    print(f"Total concepts analyzed: {report['summary']['total_concepts']}")
    
    # Concept editing demo
    print("\n" + "="*50)
    print("CONCEPT EDITING DEMO")
    print("="*50)
    
    # Test concept editing with generation
    prompt = "The weather is"
    print(f"Testing concept editing with prompt: '{prompt}'")
    
    # Generate baseline
    print("\nBaseline generation:")
    baseline_text = editor.generate_with_edits(prompt, max_length=20)
    print(f"  {baseline_text}")
    
    # Test editing a concept
    if effects:
        most_impactful = effects[0][0]
        print(f"\nEditing most impactful concept: {most_impactful}")
        
        # Edit the concept
        editor.edit_concept_activation(most_impactful, activation_value=0.5)
        
        # Generate with edit
        edited_text = editor.generate_with_edits(prompt, max_length=20)
        print(f"With concept edit: {edited_text}")
        
        # Clear edits
        editor.clear_edits()
    
    # Compare generations
    print("\nComparing generations with concept edits...")
    concept_edits = {
        demo_concepts[0]: 0.8,  # Boost first concept
        demo_concepts[1]: 0.2,  # Reduce second concept
    }
    
    comparison = editor.compare_generations(
        prompt=prompt,
        concept_edits=concept_edits,
        max_length=20,
        num_samples=2
    )
    
    print("\nBaseline generations:")
    for i, text in enumerate(comparison["baseline"]):
        print(f"  {i+1}. {text}")
    
    print("\nEdited generations:")
    for i, text in enumerate(comparison["edited"]):
        print(f"  {i+1}. {text}")
    
    # Summary
    print("\n" + "="*50)
    print("ABLATION TESTING COMPLETE")
    print("="*50)
    print("Generated files:")
    print("  - ablation_effects.png (visualization)")
    print("  - ablation_report.json (detailed results)")
    
    print("\nKey findings:")
    print(f"  - Analyzed {len(demo_concepts)} concepts")
    print(f"  - Most impactful: {effects[0][0]} ({effects[0][1]:+.4f} perplexity)")
    print(f"  - Least impactful: {effects[-1][0]} ({effects[-1][1]:+.4f} perplexity)")
    
    print("\nNext steps:")
    print("  - Run full ablation on all concepts")
    print("  - Test different ablation types (random, noise)")
    print("  - Analyze concept interactions")
    print("  - Build interactive concept editing interface")


if __name__ == "__main__":
    main() 
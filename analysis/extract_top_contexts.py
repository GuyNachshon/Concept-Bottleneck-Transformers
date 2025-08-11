#!/usr/bin/env python3
"""
Extract contexts for the top concepts directly from the model.
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

def extract_contexts_for_concept(model, tokenizer, block_name, concept_idx, num_texts=100):
    """Extract contexts where a specific concept is most active."""
    print(f"🔍 Extracting contexts for {block_name} Concept {concept_idx}...")
    
    # Get evaluation texts
    eval_texts = get_wikitext_eval_texts(num_texts)
    
    # Collect contexts where this concept is active
    contexts = []
    
    with torch.no_grad():
        for text in eval_texts:
            input_ids = tokenizer.encode(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=256
            ).to(next(model.parameters()).device)
            
            outputs = model(input_ids=input_ids, return_concepts=True)
            activations = outputs["concept_activations"]
            
            if block_name in activations:
                block_acts = activations[block_name][0].cpu().numpy()  # [seq_len, m]
                
                # Get tokens for context
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                
                for token_idx in range(min(len(tokens), block_acts.shape[0])):
                    # Check if this concept is active for this token
                    if block_acts[token_idx, concept_idx] > 0.1:  # Threshold
                        # Get context window
                        start = max(0, token_idx - 5)
                        end = min(len(tokens), token_idx + 6)
                        context = ' '.join(tokens[start:end])
                        
                        contexts.append({
                            'context': context,
                            'activation': float(block_acts[token_idx, concept_idx]),
                            'token': tokens[token_idx],
                            'token_idx': token_idx
                        })
    
    # Sort by activation strength
    contexts.sort(key=lambda x: x['activation'], reverse=True)
    return contexts

def display_top_contexts(contexts, concept_info, top_k=10):
    """Display the top contexts for a concept."""
    print(f"\n📦 {concept_info['block']} Concept {concept_info['concept_idx']} ({concept_info['label']})")
    print(f"   Impact: {concept_info['percent_change']:+.1f}% | Contexts: {len(contexts)}")
    print(f"   Avg activation: {concept_info['avg_activation']:.3f}")
    print("-" * 80)
    
    if not contexts:
        print("   No contexts found!")
        return
    
    print("   Top contexts by activation strength:")
    for i, ctx in enumerate(contexts[:top_k]):
        print(f"   {i+1:2d}. '{ctx['context']}'")
        print(f"       Activation: {ctx['activation']:.3f} | Token: '{ctx['token']}'")
    
    if len(contexts) > top_k:
        print(f"   ... and {len(contexts) - top_k} more contexts")

def main():
    print("🔍 EXTRACTING TOP CONCEPT CONTEXTS")
    print("=" * 60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    print("✅ Loaded model")
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Top concepts from causality test (sorted by impact)
    top_concepts = [
        {"block": "block_7", "concept_idx": 24, "label": "spatial_24", "percent_change": 8.45, "avg_activation": 0.999},
        {"block": "block_6", "concept_idx": 7, "label": "spatial_7", "percent_change": 2.09, "avg_activation": 0.987},
        {"block": "block_6", "concept_idx": 25, "label": "spatial_25", "percent_change": 1.30, "avg_activation": 0.895},
        {"block": "block_5", "concept_idx": 26, "label": "spatial_26", "percent_change": 1.00, "avg_activation": 0.921},
        {"block": "block_4", "concept_idx": 15, "label": "spatial_15", "percent_change": 0.79, "avg_activation": 0.996},
    ]
    
    print(f"\n🎯 Extracting contexts for top {len(top_concepts)} concepts...")
    
    # Extract contexts for each top concept
    all_contexts = {}
    
    for concept_info in top_concepts:
        contexts = extract_contexts_for_concept(
            model, 
            tokenizer, 
            concept_info['block'], 
            concept_info['concept_idx']
        )
        
        all_contexts[f"{concept_info['block']}_{concept_info['concept_idx']}"] = {
            'concept_info': concept_info,
            'contexts': contexts
        }
        
        display_top_contexts(contexts, concept_info)
    
    # Save results
    output_file = "top_concept_contexts.json"
    
    # Convert to JSON-serializable format
    serializable_contexts = {}
    for key, data in all_contexts.items():
        serializable_contexts[key] = {
            'concept_info': data['concept_info'],
            'contexts': data['contexts'][:20]  # Save top 20 contexts
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_contexts, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for key, data in all_contexts.items():
        concept_info = data['concept_info']
        contexts = data['contexts']
        print(f"{concept_info['block']} Concept {concept_info['concept_idx']}: {len(contexts)} contexts, {concept_info['percent_change']:+.1f}% impact")

if __name__ == "__main__":
    main() 
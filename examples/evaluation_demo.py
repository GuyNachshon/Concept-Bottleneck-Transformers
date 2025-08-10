#!/usr/bin/env python3
"""
Evaluation Demo for CBT Success Criteria
"""

import torch
import os
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.model import CBTModel
from cbt.evaluator import CBTEvaluator


def main():
    print("=== CBT Success Criteria Evaluation Demo ===\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    model_path = "cbt_advanced_model.pt"
    
    if not os.path.exists(model_path):
        print("❌ No trained CBT model found. Please train a model first.")
        return
    
    # Load CBT model
    cbt_model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        d_model=768,
        m=64,
        k=8,
        alpha=1.0
    )
    checkpoint = torch.load(model_path, map_location=device)
    cbt_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded CBT model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load base model
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("✅ Loaded base GPT-2 model")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Loaded tokenizer")
    
    # Move models to device
    cbt_model.to(device)
    base_model.to(device)
    
    # Get evaluation texts from WikiText dataset
    from cbt.evaluator import get_wikitext_eval_texts
    eval_texts = get_wikitext_eval_texts(num_samples=15)
    print(f"Using {len(eval_texts)} evaluation texts from WikiText")
    
    # Create evaluator
    evaluator = CBTEvaluator(cbt_model, base_model, tokenizer, device)
    
    # Run evaluation
    print("\n" + "="*50)
    results = evaluator.evaluate_all_criteria(eval_texts)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"✅ Quality criterion met: {results['quality']['quality_criterion_met']}")
    print(f"✅ Sparsity criterion met: {results['sparsity']['overall_sparsity_criterion_met']}")
    print(f"🎯 Overall success: {results['overall']['all_criteria_met']}")
    
    if results['overall']['all_criteria_met']:
        print("\n🎉 CBT implementation meets all evaluated success criteria!")
    else:
        print("\n⚠️  Some criteria not met. Consider adjusting hyperparameters.")


if __name__ == "__main__":
    main() 
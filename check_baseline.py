#!/usr/bin/env python3
"""
Check baseline GPT-2 perplexity on WikiText-2
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

def main():
    print("Checking GPT-2 baseline perplexity on WikiText-2...")
    
    # Load models and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Load WikiText-2 test set
    dataset = load_dataset('salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    
    losses = []
    texts_used = []
    
    print(f"Evaluating on {len(dataset)} test examples...")
    
    for i, item in enumerate(dataset):
        if i >= 200:  # Limit to first 200 examples
            break
            
        text = item['text'].strip()
        
        # Filter for reasonable length texts (not too short, not too long)
        if len(text) > 50 and len(text) < 500 and text:
            input_ids = tokenizer.encode(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=256
            )
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                losses.append(outputs.loss.item())
                texts_used.append(text[:100] + "..." if len(text) > 100 else text)
    
    if losses:
        baseline_perplexity = np.exp(np.mean(losses))
        print(f"\nBaseline GPT-2 perplexity on WikiText-2 test: {baseline_perplexity:.2f}")
        print(f"Evaluated on {len(losses)} texts")
        print(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")
        
        # Show a few example texts
        print(f"\nExample texts used:")
        for i, text in enumerate(texts_used[:5]):
            print(f"  {i+1}. {text}")
    else:
        print("No valid texts found!")

if __name__ == "__main__":
    main() 
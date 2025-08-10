#!/usr/bin/env python3
"""
Check GPT-2 baseline perplexity on WikiText-2
This helps identify if our evaluation setup is correct.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("=== GPT-2 Baseline Perplexity Check ===")
    print("This checks if our evaluation setup is reasonable.\n")
    
    # Load models and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Test different splits and text lengths
    splits_to_test = ["train", "validation", "test"]
    
    for split in splits_to_test:
        print(f"\n{'='*50}")
        print(f"Testing WikiText-2 {split} split")
        print(f"{'='*50}")
        
        # Load dataset
        dataset = load_dataset('salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
        
        # Test different text length filters
        length_filters = [
            ("short", 10, 200),
            ("medium", 50, 1000), 
            ("long", 100, 2000)
        ]
        
        for filter_name, min_len, max_len in length_filters:
            print(f"\n--- {filter_name} texts ({min_len}-{max_len} chars) ---")
            
            losses = []
            texts_used = []
            
            for i, item in enumerate(dataset):
                if i >= 500:  # Limit to first 500 examples
                    break
                    
                text = item['text'].strip()
                
                if len(text) > min_len and len(text) < max_len and text:
                    input_ids = tokenizer.encode(
                        text, 
                        return_tensors='pt', 
                        truncation=True, 
                        max_length=256
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_ids, labels=input_ids)
                        losses.append(outputs.loss.item())
                        texts_used.append(text[:100] + "..." if len(text) > 100 else text)
                    
                    if len(losses) >= 100:  # Stop after 100 valid texts
                        break
            
            if losses:
                perplexity = np.exp(np.mean(losses))
                print(f"  Perplexity: {perplexity:.2f}")
                print(f"  Loss mean: {np.mean(losses):.4f}")
                print(f"  Loss std: {np.std(losses):.4f}")
                print(f"  Loss range: {min(losses):.4f} to {max(losses):.4f}")
                print(f"  Texts evaluated: {len(losses)}")
                
                # Show a few example texts
                print(f"  Example texts:")
                for i, text in enumerate(texts_used[:3]):
                    print(f"    {i+1}. {text}")
            else:
                print(f"  No valid texts found!")
    
    print(f"\n{'='*50}")
    print("EXPECTED BASELINE PERPLEXITY")
    print(f"{'='*50}")
    print("According to literature, GPT-2 should achieve:")
    print("- WikiText-2 test perplexity: ~35-40")
    print("- WikiText-2 validation perplexity: ~35-40")
    print("- WikiText-2 train perplexity: ~30-35")
    print("\nIf you see much lower values (<20), there might be:")
    print("1. Data contamination (evaluating on training data)")
    print("2. Overly simple/short texts")
    print("3. Evaluation setup issues")

if __name__ == "__main__":
    main() 
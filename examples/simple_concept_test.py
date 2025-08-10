#!/usr/bin/env python3
"""
Simple test to debug concept mining.
"""

import torch
from transformers import GPT2Tokenizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.model import CBTModel
from cbt.analyzer import ConceptMiner


def main():
    """Simple concept mining test."""
    print("Simple concept mining test...")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create a simple model
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
    
    # Test with a simple sentence
    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"Testing with: {test_text}")
    
    # Tokenize
    input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)
    print(f"Input shape: {input_ids.shape}")
    
    # Get concept activations
    with torch.no_grad():
        outputs = model(input_ids, return_concepts=True)
        concept_activations = outputs["concept_activations"]
        
        print(f"Got concept activations for {len(concept_activations)} blocks")
        
        for block_name, activations in concept_activations.items():
            print(f"\n{block_name}:")
            print(f"  Shape: {activations.shape}")
            print(f"  Min: {activations.min().item():.4f}")
            print(f"  Max: {activations.max().item():.4f}")
            print(f"  Mean: {activations.mean().item():.4f}")
            print(f"  Non-zero: {(activations > 0).sum().item()}")
            print(f"  Sparsity: {(activations == 0).float().mean().item():.3f}")
            
            # Check for activations above different thresholds
            for threshold in [0.01, 0.05, 0.1, 0.2]:
                above_threshold = (activations > threshold).sum().item()
                print(f"  Above {threshold}: {above_threshold}")
    
    # Test concept miner
    print("\nTesting ConceptMiner...")
    miner = ConceptMiner(model, tokenizer, device)
    
    # Create a simple dataset
    from torch.utils.data import DataLoader, Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.encodings = []
            for text in texts:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=32,
                    return_tensors='pt'
                )
                self.encodings.append(encoding)
        
        def __len__(self):
            return len(self.encodings)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings[idx]['input_ids'].squeeze(),
                'attention_mask': self.encodings[idx]['attention_mask'].squeeze()
            }
    
    # Test with multiple sentences
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A cat sat on the mat.",
        "The sky is blue.",
        "Hello, how are you?"
    ]
    
    dataset = SimpleDataset(test_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Mine concepts with different thresholds
    for threshold in [0.01, 0.05, 0.1]:
        print(f"\nMining with threshold {threshold}...")
        results = miner.mine_concepts(dataloader, max_samples=10, activation_threshold=threshold)
        print(f"Found {len(results)} concepts")
        
        if results:
            for concept_key, data in list(results.items())[:3]:
                print(f"  {concept_key}: {data['num_activations']} activations")


if __name__ == "__main__":
    main() 
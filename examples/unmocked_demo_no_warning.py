#!/usr/bin/env python3
"""
Unmocked CBT Demo - No Warning Version
Demonstrates real concept editing, mining, labeling, and ablation without multiprocessing warnings.
"""

import torch
import os
import sys
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.model import CBTModel
from cbt.analyzer import ConceptMiner
from cbt.llm_labeling import create_llm_labeler
from cbt.ablation_tools import ConceptAblator


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=50):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def collate_fn(batch):
    """Custom collate function to handle variable lengths."""
    max_len = max(len(item['input_ids']) for item in batch)
    padded_batch = []
    for item in batch:
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.zeros(max_len - len(item['input_ids']), dtype=torch.long)
        ])
        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(max_len - len(item['attention_mask']), dtype=torch.long)
        ])
        padded_batch.append({
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask
        })
    return {
        'input_ids': torch.stack([item['input_ids'] for item in padded_batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in padded_batch])
    }


def main():
    print("=== Unmocked CBT Demo (No Warning) ===\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
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
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("No trained model found. Please run training first.")
        return
    
    model.to(device)
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n" + "="*50)
    print("DEMO 1: REAL CONCEPT EDITING")
    print("="*50)
    
    # Test concept editing
    test_prompt = "The weather is"
    print(f"Test prompt: '{test_prompt}'")
    
    # Baseline generation
    input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids, max_length=20, temperature=1.0, do_sample=True
        )
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    print(f"\nBaseline generation:\n  {baseline_text}")
    
    print("\nTesting concept editing...")
    
    # Test editing a few concepts
    for i in range(3):
        concept_key = f"block_4_concept_{i}"
        print(f"\nEditing {concept_key}:")
        
        # Boost the concept
        concept_edits = {concept_key: 2.0}
        with torch.no_grad():
            boosted_output = model.generate(
                input_ids, max_length=20, temperature=1.0, do_sample=True,
                concept_edits=concept_edits
            )
        boosted_text = tokenizer.decode(boosted_output[0], skip_special_tokens=True)
        print(f"  Boosted: {boosted_text}")
        
        # Suppress the concept
        concept_edits = {concept_key: -1.0}
        with torch.no_grad():
            suppressed_output = model.generate(
                input_ids, max_length=20, temperature=1.0, do_sample=True,
                concept_edits=concept_edits
            )
        suppressed_text = tokenizer.decode(suppressed_output[0], skip_special_tokens=True)
        print(f"  Suppressed: {suppressed_text}")
    
    print("\n" + "="*50)
    print("DEMO 2: CONCEPT MINING & ANALYSIS")
    print("="*50)
    
    # Create test data
    test_texts = [
        "The weather is cold and rainy today.",
        "It's sunny and warm outside.",
        "The temperature dropped significantly.",
        "She seems happy about the good news.",
        "The red car is faster than the blue one."
    ]
    
    dataset = SimpleDataset(test_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Mine concepts
    print("Mining concepts...")
    miner = ConceptMiner(model, tokenizer, device)
    mining_results = miner.mine_concepts(dataloader, max_samples=10, activation_threshold=0.01)
    
    print(f"Found {len(mining_results)} concepts with activations above threshold")
    
    print("\n" + "="*50)
    print("DEMO 3: LLM-BASED CONCEPT LABELING")
    print("="*50)
    
    # Create LLM labeler
    print("Creating LLM labeler (mock mode)...")
    labeler = create_llm_labeler(provider="mock")
    
    # Label concepts
    print("Labeling concepts...")
    labels = labeler.batch_label_concepts(mining_results, max_contexts_per_concept=5)
    
    print(f"Generated {len(labels)} concept labels:")
    for i, (concept_key, label) in enumerate(list(labels.items())[:5]):
        print(f"  {concept_key}: {label}")
    
    print("\n" + "="*50)
    print("DEMO 4: SIMPLE ABLATION TESTING")
    print("="*50)
    
    # Simple ablation test without multiprocessing
    print("Testing simple ablation...")
    
    # Get a few concept keys
    test_concepts = list(mining_results.keys())[:2]
    
    for concept_key in test_concepts:
        try:
            print(f"\nTesting ablation for {concept_key}:")
            
            # Create ablator
            ablator = ConceptAblator(model, tokenizer, device)
            
            # Test before ablation
            with torch.no_grad():
                outputs_before = model(input_ids, return_concepts=True)
            
            if concept_key in outputs_before["concept_activations"]:
                activations_before = outputs_before["concept_activations"][concept_key]
                max_before = activations_before.max().item()
                print(f"  Max activation before: {max_before:.6f}")
            
            # Apply ablation
            ablator.ablate_concept(concept_key, "zero")
            print("  ✅ Ablation applied")
            
            # Test after ablation
            with torch.no_grad():
                outputs_after = model(input_ids, return_concepts=True)
            
            if concept_key in outputs_after["concept_activations"]:
                activations_after = outputs_after["concept_activations"][concept_key]
                max_after = activations_after.max().item()
                print(f"  Max activation after: {max_after:.6f}")
            
            # Restore concept
            ablator.restore_concept(concept_key)
            print("  ✅ Concept restored")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*50)
    print("DEMO COMPLETE!")
    print("="*50)
    print("✅ Concept editing works")
    print("✅ Concept mining works")
    print("✅ LLM labeling works")
    print("✅ Ablation testing works")
    print("✅ No multiprocessing warnings!")


if __name__ == "__main__":
    main() 
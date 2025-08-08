#!/usr/bin/env python3
"""
WikiText Training Example
Train CBT model on WikiText-2 dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.model import CBTModel
from cbt.training import CBTTrainer


class WikiTextDataset(Dataset):
    """Dataset wrapper for WikiText."""
    
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        self.tokenized_texts = []
        for item in dataset:
            text = item['text']
            if text.strip():  # Skip empty texts
                tokens = self.tokenizer.encode(
                    text, 
                    truncation=True, 
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                if tokens.size(1) > 1:  # Skip single-token texts
                    self.tokenized_texts.append(tokens)
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        return {
            "input_ids": tokens.squeeze(0),
            "attention_mask": torch.ones_like(tokens.squeeze(0))
        }


def collate_fn(batch):
    """Custom collate function to pad sequences."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    
    # Pad sequences
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        padding_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)]))
        padded_attention_masks.append(torch.cat([mask, torch.zeros(padding_len, dtype=torch.long)]))
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks)
    }


def main():
    print("=== WikiText CBT Training ===\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    print(f"Loaded {len(dataset)} training examples")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset wrapper
    print("Preparing dataset...")
    train_dataset = WikiTextDataset(dataset, tokenizer, max_length=128)
    print(f"Prepared {len(train_dataset)} training sequences")
    
    # Create validation dataset (use a subset of training for simplicity)
    val_size = min(1000, len(train_dataset) // 10)
    val_dataset = torch.utils.data.Subset(train_dataset, range(val_size))
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
    # Create CBT model
    print("Creating CBT model...")
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],  # Middle 4 blocks
        d_model=768,
        m=64,  # 64 concepts per block
        k=8,   # 8 active concepts per token
        alpha=0.0  # Start with no concept influence
    )
    
    print(f"Model parameters: {model.count_parameters()}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = CBTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        weight_decay=0.01,
        device=device,
        use_wandb=False,  # Set to True if you have wandb
        use_advanced_losses=True,
        advanced_loss_config={
            "orthogonality_weight": 0.01,
            "stability_weight": 0.01,
            "kl_weight": 1.0,
            "concept_dropout_weight": 0.01
        }
    )
    
    # Training schedule
    print("Starting training...")
    alpha_schedule = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    trainer.train(
        num_epochs=len(alpha_schedule),
        alpha_schedule=alpha_schedule,
        save_path="cbt_wikitext_model.pt"
    )
    
    print("\nTraining completed!")
    print("Model saved to: cbt_wikitext_model.pt")
    
    # Test the trained model
    print("\nTesting trained model...")
    model.eval()
    
    test_text = "The weather is"
    input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # Generate with concept activations
        outputs = model(input_ids, return_concepts=True)
        concept_activations = outputs["concept_activations"]
        
        # Generate text
        generated_ids = model.generate(
            input_ids,
            max_length=20,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"Input: {test_text}")
        print(f"Generated: {generated_text}")
        
        # Show concept activations
        print("\nConcept activations:")
        for block_name, concepts in concept_activations.items():
            active_concepts = (concepts > 0).sum(dim=-1)
            sparsity = (concepts > 0).float().mean()
            print(f"  {block_name}: {active_concepts.mean().item():.1f} active concepts per token (sparsity: {sparsity:.3f})")


if __name__ == "__main__":
    main() 
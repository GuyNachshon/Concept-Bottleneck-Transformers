#!/usr/bin/env python3
"""
Advanced training example for CBT with advanced loss functions.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import os
import sys
from datasets import load_dataset

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.model import CBTModel
from cbt.training import CBTTrainer


class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
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


def create_sample_data():
    """Create sample training data."""
    dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    return dataset


def main():
    """Main training function with advanced losses."""
    print("Setting up CBT training with advanced losses...")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample data
    sample_texts = create_sample_data()
    train_texts = sample_texts[:15]
    val_texts = sample_texts[15:]
    
    # Create datasets
    train_dataset = SimpleTextDataset(train_texts, tokenizer, max_length=64)
    val_dataset = SimpleTextDataset(val_texts, tokenizer, max_length=64)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create CBT model
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],  # Middle 4 blocks
        d_model=768,
        m=64,  # 64 concepts
        k=8,   # 8 active concepts per token
        alpha=0.0,  # Start with no concept layer effect
        dropout=0.1
    )
    
    # Print model info
    param_counts = model.count_parameters()
    print(f"Model parameters:")
    print(f"  Base model: {param_counts['base_model']:,}")
    print(f"  Concept layers: {param_counts['concept_layers']:,}")
    print(f"  Total: {param_counts['total']:,}")
    
    # Advanced loss configuration
    advanced_loss_config = {
        "orthogonality_weight": 0.01,      # Prevent concept duplication
        "stability_weight": 0.01,          # Maintain concept stability
        "kl_weight": 1.0,                  # KL distillation from base model
        "concept_dropout_weight": 0.01,    # Ensure distinct concept roles
        "kl_temperature": 1.0,             # Temperature for KL loss
        "anchor_update_rate": 0.1,         # Rate for stability anchor updates
        "concept_dropout_rate": 0.1        # Rate for concept dropout
    }
    
    print(f"Advanced loss configuration:")
    for key, value in advanced_loss_config.items():
        print(f"  {key}: {value}")
    
    # Create trainer with advanced losses
    trainer = CBTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        weight_decay=0.01,
        device=device,
        use_wandb=False,  # Set to True if you have wandb set up
        use_advanced_losses=True,
        advanced_loss_config=advanced_loss_config
    )
    
    # Custom alpha schedule: gradual ramp-up with longer warm-up
    alpha_schedule = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
    
    print("Starting training with advanced losses...")
    trainer.train(
        num_epochs=10,
        alpha_schedule=alpha_schedule,
        save_path="cbt_advanced_model.pt"
    )
    
    # Test the trained model
    print("\nTesting the trained model with advanced losses...")
    model.eval()
    
    # Test text generation
    test_prompt = "The concept of"
    input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # Generate with concept layer
        model.set_alpha(1.0)
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
        
        print(f"Generated text: {generated_text}")
        
        # Show concept activations
        print("\nConcept activations:")
        for block_name, concepts in concept_activations.items():
            # Count active concepts per token
            active_concepts = (concepts > 0).sum(dim=-1)
            print(f"  {block_name}: {active_concepts.mean().item():.1f} active concepts per token")
            
            # Show sparsity
            sparsity = (concepts > 0).float().mean()
            print(f"    Sparsity: {sparsity:.3f}")
    
    # Analyze concept orthogonality
    print("\nAnalyzing concept orthogonality...")
    from cbt.advanced_losses import OrthogonalityLoss
    
    orthogonality_loss = OrthogonalityLoss()
    decoder_weights = [layer.decoder.weight for layer in model.concept_layers.values()]
    ortho_loss = orthogonality_loss(decoder_weights)
    print(f"Orthogonality loss: {ortho_loss.item():.6f}")
    
    # Show concept stability
    print("\nAnalyzing concept stability...")
    from cbt.advanced_losses import StabilityLoss
    
    stability_loss = StabilityLoss()
    decoder_weights_dict = {name: layer.decoder.weight for name, layer in model.concept_layers.items()}
    stability_loss.update_anchors(decoder_weights_dict)
    stab_loss = stability_loss(decoder_weights_dict)
    print(f"Stability loss: {stab_loss.item():.6f}")
    
    print("\nAdvanced training completed!")


if __name__ == "__main__":
    main() 
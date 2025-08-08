#!/usr/bin/env python3
"""
Basic training example for Concept-Bottleneck Transformers.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import os
import sys

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
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological neurons.",
        "Transformers have revolutionized natural language processing.",
        "Deep learning models require large amounts of training data.",
        "The concept of attention mechanisms is central to modern NLP.",
        "Reinforcement learning involves learning through trial and error.",
        "Computer vision tasks include image classification and object detection.",
        "Natural language understanding is a challenging problem in AI.",
        "The transformer architecture uses self-attention mechanisms.",
        "Convolutional neural networks are effective for image processing.",
        "Recurrent neural networks can process sequential data.",
        "Generative models can create new content similar to training data.",
        "Transfer learning allows models to leverage pre-trained knowledge.",
        "The field of interpretability aims to understand model decisions.",
        "Adversarial training can improve model robustness.",
        "Regularization techniques help prevent overfitting.",
        "Optimization algorithms like Adam are commonly used in deep learning.",
        "The backpropagation algorithm computes gradients efficiently."
    ]
    return sample_texts


def main():
    """Main training function."""
    print("Setting up CBT training...")
    
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
    
    # Create trainer
    trainer = CBTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        weight_decay=0.01,
        device=device,
        use_wandb=False  # Set to True if you have wandb set up
    )
    
    # Custom alpha schedule: gradual ramp-up
    alpha_schedule = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    print("Starting training...")
    trainer.train(
        num_epochs=10,
        alpha_schedule=alpha_schedule,
        save_path="cbt_model.pt"
    )
    
    # Test the trained model
    print("\nTesting the trained model...")
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
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main() 
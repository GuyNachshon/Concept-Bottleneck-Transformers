#!/usr/bin/env python3
"""
Trained CBT Experiment Runner
Runs experiments with properly trained CBT models.
"""

import os
import sys
import json
import torch
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cbt.model import CBTModel
from cbt.training import CBTTrainer
from cbt.evaluation import CBTEvaluator, get_wikitext_eval_texts


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


def train_cbt_model(config, device, results_dir):
    """Train a CBT model with given configuration."""
    print(f"\nTraining CBT model: {config}")
    
    # Load dataset
    dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset wrapper
    train_dataset = WikiTextDataset(dataset, tokenizer, max_length=128)
    
    # Create validation dataset (use a subset)
    val_size = min(500, len(train_dataset) // 20)
    val_dataset = torch.utils.data.Subset(train_dataset, range(val_size))
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create CBT model
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=config["concept_blocks"],
        m=config["m"],
        k=config["k"],
        alpha=0.0  # Start with no concept influence
    )
    
    # Create trainer
    trainer = CBTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        weight_decay=0.01,
        device=device,
        use_wandb=False,
        use_advanced_losses=True,
        advanced_loss_config={
            "orthogonality_weight": 0.01,
            "stability_weight": 0.01,
            "kl_weight": 1.0,
            "concept_dropout_weight": 0.01
        }
    )
    
    # Training schedule (shorter for experiments)
    alpha_schedule = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    
    # Train the model
    trainer.train(
        num_epochs=len(alpha_schedule),
        alpha_schedule=alpha_schedule,
        save_path=f"{results_dir}/cbt_model_{config['name']}.pt"
    )
    
    return model, tokenizer


def run_trained_granularity_sweep(device, results_dir):
    """Run granularity sweep with trained models."""
    print("\n" + "="*60)
    print("TRAINED GRANULARITY SWEEP")
    print("="*60)
    
    # Load base model for comparison
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Get evaluation texts
    eval_texts = get_wikitext_eval_texts(num_samples=15)
    print(f"Using {len(eval_texts)} evaluation texts")
    
    # Define configurations
    configs = [
        {"name": "m32_k4", "concept_blocks": [4, 5, 6, 7], "m": 32, "k": 4},
        {"name": "m32_k8", "concept_blocks": [4, 5, 6, 7], "m": 32, "k": 8},
        {"name": "m64_k4", "concept_blocks": [4, 5, 6, 7], "m": 64, "k": 4},
        {"name": "m64_k8", "concept_blocks": [4, 5, 6, 7], "m": 64, "k": 8},
        {"name": "m128_k8", "concept_blocks": [4, 5, 6, 7], "m": 128, "k": 8},
    ]
    
    sweep_results = {}
    
    for config in configs:
        print(f"\nTraining and evaluating: {config['name']}")
        
        # Train the model
        model, tokenizer = train_cbt_model(config, device, results_dir)
        
        # Evaluate
        evaluator = CBTEvaluator(model, base_model, tokenizer, device)
        
        quality_results = evaluator.evaluate_quality(eval_texts)
        sparsity_results = evaluator.evaluate_sparsity(eval_texts)
        
        sweep_results[config["name"]] = {
            "quality": quality_results,
            "sparsity": sparsity_results,
            "config": config
        }
        
        print(f"  Quality hit: {quality_results['quality_hit_percent']:.2f}%")
        print(f"  Median active: {sparsity_results['overall_median_active_concepts']:.1f}")
    
    # Save results
    with open(f"{results_dir}/trained_granularity_sweep.json", 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    return sweep_results


def run_trained_placement_study(device, results_dir):
    """Run placement study with trained models."""
    print("\n" + "="*60)
    print("TRAINED PLACEMENT STUDY")
    print("="*60)
    
    # Load base model for comparison
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Get evaluation texts
    eval_texts = get_wikitext_eval_texts(num_samples=15)
    print(f"Using {len(eval_texts)} evaluation texts")
    
    # Define placements
    placements = [
        {"name": "early", "concept_blocks": [0, 1, 2, 3], "m": 64, "k": 8},
        {"name": "mid", "concept_blocks": [4, 5, 6, 7], "m": 64, "k": 8},
        {"name": "late", "concept_blocks": [8, 9, 10, 11], "m": 64, "k": 8},
    ]
    
    placement_results = {}
    
    for config in placements:
        print(f"\nTraining and evaluating: {config['name']} blocks")
        
        # Train the model
        model, tokenizer = train_cbt_model(config, device, results_dir)
        
        # Evaluate
        evaluator = CBTEvaluator(model, base_model, tokenizer, device)
        
        quality_results = evaluator.evaluate_quality(eval_texts)
        sparsity_results = evaluator.evaluate_sparsity(eval_texts)
        
        placement_results[config["name"]] = {
            "quality": quality_results,
            "sparsity": sparsity_results,
            "blocks": config["concept_blocks"]
        }
        
        print(f"  Quality hit: {quality_results['quality_hit_percent']:.2f}%")
        print(f"  Median active: {sparsity_results['overall_median_active_concepts']:.1f}")
    
    # Save results
    with open(f"{results_dir}/trained_placement_study.json", 'w') as f:
        json.dump(placement_results, f, indent=2)
    
    return placement_results


def main():
    """Run trained CBT experiments."""
    print("=== TRAINED CBT EXPERIMENT RUNNER ===")
    print("This will train CBT models and then evaluate them.")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"trained_cbt_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}/")
    
    try:
        # Run trained granularity sweep
        granularity_results = run_trained_granularity_sweep(device, results_dir)
        
        # Run trained placement study
        placement_results = run_trained_placement_study(device, results_dir)
        
        # Generate summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        # Find best configuration
        best_config = None
        best_score = float('inf')
        
        for config_name, config_results in granularity_results.items():
            quality_hit = config_results["quality"]["quality_hit_percent"]
            if quality_hit < best_score:
                best_score = quality_hit
                best_config = config_name
        
        summary = {
            "timestamp": timestamp,
            "device": str(device),
            "best_config": best_config,
            "best_score": best_score,
            "experiments": {
                "trained_granularity_sweep": "completed",
                "trained_placement_study": "completed"
            }
        }
        
        # Save summary
        with open(f"{results_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 TRAINED EXPERIMENTS COMPLETED!")
        print(f"📁 All results saved to: {results_dir}/")
        print(f"📊 Best configuration: {best_config}")
        print(f"📊 Best quality hit: {best_score:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Trained experiments completed successfully!")
    else:
        print("\n❌ Trained experiments failed!")
        sys.exit(1) 
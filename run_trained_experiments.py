#!/usr/bin/env python3
"""
Trained CBT Experiment Runner
Runs experiments with properly trained CBT models.
"""

import os
import sys
import json
import torch
import logging
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
LOGGER_NAME = "cbt.trained_experiments"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setLevel(logging.INFO)
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _stream_handler.setFormatter(_formatter)
    logger.addHandler(_stream_handler)
    # Do not propagate to root to avoid duplicate logs in some environments
    logger.propagate = False


def add_file_logger(results_dir: str) -> None:
    """Attach a file handler to the module logger writing into results_dir.

    Safe to call multiple times; it will avoid adding duplicate file handlers.
    """
    log_path = os.path.join(results_dir, "experiment.log")
    # Avoid duplicate file handlers pointing to the same file
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if getattr(h, 'baseFilename', None) == os.path.abspath(log_path):
                    return
            except Exception:
                pass
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)

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
    logger.info(f"Training CBT model: {config}")
    
    # Load dataset
    dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset wrapper
    train_dataset = WikiTextDataset(dataset, tokenizer, max_length=128)
    logger.info(
        f"Prepared WikiText train dataset with {len(train_dataset)} tokenized samples"
    )
    
    # Create validation dataset (use a subset)
    val_size = min(500, len(train_dataset) // 20)
    val_dataset = torch.utils.data.Subset(train_dataset, range(val_size))
    logger.info(f"Validation subset size: {len(val_dataset)}")
    
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
    logger.info("Starting training")
    trainer.train(
        num_epochs=len(alpha_schedule),
        alpha_schedule=alpha_schedule,
        save_path=f"{results_dir}/cbt_model_{config['name']}.pt"
    )
    logger.info("Finished training and saved checkpoint")
    
    return model, tokenizer


def run_trained_granularity_sweep(device, results_dir):
    """Run granularity sweep with trained models."""
    logger.info("=" * 60)
    logger.info("TRAINED GRANULARITY SWEEP")
    logger.info("=" * 60)
    
    # Load base model for comparison
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Get evaluation texts
    eval_texts = get_wikitext_eval_texts(num_samples=15)
    logger.info(f"Using {len(eval_texts)} evaluation texts")
    
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
        logger.info(f"Training and evaluating configuration: {config['name']}")
        
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
        
        logger.info(f"Quality hit: {quality_results['quality_hit_percent']:.2f}%")
        logger.info(
            f"Median active concepts: {sparsity_results['overall_median_active_concepts']:.1f}"
        )
    
    # Save results
    with open(f"{results_dir}/trained_granularity_sweep.json", 'w') as f:
        json.dump(sweep_results, f, indent=2)
    logger.info(
        f"Saved trained granularity sweep results to {results_dir}/trained_granularity_sweep.json"
    )
    
    return sweep_results


def run_trained_placement_study(device, results_dir):
    """Run placement study with trained models."""
    logger.info("=" * 60)
    logger.info("TRAINED PLACEMENT STUDY")
    logger.info("=" * 60)
    
    # Load base model for comparison
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Get evaluation texts
    eval_texts = get_wikitext_eval_texts(num_samples=15)
    logger.info(f"Using {len(eval_texts)} evaluation texts")
    
    # Define placements
    placements = [
        {"name": "early", "concept_blocks": [0, 1, 2, 3], "m": 64, "k": 8},
        {"name": "mid", "concept_blocks": [4, 5, 6, 7], "m": 64, "k": 8},
        {"name": "late", "concept_blocks": [8, 9, 10, 11], "m": 64, "k": 8},
    ]
    
    placement_results = {}
    
    for config in placements:
        logger.info(f"Training and evaluating placement: {config['name']} blocks")
        
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
        
        logger.info(f"Quality hit: {quality_results['quality_hit_percent']:.2f}%")
        logger.info(
            f"Median active concepts: {sparsity_results['overall_median_active_concepts']:.1f}"
        )
    
    # Save results
    with open(f"{results_dir}/trained_placement_study.json", 'w') as f:
        json.dump(placement_results, f, indent=2)
    logger.info(
        f"Saved trained placement study results to {results_dir}/trained_placement_study.json"
    )
    
    return placement_results


def main():
    """Run trained CBT experiments."""
    logger.info("=== TRAINED CBT EXPERIMENT RUNNER ===")
    logger.info("This will train CBT models and then evaluate them.")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"trained_cbt_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    # Start logging to file as well
    add_file_logger(results_dir)
    logger.info(f"Results will be saved to: {results_dir}/")
    
    try:
        # Run trained granularity sweep
        granularity_results = run_trained_granularity_sweep(device, results_dir)
        
        # Run trained placement study
        placement_results = run_trained_placement_study(device, results_dir)
        
        # Generate summary
        logger.info("=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        
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
        logger.info("🎉 TRAINED EXPERIMENTS COMPLETED!")
        logger.info(f"📁 All results saved to: {results_dir}/")
        logger.info(f"📊 Best configuration: {best_config}")
        logger.info(f"📊 Best quality hit: {best_score:.2f}%")
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Experiment failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Trained experiments completed successfully!")
    else:
        logger.error("❌ Trained experiments failed!")
        sys.exit(1)
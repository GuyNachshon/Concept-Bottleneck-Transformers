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
from cbt.concept_analysis import ConceptAnalyzer


# -----------------------------------------------------------------------------
# JSON serialization helpers
# -----------------------------------------------------------------------------
def _json_serializable(obj):
    try:
        import numpy as np
    except Exception:
        np = None  # optional
    import torch as _torch

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(_json_serializable(k)): _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_serializable(x) for x in obj]
    # NumPy scalar types
    if np is not None and isinstance(obj, np.generic):
        return _json_serializable(obj.item())
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        return float(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, _torch.Tensor):
        if obj.numel() == 1:
            return _json_serializable(obj.item())
        return _json_serializable(obj.detach().cpu().tolist())
    # Fallback to string
    return str(obj)


# -----------------------------------------------------------------------------
# Utilities: seeding and artifact saving
# -----------------------------------------------------------------------------
def set_global_seed(seed: int = 0):
    import random
    import numpy as np
    import os as _os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional determinism (can be slower)
    try:
        # Required by cuBLAS for deterministic GEMMs on CUDA >= 10.2
        if _os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
            _os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass


def save_concept_weights(model: CBTModel, save_dir: str, tag: str):
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)
    weights = {}
    for name, layer in model.concept_layers.items():
        weights[f"{name}.encoder.weight"] = layer.encoder.weight.detach().cpu().numpy()
        weights[f"{name}.decoder.weight"] = layer.decoder.weight.detach().cpu().numpy()
    np.savez(os.path.join(save_dir, f"concept_weights_{tag}.npz"), **weights)


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


def create_mining_dataloader(tokenizer, split: str = "validation", max_samples: int = 300, batch_size: int = 4):
    """Create a small WikiText dataloader for concept mining/labeling."""
    dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    # Limit to max_samples non-empty items
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    mining_dataset = WikiTextDataset(dataset, tokenizer, max_length=128)
    logger.info(f"Mining dataset prepared from split='{split}' with {len(mining_dataset)} samples")
    return DataLoader(
        mining_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def train_cbt_model(
    config,
    device,
    results_dir,
    *,
    learning_rate: float = 5e-5,
    use_advanced_losses: bool = True,
    advanced_loss_config: dict | None = None,
    gradient_clip_max_norm: float = 0.5,
    use_mixed_precision: bool = True,
    freeze_base_until_alpha: float = 0.5,
    alpha_schedule_override: list[float] | None = None,
):
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
        learning_rate=learning_rate,
        weight_decay=0.01,
        device=device,
        use_wandb=False,
        use_advanced_losses=use_advanced_losses,
        advanced_loss_config=(advanced_loss_config or {
            "orthogonality_weight": 0.005,
            "stability_weight": 0.005,
            "kl_weight": 0.5,
            "concept_dropout_weight": 0.005
        }),
        gradient_clip_max_norm=gradient_clip_max_norm,
        use_mixed_precision=use_mixed_precision,
        freeze_base_until_alpha=freeze_base_until_alpha,
    )
    
    # Training schedule (shorter for experiments)
    alpha_schedule = alpha_schedule_override or [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9, 1.0]
    
    # Train the model
    logger.info("Starting training")
    trainer.train(
        num_epochs=len(alpha_schedule),
        alpha_schedule=alpha_schedule,
        save_path=f"{results_dir}/cbt_model_{config['name']}.pt"
    )
    logger.info("Finished training and saved checkpoint")
    
    return model, tokenizer


def run_stabilization_run(device, results_dir):
    """Single controlled training run to stabilize concepts before sweeping."""
    logger.info("=" * 60)
    logger.info("STABILIZATION RUN (single config)")
    logger.info("=" * 60)

    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)

    eval_texts = get_wikitext_eval_texts(num_samples=200)
    logger.info(f"Using {len(eval_texts)} evaluation texts")

    config = {"name": "stab_m32_k4", "concept_blocks": [4, 5, 6, 7], "m": 32, "k": 4}

    alpha_schedule = [0.0, 0.05, 0.10, 0.15, 0.20]

    # Train with conservative settings
    model, tokenizer = train_cbt_model(
        config,
        device,
        results_dir,
        learning_rate=5e-5,
        use_advanced_losses=False,
        gradient_clip_max_norm=0.5,
        use_mixed_precision=True,
        freeze_base_until_alpha=1.0,  # freeze GPT-2; learn concept layers only
        alpha_schedule_override=alpha_schedule,
    )

    # Save concept weights
    save_concept_weights(model, results_dir, tag=config["name"])

    evaluator = CBTEvaluator(model, base_model, tokenizer, device)
    quality_results = evaluator.evaluate_quality(eval_texts)
    sparsity_results = evaluator.evaluate_sparsity(eval_texts)

    results = {
        "quality": quality_results,
        "sparsity": sparsity_results,
        "config": config,
        "alpha_schedule": alpha_schedule,
    }

    with open(f"{results_dir}/stabilization_results.json", "w") as f:
        json.dump(_json_serializable(results), f, indent=2)

    logger.info(f"Stabilization results saved to {results_dir}/stabilization_results.json")
    logger.info(f"Quality hit: {quality_results.get('quality_hit_percent', float('nan'))}")
    logger.info(f"Median active: {sparsity_results.get('overall_median_active_concepts', float('nan'))}")

    return results


def run_stabilization_run_kl(device, results_dir):
    """Second controlled run: add KL-only distillation, keep α small."""
    logger.info("=" * 60)
    logger.info("STABILIZATION RUN (KL-only)")
    logger.info("=" * 60)

    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)

    eval_texts = get_wikitext_eval_texts(num_samples=200)
    logger.info(f"Using {len(eval_texts)} evaluation texts")

    config = {"name": "stab_kl_m32_k4", "concept_blocks": [4, 5, 6, 7], "m": 32, "k": 4}

    alpha_schedule = [0.0, 0.05, 0.10, 0.15, 0.20]

    # Train with KL only
    model, tokenizer = train_cbt_model(
        config,
        device,
        results_dir,
        learning_rate=5e-5,
        use_advanced_losses=True,
        advanced_loss_config={
            "kl_weight": 0.2,
            "orthogonality_weight": 0.0,
            "stability_weight": 0.0,
            "concept_dropout_weight": 0.0,
        },
        gradient_clip_max_norm=0.5,
        use_mixed_precision=True,
        freeze_base_until_alpha=1.0,  # freeze GPT-2; learn concept layers only
        alpha_schedule_override=alpha_schedule,
    )

    save_concept_weights(model, results_dir, tag=config["name"])
    evaluator = CBTEvaluator(model, base_model, tokenizer, device)
    quality_results = evaluator.evaluate_quality(eval_texts)
    sparsity_results = evaluator.evaluate_sparsity(eval_texts)

    results = {
        "quality": quality_results,
        "sparsity": sparsity_results,
        "config": config,
        "alpha_schedule": alpha_schedule,
    }

    path = f"{results_dir}/stabilization_kl_results.json"
    with open(path, "w") as f:
        json.dump(_json_serializable(results), f, indent=2)
    logger.info(f"KL-only stabilization results saved to {path}")
    logger.info(f"Quality hit: {quality_results.get('quality_hit_percent', float('nan'))}")
    logger.info(f"Median active: {sparsity_results.get('overall_median_active_concepts', float('nan'))}")

    return results


def run_stabilization_run_kl_alpha30(device, results_dir):
    logger.info("=" * 60)
    logger.info("STABILIZATION RUN (KL-only, α→0.30)")
    logger.info("=" * 60)

    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)

    eval_texts = get_wikitext_eval_texts(num_samples=200)
    logger.info(f"Using {len(eval_texts)} evaluation texts")

    config = {"name": "stab_kl_m32_k4_a30", "concept_blocks": [4, 5, 6, 7], "m": 32, "k": 4}
    alpha_schedule = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    model, tokenizer = train_cbt_model(
        config,
        device,
        results_dir,
        learning_rate=5e-5,
        use_advanced_losses=True,
        advanced_loss_config={
            "kl_weight": 0.2,
            "orthogonality_weight": 0.0,
            "stability_weight": 0.0,
            "concept_dropout_weight": 0.0,
        },
        gradient_clip_max_norm=0.5,
        use_mixed_precision=True,
        freeze_base_until_alpha=1.0,
        alpha_schedule_override=alpha_schedule,
    )

    save_concept_weights(model, results_dir, tag=config["name"])

    evaluator = CBTEvaluator(model, base_model, tokenizer, device)
    quality_results = evaluator.evaluate_quality(eval_texts)
    sparsity_results = evaluator.evaluate_sparsity(eval_texts)

    results = {
        "quality": quality_results,
        "sparsity": sparsity_results,
        "config": config,
        "alpha_schedule": alpha_schedule,
    }

    path = f"{results_dir}/stabilization_kl_a30_results.json"
    with open(path, "w") as f:
        json.dump(_json_serializable(results), f, indent=2)
    logger.info(f"KL-only α→0.30 results saved to {path}")
    logger.info(f"Quality hit: {quality_results.get('quality_hit_percent', float('nan'))}")
    logger.info(f"Median active: {sparsity_results.get('overall_median_active_concepts', float('nan'))}")

    return results


def run_cross_seed_stability_v2(device, results_dir):
    logger.info("=" * 60)
    logger.info("CROSS-SEED RUN (KL-only, α→0.30)")
    logger.info("=" * 60)

    seeds = [0, 1, 2]
    all_results = {}
    for seed in seeds:
        set_global_seed(seed)
        logger.info(f"Training seed {seed}")
        res = run_stabilization_run_kl_alpha30(device, results_dir)
        all_results[f"seed_{seed}"] = res

    path = f"{results_dir}/cross_seed_stability_v2.json"
    with open(path, "w") as f:
        json.dump(_json_serializable(all_results), f, indent=2)
    logger.info(f"Cross-seed v2 results saved to {path}")
    return all_results


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

        # Concept mining and LLM labeling
        analysis_dir = os.path.join(results_dir, f"concept_analysis_{config['name']}")
        os.makedirs(analysis_dir, exist_ok=True)
        logger.info(f"Starting concept mining and LLM labeling → {analysis_dir}")
        analyzer = ConceptAnalyzer(
            model,
            tokenizer,
            device=str(device),
            use_llm_labeling=True,
            label_provider="auto",  # auto-pick provider from env
            label_model="gpt-4",
        )
        mining_loader = create_mining_dataloader(tokenizer, split="validation", max_samples=300, batch_size=4)
        analysis_results = analyzer.analyze_concepts(
            mining_loader,
            save_path=analysis_dir,
            max_samples=300,
        )
        # Attach brief analysis summary to results
        sweep_results[config["name"]]["analysis_summary"] = analysis_results.get("summary", {})
        logger.info("Concept analysis + labeling completed")
    
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

        # Concept mining and LLM labeling for placement study
        analysis_dir = os.path.join(results_dir, f"concept_analysis_placement_{config['name']}")
        os.makedirs(analysis_dir, exist_ok=True)
        logger.info(f"Starting concept mining and LLM labeling → {analysis_dir}")
        analyzer = ConceptAnalyzer(
            model,
            tokenizer,
            device=str(device),
            use_llm_labeling=True,
            label_provider="auto",
            label_model="gpt-4",
        )
        mining_loader = create_mining_dataloader(tokenizer, split="validation", max_samples=300, batch_size=4)
        analysis_results = analyzer.analyze_concepts(
            mining_loader,
            save_path=analysis_dir,
            max_samples=300,
        )
        placement_results[config["name"]]["analysis_summary"] = analysis_results.get("summary", {})
        logger.info("Concept analysis + labeling completed")
    
    # Save results
    with open(f"{results_dir}/trained_placement_study.json", 'w') as f:
        json.dump(placement_results, f, indent=2)
    logger.info(
        f"Saved trained placement study results to {results_dir}/trained_placement_study.json"
    )
    
    return placement_results


def evaluate_trained_model(model, device, results_dir, config_name: str, eval_size: int = 500):
    """Evaluate a trained CBT model."""
    logger.info(f"Evaluating trained model: {config_name}")
    
    # Load base model for comparison
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Get evaluation texts from WikiText TEST split (not validation to avoid contamination)
    eval_texts = get_wikitext_eval_texts(num_samples=eval_size)
    logger.info(f"Using {len(eval_texts)} evaluation texts from WikiText-2 test split")
    
    # Create evaluator
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    evaluator = CBTEvaluator(model, base_model, tokenizer, device)
    
    # Run evaluation
    quality_results = evaluator.evaluate_quality(eval_texts)
    sparsity_results = evaluator.evaluate_sparsity(eval_texts)
    
    # Check for suspicious results
    if quality_results.get("suspicious_low_perplexity", False):
        logger.warning(f"⚠️  SUSPICIOUS: Very low perplexity detected for {config_name}")
        logger.warning(f"   Base perplexity: {quality_results['base_perplexity']:.2f}")
        logger.warning(f"   CBT perplexity: {quality_results['cbt_perplexity']:.2f}")
    
    results = {
        "quality": quality_results,
        "sparsity": sparsity_results,
        "config": config_name,
        "alpha_schedule": getattr(model, "alpha_schedule", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    }
    
    # Save results
    results_path = os.path.join(results_dir, f"{config_name} Results.json")
    with open(results_path, 'w') as f:
        json.dump(_json_serializable(results), f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    return results


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
        # Seeding for reproducibility
        set_global_seed(0)
        # Run stabilization-only training; skip full sweep for now
        stab_results = run_stabilization_run(device, results_dir)
        # Run KL-only stabilization
        stab_kl_results = run_stabilization_run_kl(device, results_dir)
        # Run KL-only α→0.30 and cross-seed
        stab_kl_a30_results = run_stabilization_run_kl_alpha30(device, results_dir)
        cross_seed_results = run_cross_seed_stability_v2(device, results_dir)
        
        # Generate summary
        logger.info("=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        
        # Find best configuration
        best_config = None
        best_score = float('inf')
        
        # Choose better of the two stabilization runs
        s1 = stab_results["quality"].get("quality_hit_percent", float('inf'))
        s2 = stab_kl_results["quality"].get("quality_hit_percent", float('inf'))
        if s2 < s1:
            best_config = "stab_kl_m32_k4"
            best_score = s2
        else:
            best_config = "stab_m32_k4"
            best_score = s1
        
        summary = {
            "timestamp": timestamp,
            "device": str(device),
            "best_config": best_config,
            "best_score": best_score,
            "experiments": {
                "stabilization_run": "completed",
                "stabilization_run_kl": "completed",
                "stabilization_run_kl_alpha30": "completed",
                "cross_seed_stability_v2": "completed"
            }
        }
        
        # Save summary
        with open(f"{results_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info("🎉 TRAINED EXPERIMENTS COMPLETED!")
        logger.info(f"📁 All results saved to: {results_dir}/")
        logger.info(f"📊 Best configuration: {best_config}")
        logger.info(f"📊 Best quality hit: {best_score:.2f}%")
        
        # Evaluate the best model from stabilization runs
        logger.info("=" * 60)
        logger.info("EVALUATING BEST STABILIZATION MODEL")
        logger.info("=" * 60)
        
        # Load the best model
        best_model_path = os.path.join(results_dir, f"{best_config}_model.pt")
        if os.path.exists(best_model_path):
            best_model = CBTModel(
                base_model_name="gpt2",
                concept_blocks=stab_kl_results["concept_blocks"], # Use the best config from KL run
                m=stab_kl_results["m"],
                k=stab_kl_results["k"],
                alpha=0.3  # Use the final alpha from training
            )
            checkpoint = torch.load(best_model_path, map_location=device)
            best_model.load_state_dict(checkpoint['model_state_dict'])
            best_model.to(device)
            
            # Evaluate with larger test set
            final_results = evaluate_trained_model(
                best_model, device, results_dir, "Final_Best_Model", eval_size=1000
            )
            
            logger.info(f"Final evaluation complete:")
            logger.info(f"  Quality hit: {final_results['quality']['quality_hit_percent']:.2f}%")
            logger.info(f"  Median active concepts: {final_results['sparsity']['overall_median_active_concepts']:.1f}")
        else:
            logger.warning(f"Best model checkpoint not found at {best_model_path}")
        
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
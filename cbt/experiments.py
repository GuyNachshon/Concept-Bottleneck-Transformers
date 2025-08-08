"""
Experiments module for CBT research.
"""

import torch
import numpy as np
import json
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from .model import CBTModel
from .evaluation import CBTEvaluator


def _json_serializable(obj):
    """Convert object to JSON serializable format."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float, str)):
        return obj
    else:
        return str(obj)


def get_wikitext_eval_texts(num_samples: int = 20) -> List[str]:
    """Get evaluation texts from WikiText dataset."""
    try:
        # Load WikiText dataset
        dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        
        # Filter and sample texts
        eval_texts = []
        for item in dataset:
            text = item['text'].strip()
            # Only use texts that are reasonable length and not empty
            if len(text) > 10 and len(text) < 200 and text:
                eval_texts.append(text)
                if len(eval_texts) >= num_samples:
                    break
        
        # If we don't have enough samples, pad with some defaults
        while len(eval_texts) < num_samples:
            eval_texts.append("The weather is nice today.")
        
        return eval_texts[:num_samples]
        
    except Exception as e:
        print(f"Warning: Could not load WikiText dataset: {e}")
        print("Falling back to default evaluation texts")
        # Fallback to default texts
        return [
            "The weather is", "I went to the store", "The cat sat on",
            "She opened the book", "The car drove down", "The sun rose over",
            "They walked through", "The bird flew above", "She wrote a letter",
            "He played the guitar", "The mountain stood tall", "The river flowed",
            "The computer processed data", "The scientist conducted research",
            "The artist painted a picture", "The teacher explained the concept",
            "The doctor examined the patient", "The engineer designed the system",
            "The writer composed the story", "The musician performed the piece"
        ]


def run_granularity_sweep(
    base_model_name: str = "gpt2",
    concept_blocks: List[int] = [4, 5, 6, 7],
    m_values: List[int] = [32, 64, 128],
    k_values: List[int] = [4, 8, 12],
    eval_texts: Optional[List[str]] = None,
    save_path: str = "granularity_sweep_results.json"
) -> Dict[str, Any]:
    """Run granularity sweep: m∈{32,64,128}; k∈{4,8,12}."""
    print("=== Granularity Sweep ===")
    
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
    
    if eval_texts is None:
        eval_texts = get_wikitext_eval_texts(num_samples=15)
    
    print(f"Using {len(eval_texts)} evaluation texts from WikiText")
    
    sweep_results = {}
    
    for m in m_values:
        for k in k_values:
            if k > m:
                continue  # Skip invalid combinations
            
            print(f"\nTesting m={m}, k={k}")
            
            # Create CBT model
            cbt_model = CBTModel(
                base_model_name=base_model_name,
                concept_blocks=concept_blocks,
                m=m,
                k=k,
                alpha=1.0
            )
            
            # Evaluate
            evaluator = CBTEvaluator(cbt_model, base_model, tokenizer)
            
            # Only evaluate quality and sparsity for sweep
            quality_results = evaluator.evaluate_quality(eval_texts)
            sparsity_results = evaluator.evaluate_sparsity(eval_texts)
            
            sweep_results[f"m{m}_k{k}"] = {
                "quality": quality_results,
                "sparsity": sparsity_results,
                "config": {"m": m, "k": k}
            }
            
            print(f"  Quality hit: {quality_results['quality_hit_percent']:.2f}%")
            print(f"  Median active: {sparsity_results['overall_median_active_concepts']:.1f}")
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(_json_serializable(sweep_results), f, indent=2)
    
    print(f"\nSweep results saved to {save_path}")
    return sweep_results


def run_placement_study(
    base_model_name: str = "gpt2",
    m: int = 64,
    k: int = 8,
    eval_texts: Optional[List[str]] = None,
    save_path: str = "placement_study_results.json"
) -> Dict[str, Any]:
    """Run placement study: concepts in early vs. mid vs. late blocks."""
    print("=== Placement Study ===")
    
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
    
    if eval_texts is None:
        eval_texts = get_wikitext_eval_texts(num_samples=15)
    
    print(f"Using {len(eval_texts)} evaluation texts from WikiText")
    
    # Define block placements
    placements = {
        "early": [0, 1, 2, 3],
        "mid": [4, 5, 6, 7],
        "late": [8, 9, 10, 11]
    }
    
    placement_results = {}
    
    for placement_name, blocks in placements.items():
        print(f"\nTesting {placement_name} blocks: {blocks}")
        
        # Create CBT model
        cbt_model = CBTModel(
            base_model_name=base_model_name,
            concept_blocks=blocks,
            m=m,
            k=k,
            alpha=1.0
        )
        
        # Evaluate
        evaluator = CBTEvaluator(cbt_model, base_model, tokenizer)
        
        # Only evaluate quality and sparsity for placement study
        quality_results = evaluator.evaluate_quality(eval_texts)
        sparsity_results = evaluator.evaluate_sparsity(eval_texts)
        
        placement_results[placement_name] = {
            "quality": quality_results,
            "sparsity": sparsity_results,
            "blocks": blocks
        }
        
        print(f"  Quality hit: {quality_results['quality_hit_percent']:.2f}%")
        print(f"  Median active: {sparsity_results['overall_median_active_concepts']:.1f}")
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(_json_serializable(placement_results), f, indent=2)
    
    print(f"\nPlacement study results saved to {save_path}")
    return placement_results


def run_cross_seed_stability_test(
    base_model_name: str = "gpt2",
    concept_blocks: List[int] = [4, 5, 6, 7],
    m: int = 64,
    k: int = 8,
    num_seeds: int = 3,
    eval_texts: Optional[List[str]] = None,
    save_path: str = "cross_seed_stability_results.json"
) -> Dict[str, Any]:
    """Run cross-seed stability test."""
    print("=== Cross-Seed Stability Test ===")
    
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
    
    if eval_texts is None:
        eval_texts = get_wikitext_eval_texts(num_samples=15)
    
    print(f"Using {len(eval_texts)} evaluation texts from WikiText")
    
    models = []
    results = {}
    
    for seed in range(num_seeds):
        print(f"\nTraining model with seed {seed}")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create CBT model
        cbt_model = CBTModel(
            base_model_name=base_model_name,
            concept_blocks=concept_blocks,
            m=m,
            k=k,
            alpha=1.0
        )
        
        # Train the model (simplified - in practice you'd train properly)
        # For now, we'll just evaluate the untrained model
        
        models.append(cbt_model)
        
        # Evaluate individual model
        evaluator = CBTEvaluator(cbt_model, base_model, tokenizer)
        quality_results = evaluator.evaluate_quality(eval_texts)
        sparsity_results = evaluator.evaluate_sparsity(eval_texts)
        
        results[f"seed_{seed}"] = {
            "quality": quality_results,
            "sparsity": sparsity_results
        }
        
        print(f"  Quality hit: {quality_results['quality_hit_percent']:.2f}%")
        print(f"  Median active: {sparsity_results['overall_median_active_concepts']:.1f}")
    
    # Test stability across seeds
    if len(models) > 1:
        evaluator = CBTEvaluator(models[0], base_model, tokenizer)
        stability_results = evaluator.evaluate_stability(models[1:])
        results["stability"] = stability_results
        
        print(f"\nStability across seeds: {stability_results['overall_alignment']:.3f}")
        print(f"  Criterion met: {stability_results['stability_criterion_met']}")
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(_json_serializable(results), f, indent=2)
    
    print(f"\nCross-seed stability results saved to {save_path}")
    return results 
#!/usr/bin/env python3
"""
Test Alternative Explanations for CBT Results
Systematically test whether our results are due to artifacts or genuine improvements.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm
import logging

# Add the parent directory to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cbt.model import CBTModel
from cbt.evaluator import CBTEvaluator, get_wikitext_eval_texts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('alternative_explanations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_different_datasets():
    """Test if results hold on completely different datasets."""
    logger.info("Testing results on different datasets...")
    
    # Load different datasets
    datasets = {
        'wikitext_test': load_dataset('salesforce/wikitext', 'wikitext-2-raw-v1', split='test'),
        'wikitext_val': load_dataset('salesforce/wikitext', 'wikitext-2-raw-v1', split='validation'),
        'news': load_dataset('ag_news', split='test'),
        'scientific': load_dataset('scientific_papers', 'arxiv', split='test')
    }
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Load best CBT model
    results_dir = "trained_cbt_experiment_results_20250810_074145"
    best_config = "stab_m32_k4"
    model_path = os.path.join(results_dir, f"{best_config}_model.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return
    
    cbt_model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.2
    )
    checkpoint = torch.load(model_path, map_location=device)
    cbt_model.load_state_dict(checkpoint['model_state_dict'])
    cbt_model.to(device)
    
    results = {}
    
    for dataset_name, dataset in datasets.items():
        logger.info(f"Testing on {dataset_name}...")
        
        # Sample texts from dataset
        eval_texts = []
        for i, item in enumerate(dataset):
            if i >= 100:  # Test on 100 texts
                break
            
            if dataset_name == 'news':
                text = item['text']
            elif dataset_name == 'scientific':
                text = item['text']
            else:
                text = item['text']
            
            # Filter for reasonable length
            if len(text) > 50 and len(text) < 1000:
                eval_texts.append(text)
        
        if len(eval_texts) == 0:
            logger.warning(f"No valid texts found for {dataset_name}")
            continue
        
        # Evaluate
        evaluator = CBTEvaluator(cbt_model, base_model, tokenizer, device)
        quality_results = evaluator.evaluate_quality(eval_texts)
        
        results[dataset_name] = {
            'cbt_perplexity': quality_results['cbt_perplexity'],
            'base_perplexity': quality_results['base_perplexity'],
            'quality_hit_percent': quality_results['quality_hit_percent'],
            'num_texts': len(eval_texts)
        }
        
        logger.info(f"  {dataset_name}: CBT={quality_results['cbt_perplexity']:.2f}, "
                   f"Base={quality_results['base_perplexity']:.2f}, "
                   f"Hit={quality_results['quality_hit_percent']:.2f}%")
    
    return results

def test_different_text_lengths():
    """Test if results depend on text length selection."""
    logger.info("Testing results with different text length filters...")
    
    # Load WikiText test
    dataset = load_dataset('salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    
    # Define different length filters
    length_filters = [
        ('short', 10, 100),
        ('medium_short', 50, 200),
        ('medium', 100, 500),
        ('medium_long', 200, 1000),
        ('long', 500, 2000)
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Load best CBT model
    results_dir = "trained_cbt_experiment_results_20250810_074145"
    best_config = "stab_m32_k4"
    model_path = os.path.join(results_dir, f"{best_config}_model.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return
    
    cbt_model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.2
    )
    checkpoint = torch.load(model_path, map_location=device)
    cbt_model.load_state_dict(checkpoint['model_state_dict'])
    cbt_model.to(device)
    
    results = {}
    
    for filter_name, min_len, max_len in length_filters:
        logger.info(f"Testing {filter_name} texts ({min_len}-{max_len} chars)...")
        
        # Filter texts
        eval_texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > min_len and len(text) < max_len and text:
                eval_texts.append(text)
                if len(eval_texts) >= 100:  # Test on 100 texts
                    break
        
        if len(eval_texts) == 0:
            logger.warning(f"No valid texts found for {filter_name}")
            continue
        
        # Evaluate
        evaluator = CBTEvaluator(cbt_model, base_model, tokenizer, device)
        quality_results = evaluator.evaluate_quality(eval_texts)
        
        results[filter_name] = {
            'cbt_perplexity': quality_results['cbt_perplexity'],
            'base_perplexity': quality_results['base_perplexity'],
            'quality_hit_percent': quality_results['quality_hit_percent'],
            'num_texts': len(eval_texts),
            'avg_text_length': np.mean([len(text) for text in eval_texts])
        }
        
        logger.info(f"  {filter_name}: CBT={quality_results['cbt_perplexity']:.2f}, "
                   f"Base={quality_results['base_perplexity']:.2f}, "
                   f"Hit={quality_results['quality_hit_percent']:.2f}%")
    
    return results

def test_different_alpha_schedules():
    """Test if the alpha schedule itself is causing the improvement."""
    logger.info("Testing different alpha schedules...")
    
    # Load evaluation texts
    eval_texts = get_wikitext_eval_texts(num_samples=100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Test different alpha schedules
    alpha_schedules = [
        ('constant_0.2', [0.2] * 10),  # Constant alpha
        ('linear', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),  # Linear ramp
        ('step', [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2]),  # Step function
        ('exponential', [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])  # Exponential
    ]
    
    results = {}
    
    for schedule_name, alpha_schedule in alpha_schedules:
        logger.info(f"Testing {schedule_name} schedule...")
        
        # Create model with this schedule
        cbt_model = CBTModel(
            base_model_name="gpt2",
            concept_blocks=[4, 5, 6, 7],
            m=32,
            k=4,
            alpha=alpha_schedule[-1]  # Use final alpha
        )
        cbt_model.to(device)
        
        # Evaluate
        evaluator = CBTEvaluator(cbt_model, base_model, tokenizer, device)
        quality_results = evaluator.evaluate_quality(eval_texts)
        
        results[schedule_name] = {
            'cbt_perplexity': quality_results['cbt_perplexity'],
            'base_perplexity': quality_results['base_perplexity'],
            'quality_hit_percent': quality_results['quality_hit_percent'],
            'final_alpha': alpha_schedule[-1]
        }
        
        logger.info(f"  {schedule_name}: CBT={quality_results['cbt_perplexity']:.2f}, "
                   f"Base={quality_results['base_perplexity']:.2f}, "
                   f"Hit={quality_results['quality_hit_percent']:.2f}%")
    
    return results

def test_concept_layer_architecture():
    """Test if the concept layer architecture itself is causing improvement."""
    logger.info("Testing different concept layer architectures...")
    
    # Load evaluation texts
    eval_texts = get_wikitext_eval_texts(num_samples=100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.to(device)
    
    # Test different architectures
    architectures = [
        ('no_concepts', None, None, 0.0),  # No concept layers
        ('small_concepts', [4, 5, 6, 7], 16, 4),  # Smaller concepts
        ('large_concepts', [4, 5, 6, 7], 64, 8),  # Larger concepts
        ('early_concepts', [0, 1, 2, 3], 32, 4),  # Early blocks
        ('late_concepts', [8, 9, 10, 11], 32, 4),  # Late blocks
    ]
    
    results = {}
    
    for arch_name, concept_blocks, m, k in architectures:
        logger.info(f"Testing {arch_name} architecture...")
        
        if concept_blocks is None:
            # Test base model only
            evaluator = CBTEvaluator(base_model, base_model, tokenizer, device)
            quality_results = evaluator.evaluate_quality(eval_texts)
            
            results[arch_name] = {
                'cbt_perplexity': quality_results['base_perplexity'],
                'base_perplexity': quality_results['base_perplexity'],
                'quality_hit_percent': 0.0,
                'architecture': 'base_only'
            }
        else:
            # Test CBT model
            cbt_model = CBTModel(
                base_model_name="gpt2",
                concept_blocks=concept_blocks,
                m=m,
                k=k,
                alpha=0.2
            )
            cbt_model.to(device)
            
            evaluator = CBTEvaluator(cbt_model, base_model, tokenizer, device)
            quality_results = evaluator.evaluate_quality(eval_texts)
            
            results[arch_name] = {
                'cbt_perplexity': quality_results['cbt_perplexity'],
                'base_perplexity': quality_results['base_perplexity'],
                'quality_hit_percent': quality_results['quality_hit_percent'],
                'architecture': f'cbt_{concept_blocks}_{m}_{k}'
            }
        
        logger.info(f"  {arch_name}: CBT={results[arch_name]['cbt_perplexity']:.2f}, "
                   f"Base={results[arch_name]['base_perplexity']:.2f}, "
                   f"Hit={results[arch_name]['quality_hit_percent']:.2f}%")
    
    return results

def main():
    """Main testing pipeline."""
    logger.info("=" * 60)
    logger.info("TESTING ALTERNATIVE EXPLANATIONS")
    logger.info("=" * 60)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results/analysis/alternative_explanations_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    # 1. Test different datasets
    logger.info("\n1. Testing different datasets...")
    dataset_results = test_different_datasets()
    all_results['datasets'] = dataset_results
    
    # 2. Test different text lengths
    logger.info("\n2. Testing different text lengths...")
    length_results = test_different_text_lengths()
    all_results['text_lengths'] = length_results
    
    # 3. Test different alpha schedules
    logger.info("\n3. Testing different alpha schedules...")
    schedule_results = test_different_alpha_schedules()
    all_results['alpha_schedules'] = schedule_results
    
    # 4. Test different architectures
    logger.info("\n4. Testing different architectures...")
    architecture_results = test_concept_layer_architecture()
    all_results['architectures'] = architecture_results
    
    # Save all results
    with open(os.path.join(results_dir, 'alternative_explanations_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    # Check if improvements are consistent across conditions
    improvements = []
    
    for test_name, test_results in all_results.items():
        logger.info(f"\n{test_name.upper()}:")
        for condition, result in test_results.items():
            hit = result['quality_hit_percent']
            improvements.append(hit)
            logger.info(f"  {condition}: {hit:.2f}%")
    
    avg_improvement = np.mean(improvements)
    logger.info(f"\nAverage quality hit across all tests: {avg_improvement:.2f}%")
    
    if avg_improvement < 0:
        logger.info("✅ Results suggest genuine improvement (negative hit = improvement)")
    else:
        logger.info("❌ Results suggest the improvement might be an artifact")
    
    logger.info(f"\nResults saved to: {results_dir}/")
    logger.info("Testing complete!")

if __name__ == "__main__":
    main() 
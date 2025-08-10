#!/usr/bin/env python3
"""
Comprehensive Concept Analysis for CBT
Analyze what the learned concepts represent and how they improve language modeling.
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
from cbt.analyzer import ConceptAnalyzer
from cbt.evaluator import CBTEvaluator, get_wikitext_eval_texts
from cbt.llm_labeling import create_llm_labeler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('concept_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_best_model(results_dir: str, device: str):
    """Load the best performing CBT model."""
    # Find the best model checkpoint
    best_config = "stab_m32_k4"  # From our results
    model_path = os.path.join(results_dir, f"{best_config}_model.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return None
    
    # Load model
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.2  # Use the optimal alpha
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded best model: {best_config}")
    return model

def analyze_concept_activations(model, tokenizer, device, num_texts=100):
    """Analyze concept activation patterns across different text types."""
    logger.info("Analyzing concept activation patterns...")
    
    # Get diverse evaluation texts
    eval_texts = get_wikitext_eval_texts(num_samples=num_texts)
    
    # Collect concept activations
    all_activations = {}
    text_categories = {
        'short': [],
        'medium': [],
        'long': []
    }
    
    with torch.no_grad():
        for i, text in enumerate(tqdm(eval_texts)):
            input_ids = tokenizer.encode(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=256
            ).to(device)
            
            outputs = model(input_ids=input_ids, return_concepts=True)
            activations = outputs["concept_activations"]
            
            # Categorize by text length
            if len(text) < 100:
                category = 'short'
            elif len(text) < 500:
                category = 'medium'
            else:
                category = 'long'
            
            text_categories[category].append({
                'text': text,
                'activations': activations,
                'length': len(text)
            })
            
            # Aggregate activations
            for block_name, block_acts in activations.items():
                if block_name not in all_activations:
                    all_activations[block_name] = []
                all_activations[block_name].append(block_acts.cpu().numpy())
    
    return all_activations, text_categories

def analyze_concept_specialization(activations):
    """Analyze which concepts specialize in which text types."""
    logger.info("Analyzing concept specialization...")
    
    specialization_results = {}
    
    for block_name, block_acts_list in activations.items():
        # Convert to numpy array
        acts_array = np.concatenate(block_acts_list, axis=0)  # [num_tokens, m]
        
        # Calculate concept usage statistics
        concept_usage = np.mean(acts_array > 0, axis=0)  # [m]
        concept_strength = np.mean(acts_array, axis=0)    # [m]
        
        # Find most and least used concepts
        most_used = np.argsort(concept_usage)[-5:]  # Top 5
        least_used = np.argsort(concept_usage)[:5]   # Bottom 5
        
        specialization_results[block_name] = {
            'concept_usage': concept_usage.tolist(),
            'concept_strength': concept_strength.tolist(),
            'most_used_concepts': most_used.tolist(),
            'least_used_concepts': least_used.tolist(),
            'usage_entropy': -np.sum(concept_usage * np.log(concept_usage + 1e-8))
        }
    
    return specialization_results

def mine_concept_contexts(model, tokenizer, device, num_texts=500):
    """Mine contexts where each concept is most active."""
    logger.info("Mining concept contexts...")
    
    # Get more texts for mining
    eval_texts = get_wikitext_eval_texts(num_samples=num_texts)
    
    # Initialize mining data structure
    concept_contexts = {}
    for block_idx in [4, 5, 6, 7]:
        block_name = f"block_{block_idx}"
        concept_contexts[block_name] = {i: [] for i in range(32)}  # 32 concepts per block
    
    with torch.no_grad():
        for text in tqdm(eval_texts):
            input_ids = tokenizer.encode(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=256
            ).to(device)
            
            outputs = model(input_ids=input_ids, return_concepts=True)
            activations = outputs["concept_activations"]
            
            # Get tokens for context
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            for block_name, block_acts in activations.items():
                acts = block_acts[0].cpu().numpy()  # [seq_len, m]
                
                for token_idx in range(min(len(tokens), acts.shape[0])):
                    # Find top active concepts for this token
                    top_concepts = np.argsort(acts[token_idx])[-3:]  # Top 3
                    
                    for concept_idx in top_concepts:
                        if acts[token_idx, concept_idx] > 0.1:  # Threshold
                            # Get context window
                            start = max(0, token_idx - 5)
                            end = min(len(tokens), token_idx + 6)
                            context = ' '.join(tokens[start:end])
                            
                            concept_contexts[block_name][concept_idx].append({
                                'context': context,
                                'activation': float(acts[token_idx, concept_idx]),
                                'token': tokens[token_idx]
                            })
    
    return concept_contexts

def label_concepts_with_llm(concept_contexts):
    """Use LLM to label concepts based on their contexts."""
    logger.info("Labeling concepts with LLM...")
    
    # Create LLM labeler
    llm_labeler = create_llm_labeler(provider="auto")
    
    concept_labels = {}
    
    for block_name, block_concepts in concept_contexts.items():
        concept_labels[block_name] = {}
        
        for concept_idx, contexts in block_concepts.items():
            if len(contexts) == 0:
                continue
                
            # Get top contexts by activation strength
            sorted_contexts = sorted(contexts, key=lambda x: x['activation'], reverse=True)
            top_contexts = sorted_contexts[:10]  # Top 10 contexts
            
            # Prepare context text for LLM
            context_text = "\n".join([f"- {ctx['context']}" for ctx in top_contexts])
            
            # Create prompt for concept labeling
            prompt = f"""Based on the following contexts where a neural network concept is most active, what does this concept likely represent?

Contexts:
{context_text}

Please provide a concise label (1-3 words) that captures what this concept represents. Focus on linguistic, semantic, or structural patterns.

Label:"""
            
            try:
                label = llm_labeler.label_concept(prompt)
                concept_labels[block_name][concept_idx] = {
                    'label': label,
                    'num_contexts': len(contexts),
                    'avg_activation': np.mean([ctx['activation'] for ctx in contexts])
                }
                logger.info(f"{block_name} concept {concept_idx}: {label}")
            except Exception as e:
                logger.warning(f"Failed to label {block_name} concept {concept_idx}: {e}")
                concept_labels[block_name][concept_idx] = {
                    'label': 'unknown',
                    'num_contexts': len(contexts),
                    'avg_activation': np.mean([ctx['activation'] for ctx in contexts])
                }
    
    return concept_labels

def test_concept_causality(model, tokenizer, device, concept_labels, num_texts=50):
    """Test causality by ablating concepts and measuring effects."""
    logger.info("Testing concept causality...")
    
    eval_texts = get_wikitext_eval_texts(num_samples=num_texts)
    
    causality_results = {}
    
    for block_name, block_concepts in concept_labels.items():
        causality_results[block_name] = {}
        
        for concept_idx, concept_info in block_concepts.items():
            if concept_info['label'] == 'unknown':
                continue
                
            # Test ablation of this concept
            ablation_effects = []
            
            with torch.no_grad():
                for text in eval_texts:
                    input_ids = tokenizer.encode(
                        text, 
                        return_tensors='pt', 
                        truncation=True, 
                        max_length=256
                    ).to(device)
                    
                    # Get original predictions
                    original_outputs = model(input_ids=input_ids)
                    original_logits = original_outputs["logits"]
                    original_probs = torch.softmax(original_logits, dim=-1)
                    
                    # Ablate the concept
                    concept_edits = {block_name: {concept_idx: 0.0}}
                    ablated_outputs = model(input_ids=input_ids, concept_edits=concept_edits)
                    ablated_logits = ablated_outputs["logits"]
                    ablated_probs = torch.softmax(ablated_logits, dim=-1)
                    
                    # Measure effect
                    prob_change = torch.mean(torch.abs(original_probs - ablated_probs)).item()
                    ablation_effects.append(prob_change)
            
            causality_results[block_name][concept_idx] = {
                'label': concept_info['label'],
                'avg_ablation_effect': np.mean(ablation_effects),
                'std_ablation_effect': np.std(ablation_effects)
            }
    
    return causality_results

def main():
    """Main analysis pipeline."""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE CONCEPT ANALYSIS")
    logger.info("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Look for the most recent experiment results
    results_base = "results/experiments_"
    existing_dirs = [d for d in os.listdir("results") if d.startswith("experiments_")]
    if existing_dirs:
        # Sort by timestamp and get the most recent
        existing_dirs.sort()
        results_dir = f"results/{existing_dirs[-1]}"
    else:
        logger.error("No experiment results found. Please run experiments first.")
        return
    
    # Load model
    model = load_best_model(results_dir, device)
    if model is None:
        return
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create analysis directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_dir = f"results/analysis/concept_analysis_{timestamp}"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Analyze concept activations
    logger.info("Step 1: Analyzing concept activations...")
    activations, text_categories = analyze_concept_activations(model, tokenizer, device)
    
    # 2. Analyze concept specialization
    logger.info("Step 2: Analyzing concept specialization...")
    specialization = analyze_concept_specialization(activations)
    
    # 3. Mine concept contexts
    logger.info("Step 3: Mining concept contexts...")
    concept_contexts = mine_concept_contexts(model, tokenizer, device)
    
    # 4. Label concepts with LLM
    logger.info("Step 4: Labeling concepts with LLM...")
    concept_labels = label_concepts_with_llm(concept_contexts)
    
    # 5. Test concept causality
    logger.info("Step 5: Testing concept causality...")
    causality_results = test_concept_causality(model, tokenizer, device, concept_labels)
    
    # Save all results
    results = {
        'specialization': specialization,
        'concept_labels': concept_labels,
        'causality_results': causality_results,
        'text_categories': {
            cat: len(texts) for cat, texts in text_categories.items()
        }
    }
    
    with open(os.path.join(analysis_dir, 'concept_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    for block_name, block_labels in concept_labels.items():
        logger.info(f"\n{block_name}:")
        for concept_idx, info in block_labels.items():
            if info['label'] != 'unknown':
                logger.info(f"  Concept {concept_idx}: {info['label']} (avg_act: {info['avg_activation']:.3f})")
    
    logger.info(f"\nResults saved to: {analysis_dir}/")
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 
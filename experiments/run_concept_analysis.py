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

def find_latest_results_dir():
    """Find the most recent experiment results directory."""
    # Look for directories that match our naming pattern
    import glob
    
    # Pattern for experiment result directories
    patterns = [
        "trained_cbt_experiment_results_*",
        "cbt_experiment_results_*",
        "experiment_results_*"
    ]
    
    all_dirs = []
    for pattern in patterns:
        all_dirs.extend(glob.glob(pattern))
    
    if not all_dirs:
        logger.error("No experiment results directories found!")
        logger.info("Looking for directories matching:")
        for pattern in patterns:
            logger.info(f"  - {pattern}")
        return None
    
    # Sort by creation time (newest first)
    all_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    latest_dir = all_dirs[0]
    logger.info(f"Found {len(all_dirs)} experiment directories:")
    for i, dir_path in enumerate(all_dirs[:5]):  # Show top 5
        ctime = os.path.getctime(dir_path)
        ctime_str = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
        marker = " (LATEST)" if i == 0 else ""
        logger.info(f"  {i+1}. {dir_path} - {ctime_str}{marker}")
    
    return latest_dir

def list_available_models(results_dir: str):
    """List all available models in a results directory."""
    model_files = [f for f in os.listdir(results_dir) if f.endswith('.pt') and f.startswith('cbt_model_')]
    
    if not model_files:
        logger.error(f"No model files found in {results_dir}")
        return []
    
    logger.info(f"Available models in {results_dir}:")
    for i, model_file in enumerate(model_files):
        # Try to extract model info from filename
        info = extract_model_info(model_file)
        logger.info(f"  {i+1}. {model_file}")
        logger.info(f"      Config: {info}")
    
    return model_files

def extract_model_info(model_file: str):
    """Extract model configuration from filename."""
    info = {}
    
    # Extract alpha
    if "a30" in model_file:
        info['alpha'] = 0.3
    elif "a20" in model_file:
        info['alpha'] = 0.2
    elif "a10" in model_file:
        info['alpha'] = 0.1
    else:
        info['alpha'] = 0.2  # default
    
    # Extract model type
    if "kl" in model_file:
        info['type'] = "KL (with distillation)"
    elif "stab" in model_file:
        info['type'] = "Stabilized"
    else:
        info['type'] = "Standard"
    
    # Extract cross-seed info
    if "cross_seed" in model_file:
        info['cross_seed'] = True
    else:
        info['cross_seed'] = False
    
    return info

def load_model_from_file(results_dir: str, model_file: str, device: str):
    """Load a specific model file."""
    model_path = os.path.join(results_dir, model_file)
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    # Extract model info
    model_info = extract_model_info(model_file)
    alpha = model_info['alpha']
    
    logger.info(f"Loading model: {model_file}")
    logger.info(f"Model info: {model_info}")
    
    # Try different model configurations based on filename
    if "m64" in model_file:
        m, k = 64, 8
    elif "m128" in model_file:
        m, k = 128, 12
    else:
        m, k = 32, 4  # default
    
    # Try different concept block configurations
    if "early" in model_file:
        concept_blocks = [0, 1, 2, 3]
    elif "late" in model_file:
        concept_blocks = [8, 9, 10, 11]
    else:
        concept_blocks = [4, 5, 6, 7]  # default
    
    # Load model
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=concept_blocks,
        m=m,
        k=k,
        alpha=alpha
    )
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Store metadata
        model._loaded_from = model_file
        model._model_info = model_info
        
        logger.info(f"Successfully loaded model: {model_file}")
        logger.info(f"Configuration: m={m}, k={k}, alpha={alpha}, blocks={concept_blocks}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_file}: {e}")
        return None

def load_best_model(results_dir: str, device: str, model_file: str = None):
    """Load the best performing CBT model or a specific model."""
    if model_file:
        # Load specific model
        return load_model_from_file(results_dir, model_file, device)
    
    # Look for available model files
    model_files = [f for f in os.listdir(results_dir) if f.endswith('.pt') and f.startswith('cbt_model_')]
    
    if not model_files:
        logger.error(f"No model files found in {results_dir}")
        logger.info("Available files:")
        for f in os.listdir(results_dir):
            logger.info(f"  - {f}")
        return None
    
    # Prefer the KL model with alpha 0.30 as it's likely the best performing
    preferred_models = [
        "cbt_model_stab_kl_m32_k4_a30.pt",  # KL model with alpha 0.30
        "cbt_model_stab_kl_m32_k4.pt",      # KL model
        "cbt_model_stab_m32_k4.pt",         # Basic model
        "cbt_model_stab_kl_m32_k4_cross_seed.pt",  # Cross-seed model
    ]
    
    model_file = None
    for preferred in preferred_models:
        if preferred in model_files:
            model_file = preferred
            break
    
    if model_file is None:
        # Fall back to the first available model
        model_file = model_files[0]
        logger.warning(f"No preferred model found, using: {model_file}")
    
    return load_model_from_file(results_dir, model_file, device)

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
    logger.info(f"Using device: {device}")
    
    # Find the latest results directory
    results_dir = find_latest_results_dir()
    if results_dir is None:
        return
    
    logger.info(f"Using results directory: {results_dir}")
    
    # List available models
    available_models = list_available_models(results_dir)
    
    # Check if user wants to analyze a specific model
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith('.pt'):
        model_file = sys.argv[1]
        logger.info(f"User specified model: {model_file}")
    else:
        model_file = None
        logger.info("Using best available model")
    
    # Load model
    model = load_best_model(results_dir, device, model_file)
    if model is None:
        return
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create analysis directory in organized structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = getattr(model, '_loaded_from', 'unknown').replace('.pt', '')
    
    # Create organized directory structure
    analysis_base = "results/analysis/concept_analysis"
    os.makedirs(analysis_base, exist_ok=True)
    
    analysis_dir = os.path.join(analysis_base, f"{model_name}_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Also create subdirectories for different types of results
    os.makedirs(os.path.join(analysis_dir, "contexts"), exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "causality"), exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "specialization"), exist_ok=True)
    
    # Save analysis metadata
    analysis_metadata = {
        'results_dir': results_dir,
        'model_file': getattr(model, '_loaded_from', 'unknown'),
        'model_info': getattr(model, '_model_info', {}),
        'device': str(device),
        'timestamp': timestamp,
        'model_config': {
            'base_model_name': 'gpt2',
            'concept_blocks': model.concept_blocks,
            'm': model.m,
            'k': model.k,
            'alpha': model.alpha
        }
    }
    
    with open(os.path.join(analysis_dir, 'analysis_metadata.json'), 'w') as f:
        json.dump(analysis_metadata, f, indent=2)
    
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
        'metadata': analysis_metadata,
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
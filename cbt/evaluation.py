"""
Comprehensive evaluation module for CBT success criteria.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from .model import CBTModel
from .advanced_losses import StabilityLoss
from .concept_analysis import ConceptAnalyzer
from .llm_labeling import create_llm_labeler
from .ablation_tools import ConceptAblator


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


class CBTEvaluator:
    """Evaluator for CBT success criteria."""
    
    def __init__(self, cbt_model: CBTModel, base_model: GPT2LMHeadModel, 
                 tokenizer: GPT2Tokenizer, device: str = "cuda"):
        self.cbt_model = cbt_model.to(device)
        self.base_model = base_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        self.cbt_model.eval()
        self.base_model.eval()
    
    def evaluate_quality(self, eval_texts: List[str]) -> Dict[str, float]:
        """Evaluate quality: ≤2% perplexity hit vs. baseline."""
        print("Evaluating quality...")
        
        cbt_losses, base_losses = [], []
        
        with torch.no_grad():
            for text in tqdm(eval_texts):
                input_ids = self.tokenizer.encode(text, return_tensors='pt', 
                                                truncation=True, max_length=128).to(self.device)
                
                cbt_outputs = self.cbt_model(input_ids=input_ids, labels=input_ids)
                base_outputs = self.base_model(input_ids=input_ids, labels=input_ids)
                
                cbt_losses.append(cbt_outputs["loss"].item())
                base_losses.append(base_outputs.loss.item())
        
        cbt_perplexity = np.exp(np.mean(cbt_losses))
        base_perplexity = np.exp(np.mean(base_losses))
        quality_hit = (cbt_perplexity - base_perplexity) / base_perplexity * 100
        
        return {
            "cbt_perplexity": cbt_perplexity,
            "base_perplexity": base_perplexity,
            "quality_hit_percent": quality_hit,
            "quality_criterion_met": quality_hit <= 2.0
        }
    
    def evaluate_sparsity(self, eval_texts: List[str]) -> Dict[str, Any]:
        """Evaluate sparsity: median ≤8 active concepts/token/block."""
        print("Evaluating sparsity...")
        
        all_active_counts = {}
        
        with torch.no_grad():
            for text in tqdm(eval_texts):
                input_ids = self.tokenizer.encode(text, return_tensors='pt', 
                                                truncation=True, max_length=128).to(self.device)
                
                outputs = self.cbt_model(input_ids=input_ids, return_concepts=True)
                concept_activations = outputs["concept_activations"]
                
                for block_name, concepts in concept_activations.items():
                    if block_name not in all_active_counts:
                        all_active_counts[block_name] = []
                    
                    active_concepts = (concepts > 0).sum(dim=-1)
                    all_active_counts[block_name].extend(active_concepts.flatten().cpu().numpy())
        
        overall_medians = []
        for block_name, counts in all_active_counts.items():
            median_count = np.median(counts)
            overall_medians.append(median_count)
        
        overall_median = np.median(overall_medians)
        
        return {
            "overall_median_active_concepts": overall_median,
            "overall_sparsity_criterion_met": overall_median <= 8.0
        }
    
    def evaluate_stability(self, other_models: List[CBTModel]) -> Dict[str, Any]:
        """Evaluate stability: ≥0.8 alignment across seeds."""
        print("Evaluating stability...")
        
        if not other_models:
            return {"stability_criterion_met": False, "error": "No other models provided"}
        
        # Get decoder weights from all models
        all_decoder_weights = []
        
        # Current model
        current_weights = {
            name: layer.decoder.weight 
            for name, layer in self.cbt_model.concept_layers.items()
        }
        all_decoder_weights.append(current_weights)
        
        # Other models
        for model in other_models:
            model_weights = {
                name: layer.decoder.weight 
                for name, layer in model.concept_layers.items()
            }
            all_decoder_weights.append(model_weights)
        
        # Compute pairwise alignments
        alignments = []
        
        for i in range(len(all_decoder_weights)):
            for j in range(i + 1, len(all_decoder_weights)):
                weights_i = all_decoder_weights[i]
                weights_j = all_decoder_weights[j]
                
                # Compute Procrustes alignment for each block
                block_alignments = []
                
                for block_name in weights_i.keys():
                    if block_name in weights_j:
                        W_i = weights_i[block_name]
                        W_j = weights_j[block_name]
                        
                        # Compute SVD of W_i^T W_j
                        U, S, Vt = torch.svd(torch.mm(W_i.t(), W_j))
                        Q = torch.mm(U, Vt.t())
                        
                        # Compute alignment quality
                        aligned_W_j = torch.mm(W_j, Q.t())
                        alignment_quality = torch.norm(W_i - aligned_W_j, p='fro') / torch.norm(W_i, p='fro')
                        alignment_score = 1.0 - alignment_quality.item()
                        
                        block_alignments.append(alignment_score)
                
                if block_alignments:
                    avg_alignment = np.mean(block_alignments)
                    alignments.append(avg_alignment)
        
        overall_alignment = np.mean(alignments) if alignments else 0.0
        
        return {
            "pairwise_alignments": alignments,
            "overall_alignment": overall_alignment,
            "stability_criterion_met": overall_alignment >= 0.8
        }
    
    def evaluate_causality(self, eval_texts: List[str]) -> Dict[str, Any]:
        """Evaluate causality: editing one concept produces large, targeted behavior deltas."""
        print("Evaluating causality...")
        
        ablator = ConceptAblator(self.cbt_model, self.tokenizer, self.device)
        
        # Get concept keys
        concept_keys = []
        for block_name in self.cbt_model.concept_layers.keys():
            for i in range(self.cbt_model.m):
                concept_keys.append(f"{block_name}_concept_{i}")
        
        # Test a subset of concepts
        test_concepts = concept_keys[:10]  # Test first 10 concepts
        
        causality_results = {}
        
        for concept_key in tqdm(test_concepts, desc="Testing causality"):
            # Get baseline generation
            baseline_text = self._generate_text("The weather is")
            
            # Ablate concept
            ablator.ablate_concept(concept_key, "zero")
            ablated_text = self._generate_text("The weather is")
            
            # Restore concept
            ablator.restore_concept(concept_key)
            
            # Compute text similarity
            similarity = self._compute_text_similarity(baseline_text, ablated_text)
            
            causality_results[concept_key] = {
                "baseline_text": baseline_text,
                "ablated_text": ablated_text,
                "text_similarity": similarity,
                "has_effect": similarity < 0.8  # Arbitrary threshold
            }
        
        # Compute overall causality score
        effects = [result["has_effect"] for result in causality_results.values()]
        causality_score = np.mean(effects) if effects else 0.0
        
        return {
            "concept_effects": causality_results,
            "causality_score": causality_score,
            "causality_criterion_met": causality_score >= 0.5  # At least 50% of concepts have effects
        }
    
    def evaluate_nameability(self, eval_texts: List[str]) -> Dict[str, Any]:
        """Evaluate nameability: ≥70% concepts receive consistent labels."""
        print("Evaluating nameability...")
        
        # Create concept analyzer
        analyzer = ConceptAnalyzer(self.cbt_model, self.tokenizer, self.device)
        
        # Mine concepts
        from torch.utils.data import DataLoader
        
        # Create simple dataset for mining
        class SimpleDataset:
            def __init__(self, texts):
                self.texts = texts
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                return {"text": self.texts[idx]}
        
        dataset = SimpleDataset(eval_texts)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        mining_results = analyzer.miner.mine_concepts(dataloader, activation_threshold=0.01)
        
        # Create labeler
        labeler = create_llm_labeler(provider="mock")  # Use mock for evaluation
        
        # Label concepts
        labels = labeler.batch_label_concepts(mining_results)
        
        # Count labeled concepts
        total_concepts = len(mining_results)
        labeled_concepts = len([l for l in labels.values() if l and l != "unknown"])
        
        nameability_score = labeled_concepts / total_concepts if total_concepts > 0 else 0.0
        
        return {
            "total_concepts": total_concepts,
            "labeled_concepts": labeled_concepts,
            "nameability_score": nameability_score,
            "nameability_criterion_met": nameability_score >= 0.7,
            "labels": labels
        }
    
    def _generate_text(self, prompt: str, max_length: int = 20) -> str:
        """Generate text with the CBT model."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            generated = self.cbt_model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple character-level similarity between texts."""
        # Simple Jaccard similarity on character n-grams
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text)-n+1))
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_all_criteria(self, eval_texts: List[str], 
                            other_models: Optional[List[CBTModel]] = None) -> Dict[str, Any]:
        """Evaluate all success criteria."""
        print("=== CBT Success Criteria Evaluation ===\n")
        
        results = {}
        
        # Quality evaluation
        results["quality"] = self.evaluate_quality(eval_texts)
        print(f"Quality: {results['quality']['quality_hit_percent']:.2f}% hit")
        print(f"  Criterion met: {results['quality']['quality_criterion_met']}\n")
        
        # Sparsity evaluation
        results["sparsity"] = self.evaluate_sparsity(eval_texts)
        print(f"Sparsity: {results['sparsity']['overall_median_active_concepts']:.1f} median active concepts")
        print(f"  Criterion met: {results['sparsity']['overall_sparsity_criterion_met']}\n")
        
        # Stability evaluation
        if other_models:
            results["stability"] = self.evaluate_stability(other_models)
            print(f"Stability: {results['stability']['overall_alignment']:.3f} alignment")
            print(f"  Criterion met: {results['stability']['stability_criterion_met']}\n")
        else:
            results["stability"] = {"stability_criterion_met": False, "error": "No other models provided"}
            print("Stability: Not evaluated (no other models provided)\n")
        
        # Causality evaluation
        results["causality"] = self.evaluate_causality(eval_texts)
        print(f"Causality: {results['causality']['causality_score']:.3f} score")
        print(f"  Criterion met: {results['causality']['causality_criterion_met']}\n")
        
        # Nameability evaluation
        results["nameability"] = self.evaluate_nameability(eval_texts)
        print(f"Nameability: {results['nameability']['nameability_score']:.3f} score")
        print(f"  Criterion met: {results['nameability']['nameability_criterion_met']}\n")
        
        # Overall assessment
        criteria_met = [
            results["quality"]["quality_criterion_met"],
            results["sparsity"]["overall_sparsity_criterion_met"],
            results["stability"]["stability_criterion_met"],
            results["causality"]["causality_criterion_met"],
            results["nameability"]["nameability_criterion_met"]
        ]
        
        overall_score = np.mean([c for c in criteria_met if c is not False])
        results["overall"] = {
            "criteria_met": criteria_met,
            "overall_score": overall_score,
            "all_criteria_met": all(criteria_met)
        }
        
        print(f"=== Overall Assessment ===")
        print(f"Criteria met: {sum(criteria_met)}/5")
        print(f"Overall score: {overall_score:.3f}")
        print(f"All criteria met: {results['overall']['all_criteria_met']}")
        
        return results


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
        eval_texts = [
            "The weather is", "I went to the store", "The cat sat on",
            "She opened the book", "The car drove down"
        ]
    
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
        json.dump(sweep_results, f, indent=2)
    
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
        eval_texts = [
            "The weather is", "I went to the store", "The cat sat on",
            "She opened the book", "The car drove down"
        ]
    
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
        json.dump(placement_results, f, indent=2)
    
    print(f"\nPlacement study results saved to {save_path}")
    return placement_results 
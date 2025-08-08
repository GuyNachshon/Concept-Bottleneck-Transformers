"""
Ablation testing and concept editing tools for CBT.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os


class ConceptAblator:
    """
    Tools for ablating (turning off) specific concepts and measuring effects.
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.original_concept_layers = {}
        self.ablation_results = {}
    
    def ablate_concept(self, concept_key: str, ablation_type: str = "zero") -> None:
        """
        Ablate a specific concept by modifying its activations.
        
        Args:
            concept_key: Concept to ablate (e.g., "block_4_concept_0")
            ablation_type: Type of ablation ("zero", "random", "mean", "noise")
        """
        try:
            # Parse concept key
            block_name, concept_idx = self._parse_concept_key(concept_key)
            concept_idx = int(concept_idx)
            
            # Get the concept layer
            concept_layer = self.model.concept_layers[block_name]
            
            # Store original state if not already stored
            if concept_key not in self.original_concept_layers:
                self.original_concept_layers[concept_key] = {
                    "encoder_weight": concept_layer.encoder.weight[concept_idx].clone().detach(),
                    "encoder_bias": concept_layer.encoder.bias[concept_idx].clone().detach(),
                    "decoder_weight": concept_layer.decoder.weight[:, concept_idx].clone().detach(),
                    "decoder_bias": concept_layer.decoder.bias.clone().detach() if concept_layer.decoder.bias is not None else None
                }
            
            # Apply ablation
            if ablation_type == "zero":
                # Zero out the concept
                with torch.no_grad():
                    concept_layer.encoder.weight[concept_idx] = torch.zeros_like(
                        concept_layer.encoder.weight[concept_idx]
                    )
                    concept_layer.encoder.bias[concept_idx] = torch.zeros_like(
                        concept_layer.encoder.bias[concept_idx]
                    )
                    concept_layer.decoder.weight[:, concept_idx] = torch.zeros_like(
                        concept_layer.decoder.weight[:, concept_idx]
                    )
                
            elif ablation_type == "random":
                # Randomize the concept
                with torch.no_grad():
                    concept_layer.encoder.weight[concept_idx] = torch.randn_like(
                        concept_layer.encoder.weight[concept_idx]
                    )
                    concept_layer.encoder.bias[concept_idx] = torch.randn_like(
                        concept_layer.encoder.bias[concept_idx]
                    )
                    concept_layer.decoder.weight[:, concept_idx] = torch.randn_like(
                        concept_layer.decoder.weight[:, concept_idx]
                    )
                
            elif ablation_type == "mean":
                # Replace with mean of other concepts
                with torch.no_grad():
                    other_concepts = [i for i in range(concept_layer.m) if i != concept_idx]
                    mean_encoder_weight = concept_layer.encoder.weight[other_concepts].mean(0)
                    mean_encoder_bias = concept_layer.encoder.bias[other_concepts].mean()
                    mean_decoder_weight = concept_layer.decoder.weight[:, other_concepts].mean(1)
                    
                    concept_layer.encoder.weight[concept_idx] = mean_encoder_weight
                    concept_layer.encoder.bias[concept_idx] = mean_encoder_bias
                    concept_layer.decoder.weight[:, concept_idx] = mean_decoder_weight
                
            elif ablation_type == "noise":
                # Add noise to the concept
                with torch.no_grad():
                    noise_scale = 0.1
                    concept_layer.encoder.weight[concept_idx] += torch.randn_like(
                        concept_layer.encoder.weight[concept_idx]
                    ) * noise_scale
                    concept_layer.encoder.bias[concept_idx] += torch.randn_like(
                        concept_layer.encoder.bias[concept_idx]
                    ) * noise_scale
                    concept_layer.decoder.weight[:, concept_idx] += torch.randn_like(
                        concept_layer.decoder.weight[:, concept_idx]
                    ) * noise_scale
                    
        except Exception as e:
            print(f"Error during ablation of {concept_key}: {e}")
            # Restore the concept if there was an error
            if concept_key in self.original_concept_layers:
                self.restore_concept(concept_key)
    
    def restore_concept(self, concept_key: str) -> None:
        """
        Restore a concept to its original state.
        
        Args:
            concept_key: Concept to restore
        """
        try:
            if concept_key not in self.original_concept_layers:
                return
            
            # Parse concept key
            block_name, concept_idx = self._parse_concept_key(concept_key)
            concept_idx = int(concept_idx)
            
            # Get the concept layer
            concept_layer = self.model.concept_layers[block_name]
            
            # Restore original weights
            original = self.original_concept_layers[concept_key]
            with torch.no_grad():
                concept_layer.encoder.weight[concept_idx] = original["encoder_weight"]
                concept_layer.encoder.bias[concept_idx] = original["encoder_bias"]
                concept_layer.decoder.weight[:, concept_idx] = original["decoder_weight"]
                # Don't restore decoder bias as it's shared across all concepts
                
        except Exception as e:
            print(f"Error restoring concept {concept_key}: {e}")
    
    def restore_all_concepts(self) -> None:
        """Restore all concepts to their original state."""
        for concept_key in self.original_concept_layers.keys():
            self.restore_concept(concept_key)
    
    def _parse_concept_key(self, concept_key: str) -> Tuple[str, str]:
        """Parse concept key into block name and concept index."""
        # Format: "block_4_concept_0"
        parts = concept_key.split("_")
        block_name = f"block_{parts[1]}"
        concept_idx = parts[3]
        return block_name, concept_idx
    
    def measure_ablation_effect(
        self,
        test_texts: List[str],
        concept_keys: List[str],
        ablation_type: str = "zero",
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Measure the effect of ablating concepts on model performance.
        
        Args:
            test_texts: Texts to test on
            concept_keys: Concepts to ablate
            ablation_type: Type of ablation
            metrics: Metrics to compute ["perplexity", "logits", "activations"]
            
        Returns:
            Dictionary with ablation results
        """
        if metrics is None:
            metrics = ["perplexity", "logits", "activations"]
        
        results = {}
        
        # Get baseline performance
        print("Computing baseline performance...")
        baseline = self._compute_metrics(test_texts, metrics)
        
        # Test each concept ablation
        for concept_key in tqdm(concept_keys, desc="Ablating concepts"):
            print(f"\nAblating {concept_key}...")
            
            # Apply ablation
            self.ablate_concept(concept_key, ablation_type)
            
            # Compute metrics
            ablated = self._compute_metrics(test_texts, metrics)
            
            # Compute differences
            diff = {}
            for metric in metrics:
                if metric in baseline and metric in ablated:
                    if isinstance(baseline[metric], dict):
                        diff[metric] = {
                            k: ablated[metric][k] - baseline[metric][k] 
                            for k in baseline[metric].keys()
                        }
                    else:
                        diff[metric] = ablated[metric] - baseline[metric]
            
            results[concept_key] = {
                "baseline": baseline,
                "ablated": ablated,
                "difference": diff,
                "ablation_type": ablation_type
            }
            
            # Restore concept
            self.restore_concept(concept_key)
        
        self.ablation_results = results
        return results
    
    def _compute_metrics(
        self, 
        test_texts: List[str], 
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compute specified metrics on test texts."""
        results = {}
        
        self.model.eval()
        with torch.no_grad():
            for text in test_texts:
                # Tokenize
                input_ids = self.tokenizer.encode(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=128
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    return_concepts=True
                )
                
                # Compute metrics
                if "perplexity" in metrics:
                    if outputs["loss"] is not None:
                        perplexity = torch.exp(outputs["loss"]).item()
                        if "perplexity" not in results:
                            results["perplexity"] = []
                        results["perplexity"].append(perplexity)
                
                if "logits" in metrics:
                    logits = outputs["logits"]
                    if "logits" not in results:
                        results["logits"] = []
                    results["logits"].append(logits.cpu().numpy())
                
                if "activations" in metrics:
                    activations = outputs["concept_activations"]
                    if "activations" not in results:
                        results["activations"] = {}
                    for block_name, block_acts in activations.items():
                        if block_name not in results["activations"]:
                            results["activations"][block_name] = []
                        results["activations"][block_name].append(
                            block_acts.cpu().numpy()
                        )
        
        # Average results
        for metric in results:
            if isinstance(results[metric], list):
                if len(results[metric]) > 0:
                    if isinstance(results[metric][0], np.ndarray):
                        results[metric] = np.mean(results[metric], axis=0)
                    else:
                        results[metric] = np.mean(results[metric])
            elif isinstance(results[metric], dict):
                for key in results[metric]:
                    if isinstance(results[metric][key], list):
                        results[metric][key] = np.mean(results[metric][key], axis=0)
        
        return results


class ConceptEditor:
    """
    Tools for editing concept activations during inference.
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.edited_concepts = {}
    
    def edit_concept_activation(
        self,
        concept_key: str,
        activation_value: float,
        position: Optional[int] = None
    ) -> None:
        """
        Edit a concept's activation value.
        
        Args:
            concept_key: Concept to edit
            activation_value: New activation value
            position: Token position (None for all positions)
        """
        self.edited_concepts[concept_key] = {
            "value": activation_value,
            "position": position
        }
    
    def clear_edits(self) -> None:
        """Clear all concept edits."""
        self.edited_concepts = {}
    
    def generate_with_edits(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.8,
        **kwargs
    ) -> str:
        """
        Generate text with concept edits applied.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(
            prompt, 
            return_tensors='pt'
        ).to(self.device)
        
        # Custom generation with concept edits
        generated_ids = self._generate_with_concept_edits(
            input_ids, 
            max_length, 
            temperature, 
            **kwargs
        )
        
        # Decode
        generated_text = self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def _generate_with_concept_edits(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        **kwargs
    ) -> torch.Tensor:
        """
        Custom generation function that applies concept edits.
        """
        # Apply concept edits during generation
        with torch.no_grad():
            # Convert edits to the format expected by the model
            concept_edits = {}
            for concept_key, edit_info in self.edited_concepts.items():
                concept_edits[concept_key] = edit_info["value"]
            
            # Use the model's generate method with concept edits
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                concept_edits=concept_edits,
                **kwargs
            )
        
        return outputs
    
    def compare_generations(
        self,
        prompt: str,
        concept_edits: Dict[str, float],
        max_length: int = 50,
        num_samples: int = 3
    ) -> Dict[str, List[str]]:
        """
        Compare generations with different concept edits.
        
        Args:
            prompt: Input prompt
            concept_edits: Dictionary of concept edits
            max_length: Maximum generation length
            num_samples: Number of samples per condition
            
        Returns:
            Dictionary with generations for each condition
        """
        results = {
            "baseline": [],
            "edited": []
        }
        
        # Generate baseline samples
        print("Generating baseline samples...")
        for _ in range(num_samples):
            text = self.generate_with_edits(prompt, max_length)
            results["baseline"].append(text)
        
        # Apply edits and generate
        print("Generating edited samples...")
        for concept_key, value in concept_edits.items():
            self.edit_concept_activation(concept_key, value)
        
        for _ in range(num_samples):
            text = self.generate_with_edits(prompt, max_length)
            results["edited"].append(text)
        
        # Clear edits
        self.clear_edits()
        
        return results


class AblationAnalyzer:
    """
    Analyze and visualize ablation results.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_ablation_effects(
        self,
        ablation_results: Dict[str, Any],
        metric: str = "perplexity",
        top_k: int = 10
    ) -> plt.Figure:
        """
        Plot the effects of concept ablations.
        
        Args:
            ablation_results: Results from ConceptAblator
            metric: Metric to plot
            top_k: Number of top effects to show
            
        Returns:
            Matplotlib figure
        """
        # Extract effects
        effects = []
        for concept_key, result in ablation_results.items():
            if metric in result["difference"]:
                effect = result["difference"][metric]
                if isinstance(effect, dict):
                    # Take mean if it's a dictionary
                    effect = np.mean(list(effect.values()))
                effects.append((concept_key, effect))
        
        # Sort by absolute effect size
        effects.sort(key=lambda x: abs(x[1]), reverse=True)
        effects = effects[:top_k]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        concept_keys = [e[0] for e in effects]
        effect_sizes = [e[1] for e in effects]
        colors = ['red' if e > 0 else 'blue' for e in effect_sizes]
        
        bars = ax.barh(range(len(concept_keys)), effect_sizes, color=colors, alpha=0.7)
        ax.set_yticks(range(len(concept_keys)))
        ax.set_yticklabels(concept_keys)
        ax.set_xlabel(f"Change in {metric}")
        ax.set_title(f"Top {top_k} Concept Ablation Effects on {metric}")
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_concept_importance_matrix(
        self,
        ablation_results: Dict[str, Any],
        metrics: List[str] = None
    ) -> plt.Figure:
        """
        Plot concept importance matrix across multiple metrics.
        
        Args:
            ablation_results: Results from ConceptAblator
            metrics: Metrics to include
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ["perplexity"]
        
        # Extract effects for each metric
        concept_keys = list(ablation_results.keys())
        importance_matrix = []
        
        for metric in metrics:
            metric_effects = []
            for concept_key in concept_keys:
                result = ablation_results[concept_key]
                if metric in result["difference"]:
                    effect = result["difference"][metric]
                    if isinstance(effect, dict):
                        effect = np.mean(list(effect.values()))
                    metric_effects.append(effect)
                else:
                    metric_effects.append(0)
            importance_matrix.append(metric_effects)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(concept_keys) * 0.3), 8))
        
        im = ax.imshow(
            importance_matrix, 
            cmap='RdBu_r', 
            aspect='auto',
            vmin=-np.max(np.abs(importance_matrix)),
            vmax=np.max(np.abs(importance_matrix))
        )
        
        ax.set_xticks(range(len(concept_keys)))
        ax.set_xticklabels(concept_keys, rotation=45, ha='right')
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        ax.set_title("Concept Importance Matrix")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Effect Size")
        
        plt.tight_layout()
        return fig
    
    def generate_ablation_report(
        self,
        ablation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive ablation report.
        
        Args:
            ablation_results: Results from ConceptAblator
            save_path: Path to save report
            
        Returns:
            Report dictionary
        """
        report = {
            "summary": {
                "total_concepts": len(ablation_results),
                "metrics_analyzed": set(),
                "most_affected_concepts": {},
                "least_affected_concepts": {}
            },
            "detailed_results": {}
        }
        
        # Analyze each metric
        for concept_key, result in ablation_results.items():
            report["detailed_results"][concept_key] = {
                "ablation_type": result["ablation_type"],
                "effects": {}
            }
            
            for metric, effect in result["difference"].items():
                report["summary"]["metrics_analyzed"].add(metric)
                
                if isinstance(effect, dict):
                    effect_size = np.mean(list(effect.values()))
                else:
                    effect_size = effect
                
                report["detailed_results"][concept_key]["effects"][metric] = {
                    "effect_size": effect_size,
                    "baseline": result["baseline"].get(metric, None),
                    "ablated": result["ablated"].get(metric, None)
                }
        
        # Find most/least affected concepts
        for metric in report["summary"]["metrics_analyzed"]:
            effects = []
            for concept_key, result in ablation_results.items():
                if metric in result["difference"]:
                    effect = result["difference"][metric]
                    if isinstance(effect, dict):
                        effect = np.mean(list(effect.values()))
                    effects.append((concept_key, effect))
            
            if effects:
                effects.sort(key=lambda x: abs(x[1]), reverse=True)
                report["summary"]["most_affected_concepts"][metric] = effects[:5]
                report["summary"]["least_affected_concepts"][metric] = effects[-5:]
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


def run_ablation_demo():
    """
    Demo function for ablation testing.
    """
    print("=== CBT Ablation Testing Demo ===\n")
    
    # This would be used with a trained model
    print("Ablation testing allows you to:")
    print("1. Turn off specific concepts")
    print("2. Measure the impact on model performance")
    print("3. Identify which concepts are most important")
    print("4. Understand concept-behavior causality")
    
    print("\nExample usage:")
    print("ablator = ConceptAblator(model, tokenizer)")
    print("results = ablator.measure_ablation_effect(test_texts, concept_keys)")
    print("analyzer = AblationAnalyzer()")
    print("fig = analyzer.plot_ablation_effects(results)")


if __name__ == "__main__":
    run_ablation_demo() 
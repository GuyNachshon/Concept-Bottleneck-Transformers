"""
Concept analysis tools for CBT models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import json
import os


class ConceptMiner:
    """
    Mines concept activations to find top contexts and patterns.
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.concept_contexts = defaultdict(list)
        self.concept_activations = defaultdict(list)
    
    def mine_concepts(self, dataloader, max_samples=1000, activation_threshold=0.01):
        """
        Mine concept activations from a dataset.
        
        Args:
            dataloader: DataLoader with text data
            max_samples: Maximum number of samples to process
            activation_threshold: Minimum activation to consider "active"
            
        Returns:
            Dictionary of concept mining results
        """
        self.model.eval()
        sample_count = 0
        
        print(f"Mining concepts from {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if sample_count >= max_samples:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Get concept activations
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_concepts=True
                )
                
                concept_activations = outputs["concept_activations"]
                
                # Process each sample in the batch
                for i in range(input_ids.size(0)):
                    if sample_count >= max_samples:
                        break
                    
                    # Get tokens for this sample
                    tokens = self.tokenizer.convert_ids_to_tokens(
                        input_ids[i], skip_special_tokens=True
                    )
                    
                    # Process each concept block
                    for block_name, concepts in concept_activations.items():
                        block_concepts = concepts[i]  # [seq_len, num_concepts]
                        
                        # Find active concepts for each token
                        for token_idx, token_concepts in enumerate(block_concepts):
                            if token_idx >= len(tokens):
                                continue
                            
                            # Find concepts above threshold
                            active_concept_indices = torch.where(
                                token_concepts > activation_threshold
                            )[0]
                            
                            for concept_idx in active_concept_indices:
                                concept_idx = concept_idx.item()
                                activation = token_concepts[concept_idx].item()
                                
                                # Store context
                                context = self._extract_context(tokens, token_idx, window=5)
                                concept_key = f"{block_name}_concept_{concept_idx}"
                                
                                self.concept_contexts[concept_key].append({
                                    "context": context,
                                    "token": tokens[token_idx],
                                    "activation": activation,
                                    "token_idx": token_idx,
                                    "sample_idx": sample_count
                                })
                                
                                self.concept_activations[concept_key].append(activation)
                    
                    sample_count += 1
        
        return self._summarize_mining_results()
    
    def _extract_context(self, tokens, token_idx, window=5):
        """Extract context around a token."""
        start = max(0, token_idx - window)
        end = min(len(tokens), token_idx + window + 1)
        context_tokens = tokens[start:end]
        
        # Mark the target token
        target_idx = token_idx - start
        context_tokens[target_idx] = f"[{context_tokens[target_idx]}]"
        
        return " ".join(context_tokens)
    
    def _summarize_mining_results(self):
        """Summarize mining results."""
        summary = {}
        
        for concept_key, contexts in self.concept_contexts.items():
            if not contexts:
                continue
            
            # Sort by activation strength
            contexts.sort(key=lambda x: x["activation"], reverse=True)
            
            # Get top contexts
            top_contexts = contexts[:20]
            
            # Calculate statistics
            activations = [ctx["activation"] for ctx in contexts]
            
            summary[concept_key] = {
                "num_activations": len(contexts),
                "mean_activation": np.mean(activations),
                "max_activation": np.max(activations),
                "top_contexts": top_contexts,
                "unique_tokens": list(set(ctx["token"] for ctx in contexts)),
                "token_frequencies": Counter(ctx["token"] for ctx in contexts)
            }
        
        return summary


class ConceptLabeler:
    """
    Automatically labels concepts using LLM or rule-based methods.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo", use_llm=True):
        self.model_name = model_name
        self.use_llm = use_llm
        self.labels = {}
    
    def label_concepts(self, mining_results, max_contexts_per_concept=10):
        """
        Label concepts based on their top contexts.
        
        Args:
            mining_results: Results from ConceptMiner
            max_contexts_per_concept: Maximum contexts to use for labeling
            
        Returns:
            Dictionary of concept labels
        """
        print("Labeling concepts...")
        
        for concept_key, concept_data in tqdm(mining_results.items()):
            if not concept_data["top_contexts"]:
                continue
            
            # Get top contexts for labeling
            contexts = concept_data["top_contexts"][:max_contexts_per_concept]
            context_texts = [ctx["context"] for ctx in contexts]
            
            if self.use_llm:
                label = self._label_with_llm(context_texts)
            else:
                label = self._label_with_rules(context_texts, concept_data)
            
            self.labels[concept_key] = label
        
        return self.labels
    
    def _label_with_llm(self, contexts):
        """
        Label concept using LLM (placeholder for now).
        In practice, you'd use OpenAI API or similar.
        """
        # Placeholder - in practice, you'd make an API call like:
        # prompt = f"What concept unifies these contexts?\n\n" + "\n".join(contexts)
        # response = openai.ChatCompletion.create(model=self.model_name, messages=[...])
        # return response.choices[0].message.content
        
        # For now, use a simple heuristic
        return self._label_with_rules(contexts, {})
    
    def _label_with_rules(self, contexts, concept_data):
        """
        Rule-based concept labeling.
        """
        # Extract common patterns
        all_tokens = []
        for context in contexts:
            tokens = context.split()
            all_tokens.extend(tokens)
        
        # Simple heuristics
        token_counter = Counter(all_tokens)
        most_common = token_counter.most_common(5)
        
        # Look for semantic patterns
        if any("the" in context.lower() for context in contexts):
            return "definite_article"
        elif any("and" in context.lower() for context in contexts):
            return "conjunction"
        elif any("is" in context.lower() or "are" in context.lower() for context in contexts):
            return "copula"
        elif any("." in context for context in contexts):
            return "punctuation"
        elif len(most_common) > 0 and most_common[0][1] > 3:
            return f"frequent_token_{most_common[0][0]}"
        else:
            return "general_concept"


class ConceptVisualizer:
    """
    Visualizes concept activations and patterns.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_concept_heatmap(self, concept_activations, title="Concept Activation Heatmap"):
        """
        Plot concept activation heatmap.
        
        Args:
            concept_activations: Dict of concept activations {block_name: tensor}
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        for idx, (block_name, activations) in enumerate(concept_activations.items()):
            if idx >= 4:  # Limit to 4 blocks
                break
            
            # Convert to numpy and take mean across batch
            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            
            # Average across batch dimension if present
            if activations.ndim == 3:  # [batch, seq, concepts]
                activations = activations.mean(axis=0)
            
            # Plot heatmap
            sns.heatmap(
                activations.T,  # Transpose to show concepts on y-axis
                ax=axes[idx],
                cmap='viridis',
                cbar_kws={'label': 'Activation Strength'}
            )
            axes[idx].set_title(f"{block_name}")
            axes[idx].set_xlabel("Token Position")
            axes[idx].set_ylabel("Concept Index")
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_concept_sparsity(self, concept_activations, title="Concept Sparsity Analysis"):
        """
        Plot sparsity analysis for concepts.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        for idx, (block_name, activations) in enumerate(concept_activations.items()):
            if idx >= 4:
                break
            
            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            
            # Calculate sparsity per concept
            concept_sparsity = []
            for concept_idx in range(activations.shape[-1]):
                concept_activations = activations[..., concept_idx]
                sparsity = (concept_activations == 0).mean()
                concept_sparsity.append(sparsity)
            
            # Plot sparsity distribution
            axes[idx].hist(concept_sparsity, bins=20, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f"{block_name} - Concept Sparsity")
            axes[idx].set_xlabel("Sparsity (fraction of zeros)")
            axes[idx].set_ylabel("Number of Concepts")
            axes[idx].axvline(np.mean(concept_sparsity), color='red', linestyle='--', 
                            label=f'Mean: {np.mean(concept_sparsity):.3f}')
            axes[idx].legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_concept_usage(self, mining_results, title="Concept Usage Patterns"):
        """
        Plot concept usage patterns from mining results.
        """
        # Prepare data
        concept_names = []
        activation_counts = []
        mean_activations = []
        
        for concept_key, data in mining_results.items():
            concept_names.append(concept_key)
            activation_counts.append(data["num_activations"])
            mean_activations.append(data["mean_activation"])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot activation counts
        ax1.bar(range(len(concept_names)), activation_counts, alpha=0.7)
        ax1.set_title("Number of Activations per Concept")
        ax1.set_xlabel("Concept Index")
        ax1.set_ylabel("Number of Activations")
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot mean activations
        ax2.bar(range(len(concept_names)), mean_activations, alpha=0.7, color='orange')
        ax2.set_title("Mean Activation Strength per Concept")
        ax2.set_xlabel("Concept Index")
        ax2.set_ylabel("Mean Activation")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_concept_clustering(self, concept_activations, n_clusters=8, title="Concept Clustering"):
        """
        Cluster concepts and visualize relationships.
        """
        # Prepare data for clustering
        all_concepts = []
        concept_names = []
        
        for block_name, activations in concept_activations.items():
            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            
            # Average across batch and sequence dimensions
            if activations.ndim == 3:
                activations = activations.mean(axis=(0, 1))  # [concepts]
            
            all_concepts.append(activations)
            concept_names.extend([f"{block_name}_c{i}" for i in range(len(activations))])
        
        # Combine all concepts
        X = np.array(all_concepts).T  # [num_concepts, num_blocks]
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        ax.set_title(f"{title} (PCA projection)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        
        plt.tight_layout()
        return fig


class ConceptAnalyzer:
    """
    Main class for comprehensive concept analysis.
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.miner = ConceptMiner(model, tokenizer, device)
        self.labeler = ConceptLabeler()
        self.visualizer = ConceptVisualizer()
        
        self.mining_results = None
        self.labels = None
    
    def analyze_concepts(self, dataloader, save_path=None, max_samples=1000):
        """
        Perform comprehensive concept analysis.
        
        Args:
            dataloader: DataLoader with text data
            save_path: Path to save analysis results
            max_samples: Maximum samples to analyze
            
        Returns:
            Dictionary with all analysis results
        """
        print("Starting comprehensive concept analysis...")
        
        # Step 1: Mine concepts
        self.mining_results = self.miner.mine_concepts(dataloader, max_samples)
        
        # Step 2: Label concepts
        self.labels = self.labeler.label_concepts(self.mining_results)
        
        # Step 3: Create visualizations
        self._create_visualizations()
        
        # Step 4: Save results
        if save_path:
            self._save_results(save_path)
        
        return {
            "mining_results": self.mining_results,
            "labels": self.labels,
            "summary": self._create_summary()
        }
    
    def _create_visualizations(self):
        """Create and save concept visualizations."""
        # Get concept activations for visualization
        # This would require running the model on a sample batch
        # For now, we'll create basic plots from mining results
        
        if self.mining_results:
            # Plot concept usage
            fig = self.visualizer.plot_concept_usage(self.mining_results)
            plt.savefig("concept_usage.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_summary(self):
        """Create analysis summary."""
        if not self.mining_results:
            return {
                "total_concepts": 0,
                "total_activations": 0,
                "concepts_with_labels": 0,
                "most_active_concepts": [],
                "concept_diversity": {}
            }
        
        summary = {
            "total_concepts": len(self.mining_results),
            "total_activations": sum(data["num_activations"] for data in self.mining_results.values()),
            "concepts_with_labels": len(self.labels) if self.labels else 0,
            "most_active_concepts": [],
            "concept_diversity": {}
        }
        
        # Find most active concepts
        concept_activity = [(k, v["num_activations"]) for k, v in self.mining_results.items()]
        concept_activity.sort(key=lambda x: x[1], reverse=True)
        summary["most_active_concepts"] = concept_activity[:10]
        
        # Analyze concept diversity
        for concept_key, data in self.mining_results.items():
            unique_tokens = len(data["unique_tokens"])
            summary["concept_diversity"][concept_key] = unique_tokens
        
        return summary
    
    def _save_results(self, save_path):
        """Save analysis results to file."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save mining results
        with open(os.path.join(save_path, "mining_results.json"), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for k, v in self.mining_results.items():
                json_results[k] = {
                    "num_activations": v["num_activations"],
                    "mean_activation": float(v["mean_activation"]),
                    "max_activation": float(v["max_activation"]),
                    "top_contexts": v["top_contexts"],
                    "unique_tokens": v["unique_tokens"],
                    "token_frequencies": dict(v["token_frequencies"])
                }
            json.dump(json_results, f, indent=2)
        
        # Save labels
        if self.labels:
            with open(os.path.join(save_path, "concept_labels.json"), "w") as f:
                json.dump(self.labels, f, indent=2)
        
        # Save summary
        summary = self._create_summary()
        with open(os.path.join(save_path, "analysis_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis results saved to {save_path}")
    
    def get_concept_report(self, concept_key):
        """
        Get detailed report for a specific concept.
        
        Args:
            concept_key: Concept identifier
            
        Returns:
            Dictionary with concept details
        """
        if not self.mining_results or concept_key not in self.mining_results:
            return None
        
        data = self.mining_results[concept_key]
        label = self.labels.get(concept_key, "unlabeled") if self.labels else "unlabeled"
        
        return {
            "concept_key": concept_key,
            "label": label,
            "num_activations": data["num_activations"],
            "mean_activation": data["mean_activation"],
            "max_activation": data["max_activation"],
            "top_contexts": data["top_contexts"][:10],
            "unique_tokens": data["unique_tokens"],
            "most_common_tokens": data["token_frequencies"].most_common(10)
        } 
#!/usr/bin/env python3
"""
Robust Multi-Seed Experiments for CBT
Establishes statistical rigor through multiple seeds and proper analysis.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset

# Import CBT modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cbt.config import CBTConfig, load_config
from cbt.trainer import CBTTrainer
from cbt.evaluator import CBTEvaluator
from cbt.analyzer import ConceptAnalyzer
from cbt.model import CBTModel


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
    """Collate function for DataLoader with padding."""
    # Find the maximum length in this batch
    max_length = max(item['input_ids'].size(0) for item in batch)
    
    # Pad sequences to the same length
    padded_input_ids = []
    padded_attention_mask = []
    
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        
        # Calculate padding length
        pad_length = max_length - input_ids.size(0)
        
        if pad_length > 0:
            # Pad with tokenizer's pad token ID (usually 50256 for GPT-2)
            padded_input = torch.cat([input_ids, torch.full((pad_length,), 50256, dtype=input_ids.dtype)])
            padded_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=attention_mask.dtype)])
        else:
            padded_input = input_ids
            padded_mask = attention_mask
        
        padded_input_ids.append(padded_input)
        padded_attention_mask.append(padded_mask)
    
    # Stack the padded tensors
    input_ids = torch.stack(padded_input_ids)
    attention_mask = torch.stack(padded_attention_mask)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


@dataclass
class ExperimentConfig:
    """Configuration for robust experiments"""
    num_seeds: int = 10
    base_config_path: str = "configs/training.yaml"
    output_dir: str = "results/robust_experiments"
    save_models: bool = True
    parallel_runs: int = 4


class RobustExperimentRunner:
    """Runs robust multi-seed experiments with statistical analysis"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_data_and_model(self, config: CBTConfig):
        """Setup data loaders and model for training.

        Returns a tuple of (model, train_dataloader, val_dataloader, train_dataset, tokenizer).
        """
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
            concept_blocks=config.model.concept_blocks,
            d_model=getattr(config.model, 'd_model', 768),  # Default to 768 if not present
            m=config.model.m,
            k=config.model.k,
            alpha=0.0  # Start with no concept influence
        )
        
        return model, train_dataloader, val_dataloader, train_dataset, tokenizer
        
    def run_single_seed(self, seed: int) -> Dict[str, Any]:
        """Run a single experiment with given seed"""
        self.logger.info(f"Starting experiment with seed {seed}")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load base configuration
        base_config = load_config(self.config.base_config_path)
        
        # Create seed-specific config
        config = CBTConfig(
            model=base_config.model,
            training=base_config.training,
            advanced_losses=base_config.advanced_losses,
            data=base_config.data,
            evaluation=base_config.evaluation,
            logging=base_config.logging,
            hardware=base_config.hardware
        )
        
        # Set seed in config
        config.training.seed = seed
        
        # Create experiment directory
        exp_dir = self.output_dir / f"seed_{seed:02d}"
        exp_dir.mkdir(exist_ok=True)
        
        try:
            # Setup data and model
            model, train_dataloader, val_dataloader, train_dataset, tokenizer = self.setup_data_and_model(config)
            
            # Create trainer with proper arguments
            trainer = CBTTrainer(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                learning_rate=float(config.training.learning_rate),
                weight_decay=0.01,  # Default weight decay
                device=config.get_device(),
                use_wandb=config.logging.use_wandb,
                project_name="cbt-robust-experiments",  # Default project name
                use_advanced_losses=config.advanced_losses.enabled,
                advanced_loss_config={
                    "orthogonality_weight": float(config.advanced_losses.orthogonality_weight),
                    "stability_weight": float(config.advanced_losses.stability_weight),
                    "kl_weight": float(config.advanced_losses.kl_weight),
                    "concept_dropout_weight": float(config.advanced_losses.dropout_weight)
                } if config.advanced_losses.enabled else None,
                gradient_clip_max_norm=float(config.training.gradient_clip_max_norm),
                use_mixed_precision=config.training.use_mixed_precision,
                freeze_base_until_alpha=float(config.training.freeze_base_until_alpha)
            )
            
            # Run training
            # Ensure alpha_schedule values are floats
            alpha_schedule = [float(x) for x in config.training.alpha_schedule]
            training_results = trainer.train(
                num_epochs=config.training.num_epochs,
                alpha_schedule=alpha_schedule,
                save_path=str(exp_dir / 'model.pt') if self.config.save_models else None
            )
            
            # Run evaluation
            # Create base model for evaluation (reuse tokenizer from setup)
            from transformers import GPT2LMHeadModel
            base_model = GPT2LMHeadModel.from_pretrained("gpt2")
            
            evaluator = CBTEvaluator(
                cbt_model=model,
                base_model=base_model,
                tokenizer=tokenizer,
                device=config.get_device()
            )
            
            # Load evaluation dataset
            eval_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
            
            # Create evaluation texts from the validation set
            eval_texts = []
            for item in eval_dataset:
                text = item['text'].strip()
                if text and len(text) > 20:  # Only use non-empty texts with reasonable length
                    eval_texts.append(text)
                    if len(eval_texts) >= 100:  # Limit to 100 evaluation texts
                        break
            
            eval_results = evaluator.evaluate_all_criteria(eval_texts)
            
            # Run concept analysis (simplified for now)
            try:
                analyzer = ConceptAnalyzer(
                    model=model,
                    tokenizer=tokenizer,
                    device=config.get_device(),
                    use_llm_labeling=False  # Disable LLM labeling for faster execution
                )
                # Create a simple dataloader for analysis using a small subset of the train dataset
                analysis_dataset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
                analysis_dataloader = DataLoader(
                    analysis_dataset,
                    batch_size=4,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0
                )
                analysis_results = analyzer.analyze_concepts(analysis_dataloader, max_samples=50)
            except Exception as e:
                self.logger.warning(f"Concept analysis failed for seed {seed}: {e}")
                analysis_results = {"error": str(e), "status": "failed"}
            
            # Combine results
            results = {
                'seed': seed,
                'training': training_results,
                'evaluation': eval_results,
                'analysis': analysis_results,
                'config': config.to_dict()
            }
            
            # Save results
            with open(exp_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Completed experiment with seed {seed}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed experiment with seed {seed}: {e}")
            return {
                'seed': seed,
                'error': str(e),
                'status': 'failed'
            }
    
    def run_all_seeds(self) -> List[Dict[str, Any]]:
        """Run experiments for all seeds"""
        self.logger.info(f"Starting robust experiments with {self.config.num_seeds} seeds")
        
        seeds = list(range(self.config.num_seeds))
        results = []
        
        if self.config.parallel_runs > 1:
            # Run in parallel
            with ProcessPoolExecutor(max_workers=self.config.parallel_runs) as executor:
                future_to_seed = {executor.submit(self.run_single_seed, seed): seed for seed in seeds}
                
                for future in future_to_seed:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        seed = future_to_seed[future]
                        self.logger.error(f"Failed to get result for seed {seed}: {e}")
                        results.append({'seed': seed, 'error': str(e), 'status': 'failed'})
        else:
            # Run sequentially
            for seed in seeds:
                result = self.run_single_seed(seed)
                results.append(result)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis of results"""
        self.logger.info("Performing statistical analysis")
        
        # Filter successful runs
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        self.logger.info(f"Successful runs: {len(successful_results)}/{len(results)}")
        
        if len(successful_results) < 3:
            self.logger.warning("Insufficient successful runs for statistical analysis")
            return {
                'total_runs': len(results),
                'successful_runs': len(successful_results),
                'failed_runs': len(failed_results),
                'success_rate': len(successful_results) / len(results) if results else 0.0,
                'error': 'Insufficient successful runs'
            }
        
        # Extract key metrics
        metrics = {
            'perplexity': [],
            'quality_hit': [],
            'sparsity': [],
            'active_concepts': [],
            'concept_impact': []
        }
        
        for result in successful_results:
            if 'evaluation' in result:
                eval_data = result['evaluation']
                metrics['perplexity'].append(eval_data.get('perplexity', np.nan))
                metrics['quality_hit'].append(eval_data.get('quality_hit', np.nan))
                metrics['sparsity'].append(eval_data.get('sparsity', np.nan))
                metrics['active_concepts'].append(eval_data.get('active_concepts', np.nan))
            
            if 'analysis' in result and 'causality_results' in result['analysis']:
                causality = result['analysis']['causality_results']
                if causality and len(causality) > 0:
                    max_impact = max(c['percent_change'] for c in causality)
                    metrics['concept_impact'].append(max_impact)
        
        # Statistical analysis
        stats_analysis = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                values = np.array(values)
                values = values[~np.isnan(values)]  # Remove NaN values
                
                if len(values) > 0:
                    stats_analysis[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values),
                        'confidence_interval_95': self._confidence_interval(values, 0.95),
                        'confidence_interval_99': self._confidence_interval(values, 0.99)
                    }
        
        # Overall analysis
        analysis = {
            'total_runs': len(results),
            'successful_runs': len(successful_results),
            'failed_runs': len(failed_results),
            'success_rate': len(successful_results) / len(results),
            'metrics': stats_analysis,
            'failed_seeds': [r['seed'] for r in failed_results]
        }
        
        # Save analysis
        with open(self.output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _confidence_interval(self, values: np.ndarray, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval"""
        if len(values) < 2:
            return (float(values[0]), float(values[0]))
        
        mean = np.mean(values)
        std_err = stats.sem(values)
        ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=std_err)
        return (float(ci[0]), float(ci[1]))
    
    def create_visualizations(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Create visualizations of results"""
        self.logger.info("Creating visualizations")
        
        # Filter successful runs
        successful_results = [r for r in results if 'error' not in r]
        
        if len(successful_results) < 2:
            self.logger.warning("Insufficient data for visualizations")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('CBT Robust Experiments - Multi-Seed Analysis', fontsize=16)
        
        # Perplexity distribution
        perplexities = [r['evaluation']['perplexity'] for r in successful_results if 'evaluation' in r]
        if perplexities:
            axes[0, 0].hist(perplexities, bins=10, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(np.mean(perplexities), color='red', linestyle='--', label=f'Mean: {np.mean(perplexities):.2f}')
            axes[0, 0].set_title('Perplexity Distribution')
            axes[0, 0].set_xlabel('Perplexity')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # Quality hit distribution
        quality_hits = [r['evaluation']['quality_hit'] for r in successful_results if 'evaluation' in r]
        if quality_hits:
            axes[0, 1].hist(quality_hits, bins=10, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(np.mean(quality_hits), color='red', linestyle='--', label=f'Mean: {np.mean(quality_hits):.3f}%')
            axes[0, 1].set_title('Quality Hit Distribution')
            axes[0, 1].set_xlabel('Quality Hit (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # Sparsity distribution
        sparsities = [r['evaluation']['sparsity'] for r in successful_results if 'evaluation' in r]
        if sparsities:
            axes[0, 2].hist(sparsities, bins=10, alpha=0.7, edgecolor='black')
            axes[0, 2].axvline(np.mean(sparsities), color='red', linestyle='--', label=f'Mean: {np.mean(sparsities):.3f}')
            axes[0, 2].set_title('Sparsity Distribution')
            axes[0, 2].set_xlabel('Sparsity')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
        
        # Concept impact distribution
        concept_impacts = []
        for r in successful_results:
            if 'analysis' in r and 'causality_results' in r['analysis']:
                causality = r['analysis']['causality_results']
                if causality:
                    max_impact = max(c['percent_change'] for c in causality)
                    concept_impacts.append(max_impact)
        
        if concept_impacts:
            axes[1, 0].hist(concept_impacts, bins=10, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(concept_impacts), color='red', linestyle='--', label=f'Mean: {np.mean(concept_impacts):.1f}%')
            axes[1, 0].set_title('Max Concept Impact Distribution')
            axes[1, 0].set_xlabel('Impact (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Training curves (if available)
        if 'training' in successful_results[0]:
            # Plot training loss for first few seeds
            for i, result in enumerate(successful_results[:3]):
                if 'training' in result and 'losses' in result['training']:
                    losses = result['training']['losses']
                    if losses:
                        epochs = range(len(losses))
                        axes[1, 1].plot(epochs, losses, label=f'Seed {result["seed"]}', alpha=0.7)
            
            axes[1, 1].set_title('Training Loss Curves')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
        
        # Success rate summary
        success_rate = analysis['success_rate']
        axes[1, 2].pie([success_rate, 1-success_rate], 
                      labels=['Success', 'Failure'],
                      autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 2].set_title('Experiment Success Rate')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robust_experiments_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Visualizations saved")
    
    def run_complete_analysis(self):
        """Run complete robust analysis"""
        self.logger.info("Starting complete robust analysis")
        
        # Run all seeds
        results = self.run_all_seeds()
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Create visualizations
        self.create_visualizations(results, analysis)
        
        # Save complete results
        with open(self.output_dir / 'complete_results.json', 'w') as f:
            json.dump({
                'experiment_config': self.config.__dict__,
                'individual_results': results,
                'statistical_analysis': analysis
            }, f, indent=2, default=str)
        
        self.logger.info("Complete robust analysis finished")
        return analysis


def main():
    """Main function for robust experiments"""
    config = ExperimentConfig(
        num_seeds=10,
        base_config_path="configs/training.yaml",
        output_dir="results/robust_experiments",
        save_models=True,
        parallel_runs=4
    )
    
    runner = RobustExperimentRunner(config)
    analysis = runner.run_complete_analysis()
    
    print("\n" + "="*60)
    print("ROBUST EXPERIMENT RESULTS")
    print("="*60)
    
    # Handle case where analysis might not have expected keys
    total_runs = analysis.get('total_runs', 0)
    successful_runs = analysis.get('successful_runs', 0)
    success_rate = analysis.get('success_rate', 0.0)
    
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Success rate: {success_rate:.1%}")
    
    if 'error' in analysis:
        print(f"\nAnalysis error: {analysis['error']}")
    elif 'metrics' in analysis:
        print("\nKey Metrics (mean ± std):")
        for metric, stats in analysis['metrics'].items():
            ci_95 = stats['confidence_interval_95']
            print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}])")


if __name__ == "__main__":
    main() 
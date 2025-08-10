#!/usr/bin/env python3
"""
Command-line interface for CBT experiments.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from cbt import (
    CBTModel, CBTTrainer, CBTEvaluator, CBTConfig, load_config,
    create_experiment_config
)
from cbt.evaluator import get_wikitext_eval_texts
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import logging


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def train_model(config: CBTConfig, save_path: str = None):
    """Train a CBT model with the given configuration."""
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = config.get_device()
    logger.info(f"Using device: {device}")
    
    # Setup deterministic behavior
    if config.hardware.deterministic:
        torch.manual_seed(config.hardware.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Create model
    model = CBTModel(
        base_model_name=config.model.base_model_name,
        concept_blocks=config.model.concept_blocks,
        m=config.model.m,
        k=config.model.k,
        alpha=config.model.alpha
    )
    model.to(device)
    
    # Create trainer
    trainer = CBTTrainer(
        model=model,
        config=config,
        device=device
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'trainer_state': trainer.get_state()
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    return model, trainer


def evaluate_model(model: CBTModel, config: CBTConfig, eval_texts: list = None):
    """Evaluate a CBT model."""
    logger = logging.getLogger(__name__)
    
    device = config.get_device()
    
    # Load base model for comparison
    base_model = GPT2LMHeadModel.from_pretrained(config.model.base_model_name)
    base_model.to(device)
    
    # Get tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.model.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Get evaluation texts
    if eval_texts is None:
        eval_texts = get_wikitext_eval_texts(num_samples=config.evaluation.num_samples)
    
    # Create evaluator
    evaluator = CBTEvaluator(model, base_model, tokenizer, device)
    
    # Evaluate
    logger.info("Evaluating model...")
    results = evaluator.evaluate_all_criteria(eval_texts)
    
    return results


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="CBT Experiment CLI")
    parser.add_argument("command", choices=["train", "evaluate", "train-and-evaluate", "concept-analysis"], 
                       help="Command to run")
    parser.add_argument("--config", "-c", type=str, default="configs/training.yaml",
                       help="Path to configuration file")
    parser.add_argument("--save-path", "-s", type=str, default="results/models/cbt_model.pt",
                       help="Path to save model")
    parser.add_argument("--eval-texts", "-e", type=int, default=200,
                       help="Number of evaluation texts")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--results-dir", "-r", type=str, default=None,
                       help="Specific results directory to use (auto-detects latest if not specified)")
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Specific model file to analyze (e.g., cbt_model_stab_kl_m32_k4_a30.pt)")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)
    
    logger = logging.getLogger(__name__)
    
    if args.command == "concept-analysis":
        # Run concept analysis
        logger.info("Running concept analysis...")
        
        # Import and run the concept analysis
        from experiments.run_concept_analysis import main as run_concept_analysis
        
        # If user specified a model, pass it as command line argument
        if args.model:
            import sys
            sys.argv = [sys.argv[0], args.model]
        
        run_concept_analysis()
        return
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.info("Using default configuration")
        config = CBTConfig()
    
    if args.command == "train":
        # Train model
        model, trainer = train_model(config, args.save_path)
        logger.info("Training completed!")
        
    elif args.command == "evaluate":
        # Load model and evaluate
        if not os.path.exists(args.save_path):
            logger.error(f"Model not found at {args.save_path}")
            return
        
        checkpoint = torch.load(args.save_path, map_location='cpu')
        config = CBTConfig.from_dict(checkpoint['config'])
        
        model = CBTModel(
            base_model_name=config.model.base_model_name,
            concept_blocks=config.model.concept_blocks,
            m=config.model.m,
            k=config.model.k,
            alpha=config.model.alpha
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        results = evaluate_model(model, config)
        logger.info("Evaluation results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        
    elif args.command == "train-and-evaluate":
        # Train and evaluate
        model, trainer = train_model(config, args.save_path)
        results = evaluate_model(model, config)
        
        logger.info("Training and evaluation completed!")
        logger.info("Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main() 
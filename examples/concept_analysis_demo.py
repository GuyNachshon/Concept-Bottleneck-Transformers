#!/usr/bin/env python3
"""
Concept Analysis Demo for CBT models.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbt.model import CBTModel
from cbt.concept_analysis import ConceptAnalyzer, ConceptVisualizer


class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            self.encodings.append(encoding)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx]['input_ids'].squeeze(),
            'attention_mask': self.encodings[idx]['attention_mask'].squeeze()
        }


def create_diverse_sample_data():
    """Create diverse sample training data for better concept analysis."""
    sample_texts = [
        # Articles and determiners
        "The quick brown fox jumps over the lazy dog.",
        "A cat sat on the mat.",
        "An apple fell from the tree.",
        "The book is on the table.",
        
        # Conjunctions
        "The weather is cold and windy.",
        "She likes coffee but he prefers tea.",
        "You can have cake or ice cream.",
        "The movie was long yet entertaining.",
        
        # Copulas and linking verbs
        "The sky is blue.",
        "She seems happy.",
        "They are students.",
        "The food tastes delicious.",
        
        # Punctuation
        "Hello, how are you?",
        "Stop! Don't go there.",
        "The end.",
        "What time is it?",
        
        # Numbers and quantities
        "There are five apples.",
        "The temperature is 25 degrees.",
        "I have three cats.",
        "The score was 2-1.",
        
        # Colors and descriptions
        "The red car is fast.",
        "The blue sky is clear.",
        "The green grass is wet.",
        "The yellow sun is bright.",
        
        # Actions and verbs
        "The bird flies high.",
        "The fish swims fast.",
        "The dog runs quickly.",
        "The cat sleeps peacefully.",
        
        # Emotions and states
        "She feels happy.",
        "He looks sad.",
        "They sound excited.",
        "The child appears tired.",
        
        # Time and temporal concepts
        "Today is Monday.",
        "Yesterday was Sunday.",
        "Tomorrow will be Tuesday.",
        "The meeting is at 3 PM.",
        
        # Spatial concepts
        "The book is on the shelf.",
        "The car is in the garage.",
        "The bird is above the tree.",
        "The fish is under the water.",
        
        # Questions and interrogatives
        "What is your name?",
        "Where are you going?",
        "When will you arrive?",
        "Why did you leave?",
        
        # Negation
        "I do not like coffee.",
        "She is not here.",
        "They cannot come.",
        "The door is not open.",
        
        # Possession
        "This is my book.",
        "That is her car.",
        "The dog's tail is wagging.",
        "The children's toys are everywhere.",
        
        # Comparisons
        "The cat is bigger than the mouse.",
        "She is as tall as her sister.",
        "This is the best movie.",
        "The weather is getting warmer.",
        
        # Conditionals
        "If it rains, I will stay home.",
        "When you arrive, call me.",
        "Unless you hurry, you'll be late.",
        "Since you're here, let's start.",
        
        # Lists and sequences
        "I need milk, bread, and eggs.",
        "The colors are red, blue, and green.",
        "First, second, and third place.",
        "Monday, Tuesday, and Wednesday.",
        
        # Quantifiers
        "All students passed the test.",
        "Some people like spicy food.",
        "Many birds migrate south.",
        "Few cars were on the road.",
        
        # Modal verbs
        "I can swim.",
        "She must study hard.",
        "They should arrive soon.",
        "You may enter now."
    ]
    return sample_texts


def main():
    """Main concept analysis demonstration."""
    print("Setting up CBT concept analysis demo...")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create diverse sample data
    sample_texts = create_diverse_sample_data()
    print(f"Created {len(sample_texts)} diverse text samples")
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(sample_texts, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Load or create a trained CBT model
    print("Loading CBT model...")
    
    # Try to load a trained model, or create a new one
    model_path = "cbt_advanced_model.pt"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        model = CBTModel(
            base_model_name="gpt2",
            concept_blocks=[4, 5, 6, 7],
            d_model=768,
            m=64,
            k=8,
            alpha=1.0
        )
        # Load the checkpoint and extract model state
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No trained model found, creating a new one for demo")
        model = CBTModel(
            base_model_name="gpt2",
            concept_blocks=[4, 5, 6, 7],
            d_model=768,
            m=64,
            k=8,
            alpha=1.0
        )
    
    model = model.to(device)
    model.eval()
    
    # Create concept analyzer
    print("Initializing concept analyzer...")
    analyzer = ConceptAnalyzer(model, tokenizer, device)
    
    # Perform comprehensive analysis
    print("Starting concept analysis...")
    results = analyzer.analyze_concepts(
        dataloader,
        save_path="concept_analysis_results",
        max_samples=50  # Limit for demo
    )
    
    # Print analysis summary
    print("\n" + "="*50)
    print("CONCEPT ANALYSIS SUMMARY")
    print("="*50)
    
    summary = results["summary"]
    print(f"Total concepts analyzed: {summary['total_concepts']}")
    print(f"Total activations found: {summary['total_activations']}")
    print(f"Concepts with labels: {summary['concepts_with_labels']}")
    
    # Show most active concepts
    print("\nMost Active Concepts:")
    for i, (concept_key, count) in enumerate(summary["most_active_concepts"][:5]):
        label = results["labels"].get(concept_key, "unlabeled")
        print(f"  {i+1}. {concept_key} ({label}): {count} activations")
    
    # Show some concept details
    print("\n" + "="*50)
    print("SAMPLE CONCEPT REPORTS")
    print("="*50)
    
    # Get detailed reports for top concepts
    for i, (concept_key, count) in enumerate(summary["most_active_concepts"][:3]):
        report = analyzer.get_concept_report(concept_key)
        if report:
            print(f"\nConcept: {concept_key}")
            print(f"Label: {report['label']}")
            print(f"Activations: {report['num_activations']}")
            print(f"Mean activation: {report['mean_activation']:.3f}")
            print(f"Max activation: {report['max_activation']:.3f}")
            
            print("Top contexts:")
            for j, ctx in enumerate(report["top_contexts"][:3]):
                print(f"  {j+1}. {ctx['context']} (activation: {ctx['activation']:.3f})")
            
            print("Most common tokens:")
            for token, freq in report["most_common_tokens"][:5]:
                print(f"  - {token}: {freq}")
    
    # Create additional visualizations
    print("\nCreating visualizations...")
    visualizer = ConceptVisualizer()
    
    # Get concept activations for visualization
    print("Getting concept activations for visualization...")
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        sample_batch = next(iter(dataloader))
        input_ids = sample_batch["input_ids"].to(device)
        attention_mask = sample_batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, return_concepts=True)
        concept_activations = outputs["concept_activations"]
        
        # Create heatmap
        print("Creating concept activation heatmap...")
        fig = visualizer.plot_concept_heatmap(concept_activations)
        plt.savefig("concept_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create sparsity analysis
        print("Creating sparsity analysis...")
        fig = visualizer.plot_concept_sparsity(concept_activations)
        plt.savefig("concept_sparsity.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create clustering analysis
        print("Creating concept clustering...")
        fig = visualizer.plot_concept_clustering(concept_activations)
        plt.savefig("concept_clustering.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print("Generated files:")
    print("  - concept_analysis_results/ (analysis data)")
    print("  - concept_usage.png (usage patterns)")
    print("  - concept_heatmap.png (activation heatmap)")
    print("  - concept_sparsity.png (sparsity analysis)")
    print("  - concept_clustering.png (concept clustering)")
    
    # Interactive concept exploration
    print("\nInteractive concept exploration:")
    print("You can explore specific concepts using:")
    print("  analyzer.get_concept_report('block_4_concept_0')")
    
    # Example: Show a specific concept
    if results["mining_results"]:
        example_concept = list(results["mining_results"].keys())[0]
        print(f"\nExample concept report for {example_concept}:")
        report = analyzer.get_concept_report(example_concept)
        if report:
            print(f"  Label: {report['label']}")
            print(f"  Activations: {report['num_activations']}")
            print(f"  Top context: {report['top_contexts'][0]['context']}")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Extract and display concept contexts from CBT analysis results.
"""

import json
import numpy as np
from pathlib import Path

def extract_contexts_from_results(results_path):
    """Extract concept contexts from analysis results."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Check if concept_contexts exists
    if 'concept_contexts' not in results:
        print("❌ No concept_contexts found in results!")
        print("Available keys:", list(results.keys()))
        return None
    
    return results['concept_contexts']

def display_top_contexts(concept_contexts, top_k=5):
    """Display top contexts for each concept."""
    print("🔍 CONCEPT CONTEXTS ANALYSIS")
    print("=" * 60)
    
    for block_name, block_concepts in concept_contexts.items():
        print(f"\n📦 {block_name.upper()}:")
        
        # Sort concepts by number of contexts
        concept_stats = []
        for concept_idx, contexts in block_concepts.items():
            if len(contexts) > 0:
                avg_activation = np.mean([ctx['activation'] for ctx in contexts])
                concept_stats.append((concept_idx, len(contexts), avg_activation))
        
        # Sort by number of contexts (descending)
        concept_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Show top concepts
        for concept_idx, num_contexts, avg_activation in concept_stats[:top_k]:
            print(f"\n  Concept {concept_idx} ({num_contexts} contexts, avg_act: {avg_activation:.3f}):")
            
            # Get top contexts by activation
            contexts = block_concepts[concept_idx]
            sorted_contexts = sorted(contexts, key=lambda x: x['activation'], reverse=True)
            
            for i, ctx in enumerate(sorted_contexts[:3]):  # Show top 3 contexts
                print(f"    {i+1}. '{ctx['context']}' (act: {ctx['activation']:.3f})")

def main():
    # Find the latest analysis results
    analysis_dir = Path("results/analysis/concept_analysis")
    if not analysis_dir.exists():
        print("❌ No analysis directory found!")
        return
    
    # Find the most recent analysis
    analysis_dirs = list(analysis_dir.glob("cbt_model_*"))
    if not analysis_dirs:
        print("❌ No analysis results found!")
        return
    
    latest_dir = max(analysis_dirs, key=lambda x: x.stat().st_mtime)
    results_file = latest_dir / "concept_analysis_results.json"
    
    print(f"📁 Using results from: {latest_dir}")
    
    if not results_file.exists():
        print("❌ Results file not found!")
        return
    
    # Extract contexts
    concept_contexts = extract_contexts_from_results(results_file)
    if concept_contexts is None:
        return
    
    # Display contexts
    display_top_contexts(concept_contexts)

if __name__ == "__main__":
    main() 
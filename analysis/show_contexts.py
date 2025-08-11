#!/usr/bin/env python3
"""
Show actual concept contexts from CBT analysis results.
"""

import json
from pathlib import Path

def show_concept_contexts(results_path):
    """Show actual contexts for some concepts."""
    
    with open(results_path / "concept_analysis_results.json", 'r') as f:
        results = json.load(f)
    
    print("🔍 Concept Contexts Analysis")
    print("=" * 50)
    
    # Check if we have contexts
    if 'concept_contexts' not in results:
        print("❌ No concept contexts found in results!")
        print("The analysis didn't save the actual contexts.")
        print("We only have labels and statistics.")
        return
    
    concept_contexts = results['concept_contexts']
    concept_labels = results['concept_labels']
    
    print("✅ Found concept contexts!")
    print()
    
    # Show contexts for some interesting concepts
    for block_name, block_concepts in concept_contexts.items():
        print(f"\n{block_name.upper()}:")
        print("-" * 30)
        
        # Get the most used concepts from this block
        if block_name in concept_labels:
            block_labels = concept_labels[block_name]
            sorted_concepts = sorted(block_labels.items(), 
                                   key=lambda x: x[1]['num_contexts'], 
                                   reverse=True)
            
            # Show top 3 concepts with their contexts
            for i, (concept_id, label_info) in enumerate(sorted_concepts[:3]):
                concept_id = int(concept_id)
                print(f"\n  Concept {concept_id}: {label_info['label']}")
                print(f"  Contexts: {label_info['num_contexts']:,}")
                print(f"  Avg activation: {label_info['avg_activation']:.3f}")
                
                # Show some actual contexts
                if concept_id in block_concepts:
                    contexts = block_concepts[concept_id]
                    print("  Sample contexts:")
                    
                    # Show top 5 contexts by activation
                    sorted_contexts = sorted(contexts, key=lambda x: x['activation'], reverse=True)
                    for j, ctx in enumerate(sorted_contexts[:5]):
                        print(f"    {j+1}. \"{ctx['context']}\" (activation: {ctx['activation']:.3f})")
                else:
                    print("    No contexts found for this concept")
                
                print()

def show_what_we_have(results_path):
    """Show what we actually have in the results."""
    
    with open(results_path / "concept_analysis_results.json", 'r') as f:
        results = json.load(f)
    
    print("📊 What We Actually Have:")
    print("=" * 30)
    
    print(f"Keys in results: {list(results.keys())}")
    print()
    
    if 'concept_contexts' in results:
        print("✅ We have concept contexts!")
        contexts = results['concept_contexts']
        total_contexts = sum(len(concept_contexts) 
                           for block in contexts.values() 
                           for concept_contexts in block.values())
        print(f"Total contexts: {total_contexts:,}")
    else:
        print("❌ No concept contexts saved")
    
    if 'concept_labels' in results:
        print("✅ We have concept labels!")
        labels = results['concept_labels']
        total_concepts = sum(len(block) for block in labels.values())
        print(f"Total concepts: {total_concepts}")
    
    if 'causality_results' in results:
        print("✅ We have causality results!")
    else:
        print("❌ No causality results")
    
    if 'specialization' in results:
        print("✅ We have specialization analysis!")
    else:
        print("❌ No specialization analysis")

def main():
    """Main function."""
    
    results_path = Path("results/analysis/concept_analysis/cbt_model_stab_kl_m32_k4_a30_20250810_223738")
    
    if not results_path.exists():
        print("❌ Results not found!")
        return
    
    show_what_we_have(results_path)
    print("\n" + "=" * 50)
    show_concept_contexts(results_path)

if __name__ == "__main__":
    main() 
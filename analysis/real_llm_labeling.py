#!/usr/bin/env python3
"""
Real LLM labeling of CBT concepts using GPT-4.
"""

import json
import openai
import os
from pathlib import Path
import time

def load_concept_contexts():
    """Load the concept contexts from the JSON file."""
    with open("top_concept_contexts.json", 'r') as f:
        return json.load(f)

def create_labeling_prompt(concept_info, contexts):
    """Create a prompt for GPT-4 to label a concept."""
    # Get top 10 contexts for analysis
    top_contexts = contexts[:10]
    
    # Clean up contexts (remove unicode artifacts)
    clean_contexts = []
    for ctx in top_contexts:
        context = ctx['context'].replace('\u0120', ' ').replace('\u00e6', 'æ').replace('\u013f', 'ı').replace('\u013e', 'ĭ').replace('\u00e7', 'ç').replace('\u0136', 'Ķ').replace('\u00ab', '«')
        clean_contexts.append(f"'{context}' (activation: {ctx['activation']:.3f})")
    
    contexts_text = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(clean_contexts)])
    
    prompt = f"""You are an expert in analyzing neural network concepts. I have a concept from a language model that activates on certain text patterns.

CONCEPT INFORMATION:
- Block: {concept_info['block']}
- Concept Index: {concept_info['concept_idx']}
- Current Label: {concept_info['label']}
- Impact: {concept_info['percent_change']:+.1f}% (how much removing this concept hurts model performance)
- Average Activation: {concept_info['avg_activation']:.3f}

TOP CONTEXTS WHERE THIS CONCEPT ACTIVATES:
{contexts_text}

TASK: Analyze these contexts and provide a clear, concise label for what this concept represents.

INSTRUCTIONS:
1. Look for the common pattern across all contexts
2. Identify what linguistic or semantic concept this represents
3. Provide a short, descriptive label (2-4 words)
4. Explain your reasoning in 1-2 sentences

EXAMPLES OF GOOD LABELS:
- "person_introduction" (for concepts that activate when introducing people)
- "punctuation_comma" (for concepts that activate on commas)
- "special_formatting" (for concepts that activate on special characters)
- "sentence_structure" (for concepts that activate on sentence boundaries)

Please provide your analysis in this format:
LABEL: [your label here]
REASONING: [your explanation here]
"""
    
    return prompt

def label_concept_with_gpt4(concept_info, contexts, api_key):
    """Use GPT-4 to label a concept."""
    client = openai.OpenAI(api_key=api_key)
    
    prompt = create_labeling_prompt(concept_info, contexts)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in neural network interpretability and linguistics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error calling GPT-4: {e}")
        return None

def parse_llm_response(response):
    """Parse the LLM response to extract label and reasoning."""
    if not response:
        return None, None
    
    lines = response.split('\n')
    label = None
    reasoning = None
    
    for line in lines:
        if line.startswith('LABEL:'):
            label = line.replace('LABEL:', '').strip()
        elif line.startswith('REASONING:'):
            reasoning = line.replace('REASONING:', '').strip()
    
    return label, reasoning

def main():
    print("🤖 REAL LLM LABELING OF CBT CONCEPTS")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Load concept contexts
    concept_data = load_concept_contexts()
    
    print(f"📊 Found {len(concept_data)} concepts to label")
    
    # Label each concept
    labeled_concepts = {}
    
    for concept_key, data in concept_data.items():
        concept_info = data['concept_info']
        contexts = data['contexts']
        
        print(f"\n🔍 Labeling {concept_key} ({concept_info['label']})...")
        print(f"   Impact: {concept_info['percent_change']:+.1f}% | Contexts: {len(contexts)}")
        
        # Get LLM label
        response = label_concept_with_gpt4(concept_info, contexts, api_key)
        
        if response:
            label, reasoning = parse_llm_response(response)
            
            if label:
                print(f"   ✅ New Label: {label}")
                print(f"   💭 Reasoning: {reasoning}")
                
                labeled_concepts[concept_key] = {
                    'concept_info': concept_info,
                    'old_label': concept_info['label'],
                    'new_label': label,
                    'reasoning': reasoning,
                    'contexts': contexts[:5]  # Save top 5 contexts
                }
            else:
                print(f"   ❌ Failed to parse response: {response}")
        else:
            print(f"   ❌ Failed to get response from GPT-4")
        
        # Rate limiting
        time.sleep(1)
    
    # Save results
    output_file = "real_llm_labels.json"
    with open(output_file, 'w') as f:
        json.dump(labeled_concepts, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 LABELING SUMMARY")
    print("=" * 60)
    
    for concept_key, data in labeled_concepts.items():
        print(f"{concept_key}:")
        print(f"  Old: {data['old_label']}")
        print(f"  New: {data['new_label']}")
        print(f"  Impact: {data['concept_info']['percent_change']:+.1f}%")
        print()

if __name__ == "__main__":
    main() 
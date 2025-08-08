"""
Enhanced concept labeling system for CBT.
"""

import re
from typing import List, Dict, Any
from collections import Counter
import json
from .llm_labeling import create_llm_labeler


class EnhancedConceptLabeler:
    """
    Enhanced concept labeling with sophisticated rules and LLM integration.
    """
    
    def __init__(self, use_llm=False, llm_provider="mock", llm_model="gpt-4", llm_api_key=None):
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.labels = {}
        
        # Initialize LLM labeler if needed
        if use_llm:
            self.llm_labeler = create_llm_labeler(
                provider=llm_provider,
                model_name=llm_model,
                api_key=llm_api_key
            )
        else:
            self.llm_labeler = None
        
        # Enhanced rule patterns
        self.patterns = {
            # Articles and determiners
            "definite_article": [r"\bthe\b", r"\bThe\b"],
            "indefinite_article": [r"\ba\b", r"\ban\b", r"\bA\b", r"\bAn\b"],
            "demonstrative": [r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b"],
            
            # Conjunctions
            "coordinating_conjunction": [r"\band\b", r"\bor\b", r"\bbut\b", r"\byet\b"],
            "subordinating_conjunction": [r"\bif\b", r"\bwhen\b", r"\bwhile\b", r"\bunless\b"],
            
            # Copulas and linking verbs
            "copula": [r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b", r"\bbe\b"],
            "linking_verb": [r"\bseems\b", r"\blooks\b", r"\bsounds\b", r"\bfeels\b", r"\bappears\b"],
            
            # Punctuation
            "period": [r"\."],
            "comma": [r","],
            "question_mark": [r"\?"],
            "exclamation": [r"!"],
            
            # Pronouns
            "personal_pronoun": [r"\bI\b", r"\byou\b", r"\bhe\b", r"\bshe\b", r"\bit\b", r"\bwe\b", r"\bthey\b"],
            "possessive_pronoun": [r"\bmy\b", r"\byour\b", r"\bhis\b", r"\bher\b", r"\bits\b", r"\bour\b", r"\btheir\b"],
            
            # Numbers and quantities
            "cardinal_number": [r"\b\d+\b", r"\bone\b", r"\btwo\b", r"\bthree\b", r"\bfour\b", r"\bfive\b"],
            "ordinal_number": [r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bfourth\b", r"\bfifth\b"],
            
            # Colors
            "color": [r"\bred\b", r"\bblue\b", r"\bgreen\b", r"\byellow\b", r"\bblack\b", r"\bwhite\b"],
            
            # Emotions and states
            "emotion": [r"\bhappy\b", r"\bsad\b", r"\bexcited\b", r"\btired\b", r"\bangry\b", r"\bscared\b"],
            
            # Actions
            "action_verb": [r"\bruns\b", r"\bwalks\b", r"\bjumps\b", r"\bflies\b", r"\bswims\b", r"\bsleeps\b"],
            
            # Time
            "temporal": [r"\btoday\b", r"\byesterday\b", r"\btomorrow\b", r"\bnow\b", r"\blater\b"],
            
            # Spatial
            "spatial": [r"\bon\b", r"\bin\b", r"\bunder\b", r"\babove\b", r"\bbelow\b", r"\bbetween\b"],
            
            # Questions
            "question_word": [r"\bwhat\b", r"\bwhere\b", r"\bwhen\b", r"\bwhy\b", r"\bhow\b", r"\bwho\b"],
            
            # Negation
            "negation": [r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnone\b", r"\bnothing\b"],
            
            # Modals
            "modal_verb": [r"\bcan\b", r"\bcould\b", r"\bwill\b", r"\bwould\b", r"\bshould\b", r"\bmust\b", r"\bmay\b"],
            
            # Quantifiers
            "quantifier": [r"\ball\b", r"\bsome\b", r"\bmany\b", r"\bfew\b", r"\bmost\b", r"\beach\b", r"\bevery\b"],
        }
    
    def label_concepts(self, mining_results: Dict[str, Any], max_contexts_per_concept: int = 10) -> Dict[str, str]:
        """
        Label concepts using enhanced rules and optionally LLM.
        
        Args:
            mining_results: Results from ConceptMiner
            max_contexts_per_concept: Maximum contexts to use for labeling
            
        Returns:
            Dictionary of concept labels
        """
        print("Labeling concepts with enhanced rules...")
        
        for concept_key, concept_data in mining_results.items():
            if not concept_data["top_contexts"]:
                continue
            
            # Get top contexts for labeling
            contexts = concept_data["top_contexts"][:max_contexts_per_concept]
            context_texts = [ctx["context"] for ctx in contexts]
            
            if self.use_llm and self.llm_api_key:
                label = self._label_with_llm(context_texts)
            else:
                label = self._label_with_enhanced_rules(context_texts, concept_data)
            
            self.labels[concept_key] = label
        
        return self.labels
    
    def _label_with_enhanced_rules(self, contexts: List[str], concept_data: Dict[str, Any]) -> str:
        """
        Enhanced rule-based concept labeling.
        """
        # Combine all contexts
        all_text = " ".join(contexts).lower()
        
        # Check each pattern category
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, all_text, re.IGNORECASE):
                    return category
        
        # Check token frequencies
        all_tokens = []
        for context in contexts:
            tokens = context.split()
            all_tokens.extend(tokens)
        
        token_counter = Counter(all_tokens)
        most_common = token_counter.most_common(3)
        
        # Check for specific token patterns
        if most_common:
            top_token = most_common[0][0]
            top_freq = most_common[0][1]
            
            # If a token appears frequently, it might be a concept
            if top_freq >= 3:
                # Clean the token (remove special characters)
                clean_token = re.sub(r'[^\w]', '', top_token)
                if clean_token:
                    return f"frequent_token_{clean_token}"
        
        # Check for semantic patterns
        semantic_patterns = {
            "comparison": ["bigger", "smaller", "faster", "slower", "better", "worse"],
            "possession": ["'s", "my", "your", "his", "her", "their"],
            "sequence": ["first", "second", "third", "next", "last"],
            "condition": ["if", "when", "unless", "since", "because"],
        }
        
        for category, words in semantic_patterns.items():
            if any(word in all_text for word in words):
                return category
        
        # Default label
        return "general_concept"
    
    def _label_with_llm(self, contexts: List[str]) -> str:
        """
        Label concept using LLM.
        """
        if self.llm_labeler is None:
            # Fallback to rule-based labeling
            return self._label_with_enhanced_rules(contexts, {})
        
        try:
            # Use the LLM labeler
            label = self.llm_labeler.label_concept(
                contexts=contexts,
                concept_idx=0,  # Will be set by the calling method
                block_idx=0,    # Will be set by the calling method
                max_contexts=10
            )
            return label
        except Exception as e:
            print(f"LLM labeling failed: {e}. Falling back to rule-based labeling.")
            return self._label_with_enhanced_rules(contexts, {})
    
    def get_label_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the assigned labels.
        """
        if not self.labels:
            return {}
        
        label_counts = Counter(self.labels.values())
        
        return {
            "total_concepts": len(self.labels),
            "unique_labels": len(label_counts),
            "label_distribution": dict(label_counts),
            "most_common_labels": label_counts.most_common(10)
        }
    
    def save_labels(self, filepath: str):
        """
        Save labels to a JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.labels, f, indent=2)
    
    def load_labels(self, filepath: str):
        """
        Load labels from a JSON file.
        """
        with open(filepath, 'r') as f:
            self.labels = json.load(f)


class ConceptLabelingDemo:
    """
    Demo for concept labeling.
    """
    
    def __init__(self):
        self.labeler = EnhancedConceptLabeler()
    
    def demo_labeling(self):
        """
        Demonstrate the labeling process with example contexts.
        """
        print("=== Concept Labeling Demo ===\n")
        
        # Example contexts for different concept types
        examples = {
            "definite_article": [
                "The [quick] brown fox",
                "The [lazy] dog",
                "The [red] car",
                "The [blue] sky"
            ],
            "copula": [
                "The sky [is] blue",
                "She [seems] happy",
                "They [are] students",
                "The food [tastes] delicious"
            ],
            "punctuation": [
                "Hello, [how] are you?",
                "Stop! [Don't] go there",
                "The [end].",
                "What [time] is it?"
            ],
            "conjunction": [
                "The weather is cold [and] windy",
                "She likes coffee [but] he prefers tea",
                "You can have cake [or] ice cream",
                "The movie was long [yet] entertaining"
            ]
        }
        
        for expected_label, contexts in examples.items():
            print(f"Contexts for '{expected_label}':")
            for ctx in contexts:
                print(f"  {ctx}")
            
            # Get label
            label = self.labeler._label_with_enhanced_rules(contexts, {})
            print(f"  → Labeled as: {label}")
            print(f"  → Correct: {label == expected_label}")
            print()


if __name__ == "__main__":
    # Run demo
    demo = ConceptLabelingDemo()
    demo.demo_labeling() 
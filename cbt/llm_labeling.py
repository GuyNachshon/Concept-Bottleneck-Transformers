"""
LLM-based concept labeling for CBT.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import anthropic
from anthropic import Anthropic
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMConceptLabeler:
    """
    Real LLM-based concept labeling using various providers.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize LLM concept labeler.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "local")
            model_name: Model to use for labeling
            api_key: API key (if None, will try to get from environment)
            max_retries: Maximum retries for API calls
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.provider = provider
        self.model_name = model_name
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        
        # Set up API client
        if provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
            
        elif provider == "anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
            self.client = Anthropic(api_key=api_key)
            
        elif provider == "local":
            # For local models (e.g., via Ollama)
            self.client = None
            self.base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def label_concept(
        self,
        contexts: List[str],
        concept_idx: int,
        block_idx: int,
        max_contexts: int = 10
    ) -> str:
        """
        Label a concept using LLM.
        
        Args:
            contexts: List of text contexts where concept activates
            concept_idx: Concept index
            block_idx: Block index
            max_contexts: Maximum contexts to include in prompt
            
        Returns:
            Generated label for the concept
        """
        # Limit contexts to avoid token limits
        contexts = contexts[:max_contexts]
        
        # Create prompt
        prompt = self._create_labeling_prompt(contexts, concept_idx, block_idx)
        
        # Get label from LLM
        label = self._query_llm(prompt)
        
        return label.strip()
    
    def _create_labeling_prompt(self, contexts: List[str], concept_idx: int, block_idx: int) -> str:
        """Create a prompt for concept labeling."""
        
        context_text = "\n".join([f"- {ctx}" for ctx in contexts])
        
        prompt = f"""You are analyzing a language model's internal representations. 

A concept (concept {concept_idx} in block {block_idx}) has been found to activate strongly in the following contexts:

{context_text}

Based on these contexts, what concept or semantic feature does this represent? 

Provide a concise, descriptive label (1-3 words) that captures the unifying theme or concept.

Examples of good labels:
- "weather conditions"
- "emotional states" 
- "spatial relationships"
- "temporal markers"
- "causal reasoning"

Label:"""
        
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50,
                        temperature=0.1
                    )
                    return response.choices[0].message.content
                    
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=50,
                        temperature=0.1,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
                    
                elif self.provider == "local":
                    # Local model via Ollama or similar
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.1}
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    return response.json()["response"]
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                else:
                    logger.error(f"All attempts failed for prompt: {prompt[:100]}...")
                    return "unknown_concept"
        
        return "unknown_concept"
    
    def batch_label_concepts(
        self,
        mining_results: Dict[str, Any],
        max_contexts_per_concept: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Label multiple concepts in batch.
        
        Args:
            mining_results: Results from concept mining
            max_contexts_per_concept: Max contexts per concept
            save_path: Path to save labels
            
        Returns:
            Dictionary mapping concept keys to labels
        """
        labels = {}
        
        for concept_key, concept_data in mining_results.items():
            # Handle both "contexts" and "top_contexts" formats
            contexts_key = "contexts" if "contexts" in concept_data else "top_contexts"
            
            if contexts_key in concept_data and concept_data[contexts_key]:
                logger.info(f"Labeling concept: {concept_key}")
                
                # Parse concept key to get indices
                parts = concept_key.split("_")
                if len(parts) >= 4:
                    block_idx = int(parts[1])
                    concept_idx = int(parts[3])
                    
                    # Extract context texts
                    if contexts_key == "top_contexts":
                        # Format: list of dicts with "context" key
                        context_texts = [ctx["context"] for ctx in concept_data[contexts_key]]
                    else:
                        # Format: list of strings
                        context_texts = concept_data[contexts_key]
                    
                    # Get label
                    label = self.label_concept(
                        contexts=context_texts,
                        concept_idx=concept_idx,
                        block_idx=block_idx,
                        max_contexts=max_contexts_per_concept
                    )
                    
                    labels[concept_key] = label
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
        
        # Save labels
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(labels, f, indent=2)
            logger.info(f"Labels saved to {save_path}")
        
        return labels


class MockLLMConceptLabeler:
    """
    Mock LLM labeler for testing without API calls.
    """
    
    def __init__(self):
        self.mock_labels = {
            "weather": ["weather", "climate", "temperature", "precipitation", "cold", "warm", "sunny", "rainy"],
            "emotion": ["emotion", "feeling", "mood", "sentiment", "happy", "sad", "excited", "tired", "angry", "scared"],
            "spatial": ["spatial", "location", "position", "direction", "on", "in", "under", "above", "below", "between"],
            "temporal": ["temporal", "time", "duration", "sequence", "today", "yesterday", "tomorrow", "now", "later"],
            "causal": ["causal", "reasoning", "cause", "effect", "because", "since", "therefore", "if", "then"],
            "action": ["action", "movement", "behavior", "activity", "run", "walk", "jump", "fly", "swim", "sleep"],
            "object": ["object", "entity", "thing", "item", "car", "book", "house", "tree", "animal"],
            "attribute": ["attribute", "property", "characteristic", "quality", "red", "blue", "big", "small", "fast", "slow"]
        }
    
    def label_concept(
        self,
        contexts: List[str],
        concept_idx: int,
        block_idx: int,
        max_contexts: int = 10
    ) -> str:
        """Mock labeling based on context keywords."""
        
        # Simple keyword-based labeling
        context_text = " ".join(contexts).lower()
        
        for category, keywords in self.mock_labels.items():
            if any(keyword in context_text for keyword in keywords):
                return f"{category}_{concept_idx}"
        
        return f"concept_{concept_idx}"
    
    def batch_label_concepts(
        self,
        mining_results: Dict[str, Any],
        max_contexts_per_concept: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Mock batch labeling."""
        
        labels = {}
        
        for concept_key, concept_data in mining_results.items():
            # Handle both "contexts" and "top_contexts" formats
            contexts_key = "contexts" if "contexts" in concept_data else "top_contexts"
            
            if contexts_key in concept_data and concept_data[contexts_key]:
                parts = concept_key.split("_")
                if len(parts) >= 4:
                    concept_idx = int(parts[3])
                    
                    # Extract context texts
                    if contexts_key == "top_contexts":
                        # Format: list of dicts with "context" key
                        context_texts = [ctx["context"] for ctx in concept_data[contexts_key]]
                    else:
                        # Format: list of strings
                        context_texts = concept_data[contexts_key]
                    
                    labels[concept_key] = self.label_concept(
                        contexts=context_texts,
                        concept_idx=concept_idx,
                        block_idx=0
                    )
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(labels, f, indent=2)
        
        return labels


def create_llm_labeler(
    provider: str = "mock",
    model_name: str = "gpt-4",
    api_key: Optional[str] = None
) -> LLMConceptLabeler:
    """
    Factory function to create LLM labeler.
    
    Args:
        provider: "openai", "anthropic", "local", or "mock"
        model_name: Model name
        api_key: API key
        
    Returns:
        LLMConceptLabeler instance
    """
    if provider == "mock":
        return MockLLMConceptLabeler()
    else:
        return LLMConceptLabeler(provider=provider, model_name=model_name, api_key=api_key)


# Example usage
if __name__ == "__main__":
    # Test with mock labeler
    labeler = create_llm_labeler(provider="mock")
    
    test_contexts = [
        "The weather is cold and rainy today.",
        "It's sunny and warm outside.",
        "The temperature dropped significantly."
    ]
    
    label = labeler.label_concept(test_contexts, concept_idx=0, block_idx=4)
    print(f"Generated label: {label}") 
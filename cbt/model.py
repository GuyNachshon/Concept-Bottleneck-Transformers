"""
CBT Model that wraps a base transformer and inserts concept layers.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from typing import List, Optional, Tuple, Dict, Any
from .concept_layer import ConceptLayer


class CBTModel(nn.Module):
    """
    Concept-Bottleneck Transformer model.
    
    Wraps a base transformer model and inserts concept layers into specified blocks.
    """
    
    def __init__(
        self,
        base_model_name: str = "gpt2",
        concept_blocks: Optional[List[int]] = None,
        d_model: int = 768,
        m: int = 64,
        k: int = 8,
        alpha: float = 0.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load base model
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.config = self.base_model.config
        
        # Concept layer parameters
        self.d_model = d_model
        self.m = m
        self.k = k
        self.alpha = alpha
        self.dropout = dropout
        
        # Determine which blocks to insert concept layers
        if concept_blocks is None:
            # Default: insert in middle 4 blocks
            num_blocks = self.config.n_layer
            start_block = num_blocks // 2 - 2
            self.concept_blocks = list(range(start_block, start_block + 4))
        else:
            self.concept_blocks = concept_blocks
        
        # Create concept layers for each specified block
        self.concept_layers = nn.ModuleDict()
        for block_idx in self.concept_blocks:
            self.concept_layers[f"block_{block_idx}"] = ConceptLayer(
                d_model=d_model,
                m=m,
                k=k,
                dropout=dropout
            )
        
        # Store concept activations during forward pass
        self.concept_activations = {}
        
    def set_alpha(self, alpha: float):
        """Set the bypass mixing parameter alpha."""
        self.alpha = alpha
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_concepts: bool = False,
        concept_edits: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional concept editing.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for loss computation
            return_concepts: Whether to return concept activations
            concept_edits: Dict of concept_key -> activation_value for editing
        """
        self.concept_activations = {}

        # Shortcut: if alpha == 0 and no edits, delegate to base model for numerical stability
        if self.alpha == 0.0 and concept_edits is None:
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=return_concepts,
                return_dict=True,
            )
            out = {
                "logits": base_outputs.logits,
            }
            if labels is not None and hasattr(base_outputs, "loss") and base_outputs.loss is not None:
                out["loss"] = base_outputs.loss
            if return_concepts:
                out["concept_activations"] = {}
                if hasattr(base_outputs, "hidden_states") and base_outputs.hidden_states is not None:
                    out["hidden_states"] = base_outputs.hidden_states[-1]
            return out
        hidden_states = self.base_model.transformer.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        hidden_states = hidden_states + self.base_model.transformer.wpe(position_ids)
        hidden_states = self.base_model.transformer.drop(hidden_states)

        for block_idx, block in enumerate(self.base_model.transformer.h):
            hidden_states = block(hidden_states)[0]  # GPT-2 handles attention internally
            if block_idx in self.concept_blocks:
                # Only invoke concept layer when alpha > 0 to avoid unstable early training
                if self.alpha > 0.0:
                    concept_layer = self.concept_layers[f"block_{block_idx}"]
                    hidden_states, concepts = concept_layer(hidden_states, alpha=self.alpha)
                    
                    # Apply concept edits if provided
                    if concept_edits is not None:
                        concepts = self._apply_concept_edits(concepts, block_idx, concept_edits)
                        # Reconstruct hidden states with edited concepts
                        hidden_states = (1 - self.alpha) * hidden_states + self.alpha * concept_layer.decoder(concepts)
                    
                    self.concept_activations[f"block_{block_idx}"] = concepts
                else:
                    # Alpha == 0: skip concept computation entirely
                    pass

        hidden_states = self.base_model.transformer.ln_f(hidden_states)
        lm_logits = self.base_model.lm_head(hidden_states)

        output = {"logits": lm_logits}
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            output["loss"] = loss
        
        if return_concepts:
            output["concept_activations"] = self.concept_activations
            # Store hidden states for reconstruction loss
            output["hidden_states"] = hidden_states
        
        return output

    def _apply_concept_edits(self, concepts: torch.Tensor, block_idx: int, concept_edits: Dict[str, float]) -> torch.Tensor:
        """
        Apply concept edits to the concept activations.
        
        Args:
            concepts: Concept activations tensor (batch_size, seq_len, m)
            block_idx: Current block index
            concept_edits: Dict mapping concept_key to new activation value
            
        Returns:
            Edited concept activations
        """
        edited_concepts = concepts.clone()
        
        for concept_key, new_value in concept_edits.items():
            # Parse concept key: "block_X_concept_Y"
            parts = concept_key.split("_")
            if len(parts) >= 4 and parts[0] == "block" and parts[2] == "concept":
                edit_block_idx = int(parts[1])
                concept_idx = int(parts[3])
                
                # Only apply if this is the correct block
                if edit_block_idx == block_idx:
                    edited_concepts[:, :, concept_idx] = new_value
        
        return edited_concepts
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        concept_edits: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text with optional concept editing.
        """
        if concept_edits is None:
            # Use standard generation without concept edits
            return self.base_model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                **kwargs
            )
        else:
            # Custom generation with concept edits
            return self._generate_with_concept_edits(
                input_ids,
                max_length,
                temperature,
                do_sample,
                pad_token_id,
                concept_edits,
                **kwargs
            )
    
    def _generate_with_concept_edits(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        do_sample: bool,
        pad_token_id: Optional[int],
        concept_edits: Dict[str, float],
        **kwargs
    ) -> torch.Tensor:
        """
        Custom generation that applies concept edits at each step.
        """
        batch_size = input_ids.shape[0]
        current_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass with concept edits
            outputs = self.forward(
                current_ids,
                return_concepts=True,
                concept_edits=concept_edits
            )
            
            # Get logits for next token
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            
            if do_sample:
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # Stop if we hit the pad token
            if pad_token_id is not None and (next_token == pad_token_id).any():
                break
        
        return current_ids
    
    def get_concept_activations(self) -> Dict[str, torch.Tensor]:
        """Get the concept activations from the last forward pass."""
        return self.concept_activations
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in base model vs concept layers."""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        concept_params = sum(p.numel() for p in self.concept_layers.parameters())
        
        return {
            "base_model": base_params,
            "concept_layers": concept_params,
            "total": base_params + concept_params
        } 
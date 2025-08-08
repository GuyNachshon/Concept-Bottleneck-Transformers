"""
Core Concept Layer implementation for CBT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def entmax15(inputs, dim=-1, n_iter=10):
    """
    EntMax 1.5 implementation for sparse attention.
    """
    with torch.no_grad():
        inputs = inputs / 2
        inputs -= inputs.max(dim=dim, keepdim=True)[0]

    for _ in range(n_iter):
        inputs = inputs / 2
        inputs -= inputs.max(dim=dim, keepdim=True)[0]
        
        exp_inputs = torch.exp(inputs)
        exp_sum = exp_inputs.sum(dim=dim, keepdim=True)
        exp_sq_sum = (exp_inputs ** 2).sum(dim=dim, keepdim=True)
        
        tau = exp_sum / exp_sq_sum
        inputs = inputs - tau

    return F.softmax(inputs, dim=dim)


class ConceptLayer(nn.Module):
    """
    Concept Layer that compresses hidden states into sparse concept vectors.
    
    Args:
        d_model: Hidden dimension of the transformer
        m: Number of concepts (concept dimension)
        k: Number of active concepts per token (top-k)
        dropout: Dropout rate for the concept layer
    """
    
    def __init__(self, d_model: int = 768, m: int = 64, k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.m = m
        self.k = k
        
        # Concept encoder: h -> concept logits
        self.encoder = nn.Linear(d_model, m, bias=True)
        
        # Concept decoder: concepts -> reconstructed hidden state
        self.decoder = nn.Linear(m, d_model, bias=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, h: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the concept layer.
        
        Args:
            h: Hidden states [batch_size, seq_len, d_model]
            alpha: Bypass mixing parameter (0 = no concept layer, 1 = full concept layer)
            
        Returns:
            h_out: Output hidden states
            concepts: Sparse concept vectors [batch_size, seq_len, m]
        """
        batch_size, seq_len, _ = h.shape
        
        # Encode to concept logits
        concept_logits = self.encoder(h)  # [batch_size, seq_len, m]
        
        # Apply entmax for sparse probabilities
        concept_probs = entmax15(concept_logits, dim=-1)  # [batch_size, seq_len, m]
        
        # Apply top-k sparsification
        if self.k < self.m:
            topk_vals, topk_idx = concept_probs.topk(self.k, dim=-1)
            concepts = torch.zeros_like(concept_probs)
            concepts.scatter_(-1, topk_idx, topk_vals)
        else:
            concepts = concept_probs
        
        # Apply dropout to concepts
        concepts = self.dropout(concepts)
        
        # Decode back to hidden state
        h_reconstructed = self.decoder(concepts)  # [batch_size, seq_len, d_model]
        
        # Bypass mixing: h' = (1-α)h + αh̃
        h_out = (1 - alpha) * h + alpha * h_reconstructed
        
        return h_out, concepts
    
    def get_concept_activations(self, h: torch.Tensor) -> torch.Tensor:
        """
        Get concept activations without reconstruction (for analysis).
        
        Args:
            h: Hidden states [batch_size, seq_len, d_model]
            
        Returns:
            concepts: Sparse concept vectors [batch_size, seq_len, m]
        """
        concept_logits = self.encoder(h)
        concept_probs = entmax15(concept_logits, dim=-1)
        
        if self.k < self.m:
            topk_vals, topk_idx = concept_probs.topk(self.k, dim=-1)
            concepts = torch.zeros_like(concept_probs)
            concepts.scatter_(-1, topk_idx, topk_vals)
        else:
            concepts = concept_probs
            
        return concepts 
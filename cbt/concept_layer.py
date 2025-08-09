"""
Core Concept Layer implementation for CBT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax (stable proxy for Entmax-1.5) along dimension dim.
    Reference: Martins & Astudillo (2016) - From Softmax to Sparsemax.
    """
    if dim < 0:
        dim = logits.dim() + dim

    # Shift by max for numerical stability
    z = logits - logits.max(dim=dim, keepdim=True)[0]

    # Sort and cumulative sums
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = z_sorted.cumsum(dim=dim)

    # rhos shape broadcastable to z_sorted along dim
    k_size = logits.size(dim)
    shape = [1] * logits.dim()
    shape[dim] = k_size
    rhos = torch.arange(1, k_size + 1, device=logits.device, dtype=logits.dtype).view(shape)

    # Determine support: z_sorted - (z_cumsum - 1)/rhos > 0
    threshold = (z_cumsum - 1) / rhos
    support = (z_sorted - threshold) > 0
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)

    # Compute tau = (sum_{j<=k} z_{(j)} - 1)/k
    tau = (z_cumsum.gather(dim, k - 1) - 1) / k.to(logits.dtype)

    # Sparsemax projection
    p = torch.clamp(z - tau, min=0.0)
    # Sanitize
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    # Renormalize in case of small numeric drift
    psum = p.sum(dim=dim, keepdim=True)
    p = torch.where(psum > 0, p / psum, torch.zeros_like(p))
    return p


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
        
        # Apply sparse activation (sparsemax proxy for entmax15)
        concept_probs = entmax15(concept_logits, dim=-1)  # [batch_size, seq_len, m]
        
        # Apply top-k sparsification
        if self.k < self.m:
            topk_vals, topk_idx = concept_probs.topk(self.k, dim=-1)
            concepts = torch.zeros_like(concept_probs)
            concepts.scatter_(-1, topk_idx, topk_vals)
        else:
            concepts = concept_probs
        
        # Sanitize and apply dropout to concepts
        concepts = torch.nan_to_num(concepts, nan=0.0, posinf=0.0, neginf=0.0)
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
        concepts = torch.nan_to_num(concepts, nan=0.0, posinf=0.0, neginf=0.0)
        return concepts
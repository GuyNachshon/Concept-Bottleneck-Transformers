"""
Advanced loss functions for CBT training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class OrthogonalityLoss:
    """
    Orthogonality loss to prevent concept duplication.
    
    Computes: λ₂‖W^D_ℓ^T W^D_ℓ - I‖_F^2
    """
    
    def __init__(self, weight: float = 0.01):
        self.weight = weight
    
    def __call__(self, decoder_weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute orthogonality loss for decoder weights.
        
        Args:
            decoder_weights: List of decoder weight matrices [W^D_ℓ]
            
        Returns:
            Orthogonality loss
        """
        total_loss = 0.0
        
        for W_D in decoder_weights:
            # W_D shape: [d_model, m]
            # Compute W^T W
            WtW = torch.mm(W_D.t(), W_D)  # [m, m]
            
            # Create identity matrix
            I = torch.eye(WtW.size(0), device=WtW.device, dtype=WtW.dtype)
            
            # Compute Frobenius norm of difference
            diff = WtW - I
            loss = torch.norm(diff, p='fro') ** 2
            
            total_loss += loss
        
        return self.weight * total_loss / len(decoder_weights)


class StabilityLoss:
    """
    Stability loss using Procrustes alignment to prevent concept ID shuffling.
    """
    
    def __init__(self, weight: float = 0.01, anchor_update_rate: float = 0.1):
        self.weight = weight
        self.anchor_update_rate = anchor_update_rate
        self.anchor_decoders = {}  # Store anchor decoder weights
    
    def update_anchors(self, decoder_weights: Dict[str, torch.Tensor]):
        """
        Update anchor decoder weights using exponential moving average.
        """
        for name, W_D in decoder_weights.items():
            if name not in self.anchor_decoders:
                self.anchor_decoders[name] = W_D.detach().clone()
            else:
                # Exponential moving average
                self.anchor_decoders[name] = (
                    (1 - self.anchor_update_rate) * self.anchor_decoders[name] +
                    self.anchor_update_rate * W_D.detach()
                )
    
    def procrustes_align(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute Procrustes alignment between matrices A and B.
        
        Args:
            A: Source matrix [d_model, m]
            B: Target matrix [d_model, m]
            
        Returns:
            Orthogonal transformation matrix
        """
        # Compute SVD of A^T B
        U, _, Vt = torch.svd(torch.mm(A.t(), B))
        
        # Return orthogonal transformation
        return torch.mm(U, Vt.t())
    
    def __call__(self, decoder_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute stability loss using Procrustes alignment.
        
        Args:
            decoder_weights: Dict of decoder weights {block_name: W_D}
            
        Returns:
            Stability loss
        """
        if not self.anchor_decoders:
            return torch.tensor(0.0, device=next(iter(decoder_weights.values())).device)
        
        total_loss = 0.0
        
        for name, W_D in decoder_weights.items():
            if name in self.anchor_decoders:
                anchor = self.anchor_decoders[name]
                
                # Compute Procrustes alignment
                Q = self.procrustes_align(W_D, anchor)
                
                # Compute alignment loss: ||W_D - anchor * Q^T||_F^2
                aligned_anchor = torch.mm(anchor, Q.t())
                loss = torch.norm(W_D - aligned_anchor, p='fro') ** 2
                
                total_loss += loss
        
        return self.weight * total_loss / len(decoder_weights)


class KLDistillationLoss:
    """
    KL divergence loss to maintain quality during α-ramp.
    
    Computes: KL(logits_CBT || logits_base)
    """
    
    def __init__(self, weight: float = 1.0, temperature: float = 1.0):
        self.weight = weight
        self.temperature = temperature
    
    def __call__(self, cbt_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between CBT and base model logits.
        
        Args:
            cbt_logits: Logits from CBT model
            base_logits: Logits from base model
            
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        cbt_logits = cbt_logits / self.temperature
        base_logits = base_logits / self.temperature
        
        # Compute softmax probabilities
        cbt_probs = F.softmax(cbt_logits, dim=-1)
        base_probs = F.softmax(base_logits, dim=-1)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        cbt_probs = cbt_probs + epsilon
        base_probs = base_probs + epsilon
        
        # Compute KL divergence: KL(base || cbt)
        kl_loss = torch.sum(base_probs * torch.log(base_probs / cbt_probs), dim=-1)
        
        return self.weight * torch.mean(kl_loss)


class ConceptDropoutLoss:
    """
    Concept dropout to ensure each concept learns a distinct, necessary role.
    """
    
    def __init__(self, dropout_rate: float = 0.1, weight: float = 0.01):
        self.dropout_rate = dropout_rate
        self.weight = weight
    
    def __call__(self, concept_activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply concept dropout and compute reconstruction loss.
        
        Args:
            concept_activations: Dict of concept activations {block_name: concepts}
            
        Returns:
            Concept dropout loss
        """
        total_loss = 0.0
        
        for concepts in concept_activations.values():
            # Create dropout mask
            batch_size, seq_len, num_concepts = concepts.shape
            dropout_mask = torch.bernoulli(
                torch.ones_like(concepts) * (1 - self.dropout_rate)
            )
            
            # Apply dropout to concepts
            dropped_concepts = concepts * dropout_mask
            
            # Compute reconstruction loss with dropped concepts
            # This encourages each concept to be necessary
            # (We'll need to reconstruct from the dropped concepts)
            # For now, we'll use a simple L2 loss on the dropped concepts
            loss = torch.mean((dropped_concepts - concepts) ** 2)
            
            total_loss += loss
        
        return self.weight * total_loss / len(concept_activations)


class AdvancedLossManager:
    """
    Manager for all advanced loss functions.
    """
    
    def __init__(
        self,
        orthogonality_weight: float = 0.01,
        stability_weight: float = 0.01,
        kl_weight: float = 1.0,
        concept_dropout_weight: float = 0.01,
        kl_temperature: float = 1.0,
        anchor_update_rate: float = 0.1,
        concept_dropout_rate: float = 0.1
    ):
        self.orthogonality_loss = OrthogonalityLoss(orthogonality_weight)
        self.stability_loss = StabilityLoss(stability_weight, anchor_update_rate)
        self.kl_loss = KLDistillationLoss(kl_weight, kl_temperature)
        self.concept_dropout_loss = ConceptDropoutLoss(concept_dropout_rate, concept_dropout_weight)
        
        # Track loss components for logging
        self.loss_components = {}
    
    def compute_losses(
        self,
        model,
        concept_activations: Dict[str, torch.Tensor],
        cbt_logits: Optional[torch.Tensor] = None,
        base_logits: Optional[torch.Tensor] = None,
        update_anchors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all advanced losses.
        
        Args:
            model: CBT model
            concept_activations: Concept activations from forward pass
            cbt_logits: Logits from CBT model (for KL loss)
            base_logits: Logits from base model (for KL loss)
            update_anchors: Whether to update stability anchors
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Get decoder weights from concept layers
        decoder_weights = []
        decoder_weights_dict = {}
        
        for name, layer in model.concept_layers.items():
            decoder_weights.append(layer.decoder.weight)
            decoder_weights_dict[name] = layer.decoder.weight
        
        # Orthogonality loss
        losses['orthogonality'] = self.orthogonality_loss(decoder_weights)
        
        # Stability loss
        if update_anchors:
            self.stability_loss.update_anchors(decoder_weights_dict)
        losses['stability'] = self.stability_loss(decoder_weights_dict)
        
        # KL distillation loss
        if cbt_logits is not None and base_logits is not None:
            losses['kl_distillation'] = self.kl_loss(cbt_logits, base_logits)
        else:
            losses['kl_distillation'] = torch.tensor(0.0, device=next(iter(concept_activations.values())).device)
        
        # Concept dropout loss
        losses['concept_dropout'] = self.concept_dropout_loss(concept_activations)
        
        # Store for logging
        self.loss_components = losses
        
        return losses
    
    def get_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute total advanced loss.
        
        Args:
            losses: Dictionary of loss components
            
        Returns:
            Total loss
        """
        return sum(losses.values())
    
    def get_loss_summary(self) -> Dict[str, float]:
        """
        Get summary of loss components for logging.
        
        Returns:
            Dictionary of loss values
        """
        return {name: loss.item() for name, loss in self.loss_components.items()} 
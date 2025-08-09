"""
Training utilities for CBT models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import wandb
from .model import CBTModel
from .advanced_losses import AdvancedLossManager
from torch.nn.utils import clip_grad_norm_


class CBTTrainer:
    """
    Trainer for Concept-Bottleneck Transformer models.
    """
    
    def __init__(
        self,
        model: CBTModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda",
        use_wandb: bool = False,
        project_name: str = "cbt-experiment",
        use_advanced_losses: bool = True,
        advanced_loss_config: Optional[Dict] = None,
        gradient_clip_max_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.use_wandb = use_wandb
        self.use_advanced_losses = use_advanced_losses
        self.gradient_clip_max_norm = gradient_clip_max_norm
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss weights
        self.loss_weights = {
            "task": 1.0,           # Language modeling loss
            "reconstruction": 0.1,  # Reconstruction loss
            "sparsity": 0.01,      # L1 sparsity loss
        }
        
        # Advanced loss manager
        if use_advanced_losses:
            if advanced_loss_config is None:
                advanced_loss_config = {}
            self.advanced_loss_manager = AdvancedLossManager(**advanced_loss_config)
        else:
            self.advanced_loss_manager = None
        
        # Base model for KL distillation (lazy loading)
        self.base_model = None
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(project=project_name, config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "m": model.m,
                "k": model.k,
                "concept_blocks": model.concept_blocks,
                "use_advanced_losses": use_advanced_losses
            })
    
    def compute_reconstruction_loss(self, hidden_states: torch.Tensor, concept_activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute reconstruction loss between original and reconstructed hidden states.
        """
        total_loss = 0.0
        num_blocks = len(concept_activations)
        
        for block_name, concepts in concept_activations.items():
            # Get the concept layer for this block
            concept_layer = self.model.concept_layers[block_name]
            
            # Reconstruct hidden states from concepts
            reconstructed = concept_layer.decoder(concepts)
            
            # Compute MSE loss
            loss = nn.MSELoss()(reconstructed, hidden_states)
            total_loss += loss
        
        return total_loss / num_blocks if num_blocks > 0 else torch.tensor(0.0, device=self.device)
    
    def compute_sparsity_loss(self, concept_activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute L1 sparsity loss on concept activations.
        """
        total_loss = 0.0
        num_blocks = len(concept_activations)
        
        for concepts in concept_activations.values():
            # L1 loss on concept activations
            loss = torch.mean(torch.abs(concepts))
            total_loss += loss
        
        return total_loss / num_blocks if num_blocks > 0 else torch.tensor(0.0, device=self.device)
    
    def compute_kl_loss(self, cbt_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between CBT and base model logits.
        """
        # Apply softmax to get probabilities
        cbt_probs = torch.softmax(cbt_logits, dim=-1)
        base_probs = torch.softmax(base_logits, dim=-1)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        cbt_probs = cbt_probs + epsilon
        base_probs = base_probs + epsilon
        
        # Compute KL divergence
        kl_loss = torch.sum(base_probs * torch.log(base_probs / cbt_probs), dim=-1)
        return torch.mean(kl_loss)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass with concept activations
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            return_concepts=True
        )
        
        # Extract losses and activations
        task_loss = outputs["loss"]
        if torch.isnan(task_loss) or torch.isinf(task_loss):
            return {
                "total_loss": float('nan'),
                "task_loss": float('nan'),
                "reconstruction_loss": float('nan'),
                "sparsity_loss": float('nan')
            }
        concept_activations = outputs["concept_activations"]
        
        # Compute basic losses
        reconstruction_loss = self.compute_reconstruction_loss(
            outputs.get("hidden_states", torch.zeros(1)), concept_activations
        )
        sparsity_loss = self.compute_sparsity_loss(concept_activations)
        
        # Initialize total loss with basic losses
        total_loss = (
            self.loss_weights["task"] * task_loss +
            self.loss_weights["reconstruction"] * reconstruction_loss +
            self.loss_weights["sparsity"] * sparsity_loss
        )
        
        # Compute advanced losses if enabled
        advanced_losses = {}
        if self.use_advanced_losses and self.advanced_loss_manager is not None:
            # Get base model logits for KL distillation
            base_logits = None
            if self.base_model is None:
                # Lazy load base model
                from transformers import GPT2LMHeadModel
                self.base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
                self.base_model.eval()
            
            with torch.no_grad():
                base_outputs = self.base_model(input_ids, attention_mask=attention_mask)
                base_logits = base_outputs.logits
            
            # Compute advanced losses
            advanced_losses = self.advanced_loss_manager.compute_losses(
                model=self.model,
                concept_activations=concept_activations,
                cbt_logits=outputs["logits"],
                base_logits=base_logits,
                update_anchors=True
            )
            
            # Add advanced losses to total
            advanced_total = self.advanced_loss_manager.get_total_loss(advanced_losses)
            total_loss += advanced_total
        
        # Backward pass
        total_loss.backward()
        if self.gradient_clip_max_norm is not None and self.gradient_clip_max_norm > 0:
            clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_max_norm)
        self.optimizer.step()
        
        # Prepare return dictionary
        loss_dict = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "sparsity_loss": sparsity_loss.item()
        }
        
        # Add advanced loss components
        if advanced_losses:
            for name, loss in advanced_losses.items():
                loss_dict[f"advanced_{name}"] = loss.item()
        
        return loss_dict
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation set.
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {"val_loss": avg_loss, "val_perplexity": torch.exp(torch.tensor(avg_loss)).item()}
    
    def train(
        self,
        num_epochs: int,
        alpha_schedule: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ):
        """
        Train the model with optional alpha scheduling.
        
        Args:
            num_epochs: Number of training epochs
            alpha_schedule: List of alpha values to use per epoch (for gradual ramp-up)
            save_path: Path to save the best model
        """
        best_val_loss = float('inf')
        
        # Default alpha schedule: gradual ramp from 0 to 1
        if alpha_schedule is None:
            alpha_schedule = np.linspace(0.0, 1.0, num_epochs).tolist()
        
        for epoch in range(num_epochs):
            # Set alpha for this epoch
            current_alpha = alpha_schedule[min(epoch, len(alpha_schedule) - 1)]
            self.model.set_alpha(current_alpha)
            
            # Training loop
            train_losses = []
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                loss_dict = self.train_step(batch)
                train_losses.append(loss_dict)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "alpha": f"{current_alpha:.2f}",
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "task": f"{loss_dict['task_loss']:.4f}"
                })
            
            # Compute average training losses
            avg_train_losses = {}
            for key in train_losses[0].keys():
                avg_train_losses[f"train_{key}"] = np.mean([loss[key] for loss in train_losses])
            
            # Evaluation
            val_metrics = self.evaluate()
            
            # Logging
            metrics = {**avg_train_losses, **val_metrics, "alpha": current_alpha}
            
            if self.use_wandb:
                wandb.log(metrics, step=epoch)
            
            print(f"Epoch {epoch+1}: alpha={current_alpha:.2f}, "
                  f"train_loss={avg_train_losses.get('train_total_loss', 0):.4f}, "
                  f"val_loss={val_metrics.get('val_loss', 0):.4f}")
            
            # Save best model
            if save_path and val_metrics.get('val_loss', float('inf')) < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'alpha': current_alpha,
                    'val_loss': best_val_loss
                }, save_path)
                print(f"Saved best model with val_loss: {best_val_loss:.4f}")
    
    def get_concept_statistics(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Compute statistics about concept usage across a dataset.
        """
        self.model.eval()
        concept_stats = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing concept statistics"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_concepts=True
                )
                
                concept_activations = outputs["concept_activations"]
                
                for block_name, concepts in concept_activations.items():
                    if block_name not in concept_stats:
                        concept_stats[block_name] = {
                            "activations": [],
                            "sparsity": []
                        }
                    
                    # Store activations
                    concept_stats[block_name]["activations"].append(concepts.cpu())
                    
                    # Compute sparsity (number of non-zero concepts per token)
                    sparsity = (concepts > 0).sum(dim=-1).float().mean(dim=0)
                    concept_stats[block_name]["sparsity"].append(sparsity.cpu())
        
        # Aggregate statistics
        for block_name in concept_stats:
            activations = torch.cat(concept_stats[block_name]["activations"], dim=0)
            sparsity = torch.cat(concept_stats[block_name]["sparsity"], dim=0)
            
            concept_stats[block_name] = {
                "mean_activation": activations.mean(dim=(0, 1)).tolist(),
                "std_activation": activations.std(dim=(0, 1)).tolist(),
                "mean_sparsity": sparsity.mean().item(),
                "std_sparsity": sparsity.std().item(),
                "total_tokens": activations.shape[0] * activations.shape[1]
            }
        
        return concept_stats 
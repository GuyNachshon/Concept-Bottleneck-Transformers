"""
Configuration management for CBT experiments.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    base_model_name: str = "gpt2"
    concept_blocks: list = field(default_factory=lambda: [4, 5, 6, 7])
    m: int = 32
    k: int = 4
    alpha: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 5
    gradient_clip_max_norm: float = 0.5
    use_mixed_precision: bool = True
    freeze_base_until_alpha: float = 1.0
    alpha_schedule: list = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20])
    
    # Loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "task": 1.0,
        "reconstruction": 1.0,
        "sparsity": 0.1
    })


@dataclass
class AdvancedLossesConfig:
    """Advanced losses configuration."""
    enabled: bool = True
    orthogonality_weight: float = 0.1
    stability_weight: float = 0.1
    kl_weight: float = 0.2
    dropout_weight: float = 0.05


@dataclass
class DataConfig:
    """Data configuration parameters."""
    dataset_name: str = "salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "train"
    max_length: int = 256
    num_workers: int = 0


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    num_samples: int = 200
    eval_interval: int = 1
    save_best: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    log_interval: int = 10
    save_interval: int = 1
    use_wandb: bool = False


@dataclass
class HardwareConfig:
    """Hardware configuration parameters."""
    device: str = "auto"
    deterministic: bool = True
    seed: int = 42


@dataclass
class CBTConfig:
    """Complete CBT configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    advanced_losses: AdvancedLossesConfig = field(default_factory=AdvancedLossesConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "CBTConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CBTConfig":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            advanced_losses=AdvancedLossesConfig(**config_dict.get("advanced_losses", {})),
            data=DataConfig(**config_dict.get("data", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            hardware=HardwareConfig(**config_dict.get("hardware", {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "advanced_losses": self.advanced_losses.__dict__,
            "data": self.data.__dict__,
            "evaluation": self.evaluation.__dict__,
            "logging": self.logging.__dict__,
            "hardware": self.hardware.__dict__
        }
    
    def save(self, config_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def get_device(self) -> str:
        """Get the device to use for training."""
        if self.hardware.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.hardware.device


def load_config(config_path: Optional[str] = None) -> CBTConfig:
    """Load configuration from file or use defaults."""
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            "configs/training.yaml",
            "../configs/training.yaml",
            "training.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        return CBTConfig.from_yaml(config_path)
    else:
        # Return default configuration
        return CBTConfig()


def create_experiment_config(
    base_model_name: str = "gpt2",
    concept_blocks: list = None,
    m: int = 32,
    k: int = 4,
    alpha: float = 0.2,
    learning_rate: float = 5e-5,
    num_epochs: int = 5,
    **kwargs
) -> CBTConfig:
    """Create a configuration for a specific experiment."""
    if concept_blocks is None:
        concept_blocks = [4, 5, 6, 7]
    
    config = CBTConfig()
    
    # Update model config
    config.model.base_model_name = base_model_name
    config.model.concept_blocks = concept_blocks
    config.model.m = m
    config.model.k = k
    config.model.alpha = alpha
    
    # Update training config
    config.training.learning_rate = learning_rate
    config.training.num_epochs = num_epochs
    
    # Update with any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.advanced_losses, key):
            setattr(config.advanced_losses, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.evaluation, key):
            setattr(config.evaluation, key, value)
        elif hasattr(config.logging, key):
            setattr(config.logging, key, value)
        elif hasattr(config.hardware, key):
            setattr(config.hardware, key, value)
    
    return config 
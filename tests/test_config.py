"""
Test configuration management.
"""

import pytest
import tempfile
import os
from cbt import CBTConfig, load_config, create_experiment_config


class TestCBTConfig:
    """Test CBT configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = CBTConfig()
        
        assert config.model.base_model_name == "gpt2"
        assert config.model.concept_blocks == [4, 5, 6, 7]
        assert config.model.m == 32
        assert config.model.k == 4
        assert config.model.alpha == 0.2
        
        assert config.training.batch_size == 4
        assert config.training.learning_rate == 5e-5
        assert config.training.num_epochs == 5
        
        assert config.advanced_losses.enabled is True
        assert config.advanced_losses.kl_weight == 0.2
    
    def test_config_modification(self):
        """Test configuration modification."""
        config = CBTConfig()
        
        # Modify config
        config.model.m = 64
        config.model.k = 8
        config.training.learning_rate = 1e-4
        
        assert config.model.m == 64
        assert config.model.k == 8
        assert config.training.learning_rate == 1e-4
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = CBTConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "training" in config_dict
        assert "advanced_losses" in config_dict
        
        # Check some values
        assert config_dict["model"]["base_model_name"] == "gpt2"
        assert config_dict["model"]["m"] == 32
    
    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {
            "model": {
                "base_model_name": "gpt2-medium",
                "concept_blocks": [2, 3, 4, 5],
                "m": 64,
                "k": 8,
                "alpha": 0.3
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-4,
                "num_epochs": 10
            }
        }
        
        config = CBTConfig.from_dict(config_dict)
        
        assert config.model.base_model_name == "gpt2-medium"
        assert config.model.concept_blocks == [2, 3, 4, 5]
        assert config.model.m == 64
        assert config.model.k == 8
        assert config.model.alpha == 0.3
        
        assert config.training.batch_size == 8
        assert config.training.learning_rate == 1e-4
        assert config.training.num_epochs == 10
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config = CBTConfig()
        config.model.m = 64
        config.training.learning_rate = 1e-4
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save(f.name)
            temp_path = f.name
        
        try:
            # Load from file
            loaded_config = CBTConfig.from_yaml(temp_path)
            
            assert loaded_config.model.m == 64
            assert loaded_config.training.learning_rate == 1e-4
            assert loaded_config.model.base_model_name == "gpt2"
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_create_experiment_config(self):
        """Test experiment configuration creation."""
        config = create_experiment_config(
            base_model_name="gpt2-medium",
            concept_blocks=[2, 3, 4, 5],
            m=64,
            k=8,
            alpha=0.3,
            learning_rate=1e-4,
            num_epochs=10
        )
        
        assert config.model.base_model_name == "gpt2-medium"
        assert config.model.concept_blocks == [2, 3, 4, 5]
        assert config.model.m == 64
        assert config.model.k == 8
        assert config.model.alpha == 0.3
        
        assert config.training.learning_rate == 1e-4
        assert config.training.num_epochs == 10
    
    def test_device_detection(self):
        """Test device detection."""
        config = CBTConfig()
        
        # Test auto device detection
        device = config.get_device()
        assert device in ["cuda", "cpu", "mps"]
        
        # Test explicit device
        config.hardware.device = "cpu"
        assert config.get_device() == "cpu" 
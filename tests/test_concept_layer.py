"""
Test concept layer functionality.
"""

import pytest
import torch
from cbt import ConceptLayer


class TestConceptLayer:
    """Test concept layer functionality."""
    
    def test_concept_layer_creation(self):
        """Test concept layer creation."""
        d_model = 768
        m = 32
        k = 4
        
        layer = ConceptLayer(d_model=d_model, m=m, k=k)
        
        assert layer is not None
        assert isinstance(layer, ConceptLayer)
        assert layer.m == m
        assert layer.k == k
    
    def test_concept_layer_parameters(self):
        """Test concept layer parameter counting."""
        d_model = 768
        m = 32
        k = 4
        
        layer = ConceptLayer(d_model=d_model, m=m, k=k)
        
        # Count parameters
        total_params = sum(p.numel() for p in layer.parameters())
        
        # Expected: encoder (d_model * m + m) + decoder (m * d_model + d_model)
        expected_params = d_model * m + m + m * d_model + d_model
        expected_params = 768 * 32 + 32 + 32 * 768 + 768  # 49,152 + 32 + 49,152 + 768 = 99,104
        
        assert total_params == expected_params
        print(f"Concept layer parameters: {total_params:,}")
    
    def test_concept_layer_forward(self):
        """Test concept layer forward pass."""
        d_model = 768
        m = 32
        k = 4
        batch_size = 2
        seq_len = 10
        
        layer = ConceptLayer(d_model=d_model, m=m, k=k)
        
        # Create input
        h = torch.randn(batch_size, seq_len, d_model)
        alpha = 0.5
        
        # Forward pass
        h_tilde, c = layer(h, alpha)
        
        # Check shapes
        assert h_tilde.shape == (batch_size, seq_len, d_model)
        assert c.shape == (batch_size, seq_len, m)
        
        # Check that c is sparse (only k non-zero values per token)
        non_zero_per_token = (c > 0).sum(dim=-1)
        assert torch.all(non_zero_per_token <= k)
    
    def test_concept_layer_sparsity(self):
        """Test that concept layer produces sparse activations."""
        d_model = 768
        m = 32
        k = 4
        batch_size = 2
        seq_len = 10
        
        layer = ConceptLayer(d_model=d_model, m=m, k=k)
        
        # Create input
        h = torch.randn(batch_size, seq_len, d_model)
        alpha = 0.5
        
        # Forward pass
        h_tilde, c = layer(h, alpha)
        
        # Check sparsity
        sparsity = (c == 0).float().mean()
        print(f"Sparsity: {sparsity:.3f}")
        
        # Should be mostly sparse (most values should be 0)
        assert sparsity > 0.8  # At least 80% sparse
    
    def test_concept_layer_top_k(self):
        """Test that only top-k concepts are active."""
        d_model = 768
        m = 32
        k = 4
        batch_size = 2
        seq_len = 10
        
        layer = ConceptLayer(d_model=d_model, m=m, k=k)
        
        # Create input
        h = torch.randn(batch_size, seq_len, d_model)
        alpha = 0.5
        
        # Forward pass
        h_tilde, c = layer(h, alpha)
        
        # Check that exactly k concepts are active per token
        active_concepts = (c > 0).sum(dim=-1)
        assert torch.all(active_concepts == k)
    
    def test_concept_layer_alpha_schedule(self):
        """Test concept layer with different alpha values."""
        d_model = 768
        m = 32
        k = 4
        batch_size = 2
        seq_len = 10
        
        layer = ConceptLayer(d_model=d_model, m=m, k=k)
        
        # Create input
        h = torch.randn(batch_size, seq_len, d_model)
        
        # Test alpha = 0 (should return original input)
        h_tilde_0, c_0 = layer(h, alpha=0.0)
        assert torch.allclose(h_tilde_0, h, atol=1e-6)
        
        # Test alpha = 1 (should return reconstructed input)
        h_tilde_1, c_1 = layer(h, alpha=1.0)
        assert not torch.allclose(h_tilde_1, h, atol=1e-6)  # Should be different
    
    def test_concept_layer_device_placement(self):
        """Test concept layer device placement."""
        d_model = 768
        m = 32
        k = 4
        
        layer = ConceptLayer(d_model=d_model, m=m, k=k)
        
        # Test CPU
        layer_cpu = layer.to("cpu")
        assert next(layer_cpu.parameters()).device.type == "cpu"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            layer_cuda = layer.to("cuda")
            assert next(layer_cuda.parameters()).device.type == "cuda" 
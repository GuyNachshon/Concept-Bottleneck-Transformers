#!/usr/bin/env python3
"""
Simple test runner for CBT project.
"""

import sys
import os
import traceback
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_test_suite():
    """Run all test suites."""
    print("=" * 60)
    print("CBT TEST SUITE")
    print("=" * 60)
    
    test_suites = [
        ("Import Tests", test_imports),
        ("Model Tests", test_model),
        ("Config Tests", test_config),
        ("Concept Layer Tests", test_concept_layer),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for suite_name, test_suite in test_suites:
        print(f"\n{suite_name}")
        print("-" * len(suite_name))
        
        suite_tests, suite_passed, suite_failed = test_suite()
        total_tests += suite_tests
        passed_tests += suite_passed
        failed_tests.extend(suite_failed)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for test_name, error in failed_tests:
            print(f"  ❌ {test_name}: {error}")
    
    success = passed_tests == total_tests
    if success:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed.")
    
    return success


def test_imports():
    """Test all imports."""
    tests = [
        ("Core imports", test_core_imports),
        ("Transformer imports", test_transformer_imports),
        ("Torch imports", test_torch_imports),
    ]
    
    return run_tests(tests)


def test_model():
    """Test model functionality."""
    tests = [
        ("Model creation", test_model_creation),
        ("Model parameters", test_model_parameters),
        ("Model forward pass", test_model_forward_pass),
        ("Model with concepts", test_model_with_concepts),
        ("Model device placement", test_model_device_placement),
    ]
    
    return run_tests(tests)


def test_config():
    """Test configuration functionality."""
    tests = [
        ("Default config", test_default_config),
        ("Config modification", test_config_modification),
        ("Config serialization", test_config_serialization),
        ("Config deserialization", test_config_deserialization),
        ("Experiment config", test_experiment_config),
    ]
    
    return run_tests(tests)


def test_concept_layer():
    """Test concept layer functionality."""
    tests = [
        ("Concept layer creation", test_concept_layer_creation),
        ("Concept layer parameters", test_concept_layer_parameters),
        ("Concept layer forward", test_concept_layer_forward),
        ("Concept layer sparsity", test_concept_layer_sparsity),
        ("Concept layer top-k", test_concept_layer_top_k),
    ]
    
    return run_tests(tests)


def run_tests(tests):
    """Run a list of tests."""
    total = len(tests)
    passed = 0
    failed = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"  ✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_name}: {e}")
            failed.append((test_name, str(e)))
    
    return total, passed, failed


# Test functions
def test_core_imports():
    """Test core CBT imports."""
    from cbt import (
        CBTModel, CBTTrainer, CBTEvaluator, CBTConfig,
        ConceptLayer, OrthogonalityLoss, StabilityLoss,
        KLDistillationLoss, ConceptDropoutLoss, AdvancedLossManager,
        ConceptMiner, ConceptLabeler, ConceptVisualizer, ConceptAnalyzer,
        ConceptAblator, ConceptEditor, AblationAnalyzer,
        LLMConceptLabeler, MockLLMConceptLabeler, create_llm_labeler,
        get_wikitext_eval_texts, run_granularity_sweep,
        run_placement_study, run_cross_seed_stability_test,
        load_config, create_experiment_config
    )
    
    # Verify all imports are callable
    assert callable(CBTModel)
    assert callable(CBTTrainer)
    assert callable(CBTEvaluator)
    assert callable(CBTConfig)


def test_transformer_imports():
    """Test transformer imports."""
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    assert tokenizer is not None
    assert model is not None


def test_torch_imports():
    """Test torch imports."""
    import torch
    assert torch.__version__ is not None


def test_model_creation():
    """Test model creation."""
    from cbt import CBTModel
    
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.2
    )
    
    assert model is not None
    assert isinstance(model, CBTModel)


def test_model_parameters():
    """Test model parameter counting."""
    from cbt import CBTModel
    
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # GPT-2 has ~124M parameters, plus concept layers
    expected_base_params = 124_439_808  # GPT-2 parameters
    expected_concept_params = 4 * (768 * 32 + 32 * 768 + 32 + 768)  # ~397K
    expected_total = expected_base_params + expected_concept_params
    
    print(f"    Total parameters: {total_params:,}")
    print(f"    Expected total: {expected_total:,}")
    
    # Should be close to expected (allow for small differences)
    assert abs(total_params - expected_total) < 1000000  # Within 1M parameters
    
    # Should have reasonable number of parameters (not billions!)
    assert total_params < 200_000_000  # Less than 200M parameters
    assert total_params > 120_000_000  # More than 120M parameters


def test_model_forward_pass():
    """Test model forward pass."""
    from cbt import CBTModel
    import torch
    
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.2
    )
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, 50257)


def test_model_with_concepts():
    """Test model with concept activations."""
    from cbt import CBTModel
    import torch
    
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.2
    )
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_concepts=True)
    
    assert "logits" in outputs
    assert "concept_activations" in outputs
    assert "reconstruction_targets" in outputs


def test_model_device_placement():
    """Test model device placement."""
    from cbt import CBTModel
    import torch
    
    model = CBTModel(
        base_model_name="gpt2",
        concept_blocks=[4, 5, 6, 7],
        m=32,
        k=4,
        alpha=0.2
    )
    
    model_cpu = model.to("cpu")
    assert next(model_cpu.parameters()).device.type == "cpu"


def test_default_config():
    """Test default configuration."""
    from cbt import CBTConfig
    
    config = CBTConfig()
    
    assert config.model.base_model_name == "gpt2"
    assert config.model.concept_blocks == [4, 5, 6, 7]
    assert config.model.m == 32
    assert config.model.k == 4
    assert config.model.alpha == 0.2


def test_config_modification():
    """Test configuration modification."""
    from cbt import CBTConfig
    
    config = CBTConfig()
    config.model.m = 64
    config.model.k = 8
    
    assert config.model.m == 64
    assert config.model.k == 8


def test_config_serialization():
    """Test configuration serialization."""
    from cbt import CBTConfig
    
    config = CBTConfig()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert "model" in config_dict
    assert "training" in config_dict


def test_config_deserialization():
    """Test configuration deserialization."""
    from cbt import CBTConfig
    
    config_dict = {
        "model": {
            "base_model_name": "gpt2-medium",
            "concept_blocks": [2, 3, 4, 5],
            "m": 64,
            "k": 8,
            "alpha": 0.3
        }
    }
    
    config = CBTConfig.from_dict(config_dict)
    
    assert config.model.base_model_name == "gpt2-medium"
    assert config.model.m == 64
    assert config.model.k == 8


def test_experiment_config():
    """Test experiment configuration creation."""
    from cbt import create_experiment_config
    
    config = create_experiment_config(
        base_model_name="gpt2-medium",
        concept_blocks=[2, 3, 4, 5],
        m=64,
        k=8,
        alpha=0.3
    )
    
    assert config.model.base_model_name == "gpt2-medium"
    assert config.model.m == 64
    assert config.model.k == 8


def test_concept_layer_creation():
    """Test concept layer creation."""
    from cbt import ConceptLayer
    
    layer = ConceptLayer(d_model=768, m=32, k=4)
    
    assert layer is not None
    assert isinstance(layer, ConceptLayer)
    assert layer.m == 32
    assert layer.k == 4


def test_concept_layer_parameters():
    """Test concept layer parameter counting."""
    from cbt import ConceptLayer
    
    layer = ConceptLayer(d_model=768, m=32, k=4)
    
    total_params = sum(p.numel() for p in layer.parameters())
    expected_params = 768 * 32 + 32 + 32 * 768 + 768  # 99,104
    
    assert total_params == expected_params
    print(f"    Concept layer parameters: {total_params:,}")


def test_concept_layer_forward():
    """Test concept layer forward pass."""
    from cbt import ConceptLayer
    import torch
    
    layer = ConceptLayer(d_model=768, m=32, k=4)
    
    batch_size, seq_len = 2, 10
    h = torch.randn(batch_size, seq_len, 768)
    alpha = 0.5
    
    h_tilde, c = layer(h, alpha)
    
    assert h_tilde.shape == (batch_size, seq_len, 768)
    assert c.shape == (batch_size, seq_len, 32)


def test_concept_layer_sparsity():
    """Test concept layer sparsity."""
    from cbt import ConceptLayer
    import torch
    
    layer = ConceptLayer(d_model=768, m=32, k=4)
    
    batch_size, seq_len = 2, 10
    h = torch.randn(batch_size, seq_len, 768)
    alpha = 0.5
    
    h_tilde, c = layer(h, alpha)
    
    sparsity = (c == 0).float().mean()
    print(f"    Sparsity: {sparsity:.3f}")
    
    assert sparsity > 0.8  # At least 80% sparse


def test_concept_layer_top_k():
    """Test concept layer top-k sparsification."""
    from cbt import ConceptLayer
    import torch
    
    layer = ConceptLayer(d_model=768, m=32, k=4)
    
    batch_size, seq_len = 2, 10
    h = torch.randn(batch_size, seq_len, 768)
    alpha = 0.5
    
    h_tilde, c = layer(h, alpha)
    
    active_concepts = (c > 0).sum(dim=-1)
    assert torch.all(active_concepts == 4)  # Exactly k=4 concepts active


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1) 
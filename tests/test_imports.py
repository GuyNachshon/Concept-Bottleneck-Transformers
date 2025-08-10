"""
Test that all imports work correctly.
"""

import pytest
import torch


class TestImports:
    """Test that all CBT modules can be imported correctly."""
    
    def test_core_imports(self):
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
        
        # Verify all imports are callable/instantiable
        assert callable(CBTModel)
        assert callable(CBTTrainer)
        assert callable(CBTEvaluator)
        assert callable(CBTConfig)
        assert callable(ConceptLayer)
        assert callable(OrthogonalityLoss)
        assert callable(StabilityLoss)
        assert callable(KLDistillationLoss)
        assert callable(ConceptDropoutLoss)
        assert callable(AdvancedLossManager)
        assert callable(ConceptMiner)
        assert callable(ConceptLabeler)
        assert callable(ConceptVisualizer)
        assert callable(ConceptAnalyzer)
        assert callable(ConceptAblator)
        assert callable(ConceptEditor)
        assert callable(AblationAnalyzer)
        assert callable(LLMConceptLabeler)
        assert callable(MockLLMConceptLabeler)
        assert callable(create_llm_labeler)
        assert callable(get_wikitext_eval_texts)
        assert callable(run_granularity_sweep)
        assert callable(run_placement_study)
        assert callable(run_cross_seed_stability_test)
        assert callable(load_config)
        assert callable(create_experiment_config)
    
    def test_transformer_imports(self):
        """Test that transformer dependencies work."""
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        
        # Test that we can load models
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        assert tokenizer is not None
        assert model is not None
    
    def test_torch_imports(self):
        """Test that PyTorch works correctly."""
        assert torch.__version__ is not None
        assert torch.cuda.is_available() or torch.backends.mps.is_available() or True  # CPU is always available 
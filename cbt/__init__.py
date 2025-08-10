"""
Concept-Bottleneck Transformers (CBT)

A framework for adding sparse concept layers to transformer models
to create human-auditable, steerable concepts.
"""

from .concept_layer import ConceptLayer
from .model import CBTModel
from .trainer import CBTTrainer
from .advanced_losses import (
    OrthogonalityLoss,
    StabilityLoss,
    KLDistillationLoss,
    ConceptDropoutLoss,
    AdvancedLossManager
)
from .analyzer import (
    ConceptMiner,
    ConceptLabeler,
    ConceptVisualizer,
    ConceptAnalyzer
)
from .ablation_tools import (
    ConceptAblator,
    ConceptEditor,
    AblationAnalyzer
)
from .llm_labeling import (
    LLMConceptLabeler,
    MockLLMConceptLabeler,
    create_llm_labeler
)
from .evaluator import CBTEvaluator, get_wikitext_eval_texts
from .experiments import (
    run_granularity_sweep,
    run_placement_study,
    run_cross_seed_stability_test
)
from .config import (
    CBTConfig,
    ModelConfig,
    TrainingConfig,
    AdvancedLossesConfig,
    DataConfig,
    EvaluationConfig,
    LoggingConfig,
    HardwareConfig,
    load_config,
    create_experiment_config
)

__version__ = "0.1.0"
__all__ = [
    "ConceptLayer", 
    "CBTModel", 
    "CBTTrainer",
    "OrthogonalityLoss",
    "StabilityLoss", 
    "KLDistillationLoss",
    "ConceptDropoutLoss",
    "AdvancedLossManager",
    "ConceptMiner",
    "ConceptLabeler",
    "ConceptVisualizer",
    "ConceptAnalyzer",
    "ConceptAblator",
    "ConceptEditor",
    "AblationAnalyzer",
    "LLMConceptLabeler",
    "MockLLMConceptLabeler",
    "create_llm_labeler",
    "CBTEvaluator",
    "get_wikitext_eval_texts",
    "run_granularity_sweep",
    "run_placement_study",
    "run_cross_seed_stability_test",
    "CBTConfig",
    "ModelConfig",
    "TrainingConfig",
    "AdvancedLossesConfig",
    "DataConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "HardwareConfig",
    "load_config",
    "create_experiment_config"
] 
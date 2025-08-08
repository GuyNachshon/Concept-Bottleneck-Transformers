# CBT Implementation Guide

This is a minimal working implementation of Concept-Bottleneck Transformers (CBT) as described in the README.md.

## Core Components

### 1. ConceptLayer (`cbt/concept_layer.py`)
The core concept layer that compresses hidden states into sparse concept vectors:

```python
from cbt.concept_layer import ConceptLayer

# Create concept layer
concept_layer = ConceptLayer(
    d_model=768,  # Hidden dimension
    m=64,        # Number of concepts
    k=8,         # Active concepts per token
    dropout=0.1
)

# Forward pass
h_out, concepts = concept_layer(hidden_states, alpha=0.5)
```

**Key Features:**
- **EntMax 1.5 sparsification** for clean concept selection
- **Top-k selection** to limit active concepts per token
- **Bypass mixing** with α parameter for gradual integration
- **Reconstruction** from concepts back to hidden states

### 2. CBTModel (`cbt/model.py`)
Wraps a base transformer model and inserts concept layers:

```python
from cbt.model import CBTModel

# Create CBT model
model = CBTModel(
    base_model_name="gpt2",
    concept_blocks=[4, 5, 6, 7],  # Which blocks to insert concepts
    m=64, k=8, alpha=0.0
)

# Forward pass
outputs = model(input_ids, return_concepts=True)
concept_activations = outputs["concept_activations"]
```

**Key Features:**
- **Flexible insertion** into any transformer blocks
- **Preserves base model** functionality
- **Concept tracking** during forward pass
- **Parameter counting** for analysis

### 3. CBTTrainer (`cbt/training.py`)
Training utilities with multiple loss functions:

```python
from cbt.training import CBTTrainer

trainer = CBTTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=1e-4,
    use_advanced_losses=True,  # Enable advanced loss functions
    advanced_loss_config={
        "orthogonality_weight": 0.01,
        "stability_weight": 0.01,
        "kl_weight": 1.0,
        "concept_dropout_weight": 0.01
    }
)

# Train with alpha scheduling
alpha_schedule = [0.0, 0.25, 0.5, 0.75, 1.0]
trainer.train(num_epochs=5, alpha_schedule=alpha_schedule)
```

### 4. Advanced Losses (`cbt/advanced_losses.py`)
Sophisticated loss functions for better concept learning:

```python
from cbt.advanced_losses import AdvancedLossManager

# Individual loss functions
orthogonality_loss = OrthogonalityLoss(weight=0.01)
stability_loss = StabilityLoss(weight=0.01)
kl_loss = KLDistillationLoss(weight=1.0, temperature=1.0)
concept_dropout_loss = ConceptDropoutLoss(dropout_rate=0.1, weight=0.01)

# Or use the manager
loss_manager = AdvancedLossManager(
    orthogonality_weight=0.01,
    stability_weight=0.01,
    kl_weight=1.0,
    concept_dropout_weight=0.01
)
```

### 5. Concept Analysis (`cbt/concept_analysis.py`)
Comprehensive tools for analyzing and visualizing concepts:

```python
from cbt.concept_analysis import ConceptAnalyzer, ConceptVisualizer

# Complete analysis pipeline
analyzer = ConceptAnalyzer(model, tokenizer, device)
results = analyzer.analyze_concepts(
    dataloader,
    save_path="analysis_results",
    max_samples=1000
)

# Individual components
miner = ConceptMiner(model, tokenizer, device)
mining_results = miner.mine_concepts(dataloader, activation_threshold=0.01)

labeler = ConceptLabeler()
labels = labeler.label_concepts(mining_results)

visualizer = ConceptVisualizer()
fig = visualizer.plot_concept_heatmap(concept_activations)
```

### 6. Ablation Testing (`cbt/ablation_tools.py`)
Tools for concept ablation and editing:

```python
from cbt.ablation_tools import ConceptAblator, ConceptEditor, AblationAnalyzer

# Concept ablation
ablator = ConceptAblator(model, tokenizer, device)
results = ablator.measure_ablation_effect(
    test_texts, concept_keys, ablation_type="zero"
)

# Concept editing (unmocked)
editor = ConceptEditor(model, tokenizer, device)
editor.edit_concept_activation("block_4_concept_0", 0.5)
edited_text = editor.generate_with_edits("The weather is", max_length=20)

# Analysis and visualization
analyzer = AblationAnalyzer()
fig = analyzer.plot_ablation_effects(results, metric="perplexity")
```

### 7. LLM-based Concept Labeling (`cbt/llm_labeling.py`)
Real LLM integration for concept labeling:

```python
from cbt.llm_labeling import create_llm_labeler

# Create labeler (supports OpenAI, Anthropic, local models)
labeler = create_llm_labeler(provider="openai", model_name="gpt-4")

# Label concepts
labels = labeler.batch_label_concepts(mining_results, save_path="labels.json")

# Mock labeler for testing
mock_labeler = create_llm_labeler(provider="mock")
```

### 8. Evaluation (`cbt/evaluation.py`)
Comprehensive evaluation of CBT success criteria:

```python
from cbt.evaluation import CBTEvaluator, get_wikitext_eval_texts

# Get evaluation texts from WikiText dataset
eval_texts = get_wikitext_eval_texts(num_samples=20)

# Create evaluator
evaluator = CBTEvaluator(cbt_model, base_model, tokenizer, device)

# Evaluate all criteria
results = evaluator.evaluate_all_criteria(eval_texts)

# Individual evaluations
quality_results = evaluator.evaluate_quality(eval_texts)
sparsity_results = evaluator.evaluate_sparsity(eval_texts)
stability_results = evaluator.evaluate_stability(other_models)
causality_results = evaluator.evaluate_causality(eval_texts)
nameability_results = evaluator.evaluate_nameability(eval_texts)
```

### 9. Experiments (`cbt/experiments.py`)
Research experiments for CBT optimization:

```python
from cbt.experiments import (
    run_granularity_sweep,
    run_placement_study,
    run_cross_seed_stability_test
)

# Granularity sweep: test different m and k values
results = run_granularity_sweep(
    m_values=[32, 64, 128],
    k_values=[4, 8, 12],
    eval_texts=eval_texts
)

# Placement study: test early vs. mid vs. late blocks
results = run_placement_study(
    m=64, k=8,
    eval_texts=eval_texts
)

# Cross-seed stability test
results = run_cross_seed_stability_test(
    num_seeds=3,
    eval_texts=eval_texts
)
```

**Loss Functions:**
- **Task loss**: Standard language modeling cross-entropy
- **Reconstruction loss**: MSE between original and reconstructed hidden states
- **Sparsity loss**: L1 penalty on concept activations
- **✅ Orthogonality loss**: Prevent concept duplication (‖W^D^T W^D - I‖_F^2)
- **✅ Stability loss**: Procrustes alignment for concept consistency
- **✅ KL distillation**: Maintain quality during α-ramp (KL(base || cbt))
- **✅ Concept dropout**: Ensure distinct concept roles

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Basic Test
```bash
uv run python test_basic.py
```

### 3. Run Training Examples
```bash
# Basic training
uv run python examples/basic_training.py

# Advanced training with all loss functions
uv run python examples/advanced_training.py

# Concept analysis demo
uv run python examples/concept_analysis_demo.py

# Ablation testing demo
uv run python examples/ablation_demo.py

# Unmocked functionality demo
uv run python examples/unmocked_demo.py

# Success criteria evaluation demo
uv run python examples/evaluation_demo.py

# Experiments demos
uv run python examples/granularity_sweep_demo.py
uv run python examples/placement_study_demo.py
uv run python examples/comprehensive_experiments_demo.py

# WikiText training demo
uv run python examples/wikitext_training.py
```

## Training Strategy

The implementation follows the training schedule from the README:

1. **Warm-start** (α=0): Insert concept layers with no effect
2. **Reconstruction** (few epochs): Train E,D to copy h_ℓ
3. **Blend-in** (α: 0→1): Gradually ramp up concept layer influence
4. **Sparsify**: Increase sparsity penalties once quality plateaus

### Alpha Scheduling
```python
# Gradual ramp-up over 10 epochs
alpha_schedule = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
```

## Model Configuration

### Default Settings (GPT-2-small)
- **Base model**: GPT-2 (124M parameters)
- **Concept blocks**: Middle 4 blocks (layers 4-7)
- **Concepts per block**: 64 (m=64)
- **Active concepts**: 8 per token (k=8)
- **Sparsity**: ~12.5% (8/64)

### Parameter Counts
- **Base model**: ~124M parameters
- **Concept layers**: ~4M parameters (4 blocks × 64 × 768 × 2)
- **Total**: ~128M parameters

## Success Criteria

The implementation tracks these metrics:

1. **✅ Quality**: Perplexity comparison with base model (≤2% hit target)
2. **✅ Sparsity**: Active concepts per token (target: ≤8)
3. **Stability**: Concept usage consistency across runs (≥0.8 alignment)
4. **Causality**: Concept editing effects (large, targeted behavior deltas)
5. **Nameability**: Consistent concept labels (≥70% concepts labeled)

## Next Steps

This implementation now includes advanced loss functions, comprehensive concept analysis, ablation testing, evaluation, and experiments. Remaining additions:

1. **✅ Advanced losses**: Orthogonality, stability, KL distillation, concept dropout
2. **✅ Concept analysis**: Mining, labeling, visualization, clustering
3. **✅ Ablation testing**: Concept editing, ablation analysis, causality testing
4. **✅ Evaluation**: Success criteria evaluation (quality, sparsity, stability, causality, nameability)
5. **✅ Experiments**: Granularity sweeps, placement studies, cross-seed stability
6. **Scale up**: Support for larger models (7B+)

## Usage Examples

### Basic Concept Analysis
```python
# Get concept activations
outputs = model(input_ids, return_concepts=True)
concepts = outputs["concept_activations"]

# Analyze sparsity
for block_name, block_concepts in concepts.items():
    sparsity = (block_concepts > 0).float().mean()
    print(f"{block_name}: {sparsity:.3f} sparsity")
```

### Concept Statistics
```python
# Compute concept usage statistics
stats = trainer.get_concept_statistics(val_dataloader)
print(f"Mean sparsity: {stats['block_4']['mean_sparsity']:.2f}")

# Comprehensive concept analysis
analyzer = ConceptAnalyzer(model, tokenizer, device)
results = analyzer.analyze_concepts(dataloader, save_path="analysis")

# Get detailed concept report
report = analyzer.get_concept_report("block_4_concept_0")
print(f"Concept label: {report['label']}")
print(f"Top contexts: {report['top_contexts'][:3]}")
```

### Model Comparison
```python
# Compare with base model
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
base_outputs = base_model(input_ids)
cbt_outputs = model(input_ids)

# Compare perplexity
base_loss = base_outputs.loss
cbt_loss = cbt_outputs["loss"]
print(f"Quality hit: {(cbt_loss - base_loss) / base_loss * 100:.2f}%")
```

## Architecture Notes

The implementation inserts concept layers **after** each transformer block, following the residual stream:

```
h_ℓ → Transformer Block → Concept Layer → h_{ℓ+1}
```

This preserves the original transformer architecture while adding the concept bottleneck. The concept layer compresses the hidden state to m sparse channels and reconstructs it, with the bypass mixing parameter α controlling the influence.

## Limitations

This implementation now includes advanced loss functions, concept analysis, ablation testing, and unmocked functionality but still has some limitations:
- Limited to GPT-2 architecture (though easily extensible)
- Concept editing during generation works but could be optimized
- Real LLM API integration requires API keys
- Multiprocessing resource warnings in some environments (doesn't affect functionality)

The focus is on demonstrating the core concept and providing a foundation for further development. 
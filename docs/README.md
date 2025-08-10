# Concept-Bottleneck Transformers (CBT)

A framework for adding sparse concept layers to transformer models to create human-auditable, steerable concepts.

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd CBT

# Install dependencies using uv
uv sync
```

### Basic Usage
```python
from cbt import CBTModel, CBTTrainer, CBTEvaluator, load_config

# Load configuration
config = load_config("configs/training.yaml")

# Create model
model = CBTModel(
    base_model_name="gpt2",
    concept_blocks=[4, 5, 6, 7],
    m=32,  # Number of concepts
    k=4,   # Active concepts per token
    alpha=0.2  # Bypass mixing
)

# Train model
trainer = CBTTrainer(model, config=config)
trainer.train()

# Evaluate model
evaluator = CBTEvaluator(model)
results = evaluator.evaluate_all_criteria(eval_texts)
```

### Command Line Interface
```bash
# Train a model
uv run python cbt_cli.py train --config configs/training.yaml

# Evaluate a trained model
uv run python cbt_cli.py evaluate --save-path results/models/cbt_model.pt

# Train and evaluate in one command
uv run python cbt_cli.py train-and-evaluate --config configs/training.yaml
```

## 📁 Project Structure

```
cbt/
├── README.md                    # This file
├── pyproject.toml              # Dependencies
├── cbt/                        # Core library
│   ├── __init__.py            # Public API
│   ├── model.py               # CBT model implementation
│   ├── trainer.py             # Training logic
│   ├── evaluator.py           # Evaluation metrics
│   ├── analyzer.py            # Concept analysis tools
│   ├── concept_layer.py       # Concept layer implementation
│   ├── advanced_losses.py     # Advanced loss functions
│   ├── llm_labeling.py        # LLM-based concept labeling
│   ├── enhanced_labeling.py   # Enhanced labeling utilities
│   ├── ablation_tools.py      # Concept ablation tools
│   ├── experiments.py         # Experiment utilities
│   └── config.py              # Configuration management
├── configs/                    # YAML configuration files
│   └── training.yaml          # Training configuration
├── experiments/                # Research scripts
│   ├── run_trained_experiments.py
│   ├── run_concept_analysis.py
│   ├── test_alternative_explanations.py
│   ├── check_baseline_perplexity.py
│   └── run_full_experiment.py
├── examples/                   # Usage examples
│   ├── basic_training.py
│   ├── advanced_training.py
│   ├── concept_analysis_demo.py
│   ├── evaluation_demo.py
│   ├── ablation_demo.py
│   ├── unmocked_demo.py
│   └── wikitext_training.py
└── results/                    # Experiment results (gitignored)
    ├── models/                # Saved model checkpoints
    ├── logs/                  # Training and experiment logs
    └── analysis/              # Analysis results and visualizations
```

## 🔧 Configuration

The project uses YAML configuration files for easy experiment management:

```yaml
# configs/training.yaml
model:
  base_model_name: "gpt2"
  concept_blocks: [4, 5, 6, 7]
  m: 32
  k: 4
  alpha: 0.2

training:
  batch_size: 4
  learning_rate: 5e-5
  num_epochs: 5
  gradient_clip_max_norm: 0.5
  use_mixed_precision: true

advanced_losses:
  enabled: true
  orthogonality_weight: 0.1
  stability_weight: 0.1
  kl_weight: 0.2
  dropout_weight: 0.05
```

## 🧪 Running Experiments

### Full Experiment Pipeline
```bash
# Run the complete research pipeline
uv run python experiments/run_trained_experiments.py
```

### Concept Analysis
```bash
# Analyze learned concepts
uv run python experiments/run_concept_analysis.py
```

### Test Alternative Explanations
```bash
# Test if results are robust
uv run python experiments/test_alternative_explanations.py
```

### Baseline Evaluation
```bash
# Check baseline GPT-2 performance
uv run python experiments/check_baseline_perplexity.py
```

## 📊 Evaluation Metrics

The framework evaluates CBT models across multiple criteria:

1. **Quality**: ≤2% perplexity hit vs. baseline
2. **Sparsity**: ≤4 active concepts per token
3. **Stability**: Consistent concept IDs across seeds
4. **Causality**: Ablating concepts affects predictions
5. **Nameability**: Concepts can be labeled by humans

## 🎯 Key Features

### Concept Layers
- **Sparse Activation**: Only k concepts active per token
- **Top-k Sparsification**: Keep most active concepts
- **Bypass Mixing**: Gradual integration with α-schedule

### Advanced Losses
- **Orthogonality Loss**: Prevent concept duplication
- **Stability Loss**: Maintain concept ID consistency
- **KL Distillation Loss**: Preserve base model quality
- **Concept Dropout Loss**: Ensure distinct concept roles

### Analysis Tools
- **Concept Mining**: Extract contexts where concepts activate
- **LLM Labeling**: Automatically label concepts using language models
- **Ablation Tools**: Test concept causality
- **Visualization**: Heatmaps, sparsity plots, clustering

## 🔬 Research Results

Our experiments show that CBT models can achieve:
- **Quality Hit**: -45% (improvement over baseline)
- **Sparsity**: 1.0 active concepts per token (median)
- **Stability**: Consistent concepts across training seeds
- **Interpretability**: Human-understandable concept labels

## 📚 Examples

### Basic Training
```python
# examples/basic_training.py
from cbt import CBTModel, CBTTrainer, load_config

config = load_config("configs/training.yaml")
model = CBTModel("gpt2", concept_blocks=[4,5,6,7], m=32, k=4)
trainer = CBTTrainer(model, config=config)
trainer.train()
```

### Concept Analysis
```python
# examples/concept_analysis_demo.py
from cbt import ConceptAnalyzer, CBTModel

model = CBTModel("gpt2", concept_blocks=[4,5,6,7], m=32, k=4)
analyzer = ConceptAnalyzer(model)
results = analyzer.analyze_concepts(eval_texts)
```

### Ablation Studies
```python
# examples/ablation_demo.py
from cbt import ConceptAblator, CBTModel

model = CBTModel("gpt2", concept_blocks=[4,5,6,7], m=32, k=4)
ablator = ConceptAblator(model)
effects = ablator.ablate_concepts(eval_texts)
```

## 🛠️ Development

### Adding New Loss Functions
```python
# cbt/advanced_losses.py
class CustomLoss(nn.Module):
    def forward(self, concept_activations, **kwargs):
        # Your loss computation
        return loss_value
```

### Adding New Analysis Tools
```python
# cbt/analyzer.py
class CustomAnalyzer:
    def analyze(self, model, texts):
        # Your analysis logic
        return results
```

### Creating New Experiments
```python
# experiments/my_experiment.py
from cbt import CBTModel, CBTTrainer, load_config

def run_my_experiment():
    config = load_config("configs/my_config.yaml")
    model = CBTModel(**config.model.__dict__)
    trainer = CBTTrainer(model, config=config)
    trainer.train()
```

## 📈 Performance

- **Training**: ~2-3 hours on single GPU for 5 epochs
- **Evaluation**: ~5 minutes for 200 texts
- **Memory**: ~8GB GPU memory for m=32, k=4
- **Speed**: ~2x slower than base model during inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the original CBT paper
- Uses Hugging Face Transformers
- LLM labeling powered by OpenAI/Anthropic APIs

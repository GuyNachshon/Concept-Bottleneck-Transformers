# CBT Project Structure

## Overview
Simple, clean structure for Concept-Bottleneck Transformers research.

## Directory Structure
```
cbt/
├── README.md                    # Project overview
├── pyproject.toml              # Dependencies
├── cbt/                        # Core library
│   ├── __init__.py            # Public API
│   ├── model.py               # CBT model
│   ├── trainer.py             # Training logic
│   ├── evaluator.py           # Evaluation logic
│   ├── analyzer.py            # Concept analysis
│   ├── concept_layer.py       # Concept layer implementation
│   ├── advanced_losses.py     # Advanced loss functions
│   ├── llm_labeling.py        # LLM-based concept labeling
│   ├── enhanced_labeling.py   # Enhanced labeling utilities
│   ├── ablation_tools.py      # Concept ablation tools
│   └── experiments.py         # Experiment utilities
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

## Usage

### Basic Usage
```python
from cbt import CBTModel, CBTTrainer, CBTEvaluator

# Create model
model = CBTModel("gpt2", concept_blocks=[4,5,6,7], m=32, k=4)

# Train model
trainer = CBTTrainer(model, config="configs/training.yaml")
trainer.train()

# Evaluate model
evaluator = CBTEvaluator(model)
results = evaluator.evaluate()
```

### Running Experiments
```bash
# Run full experiment pipeline
python experiments/run_trained_experiments.py

# Run concept analysis
python experiments/run_concept_analysis.py

# Test alternative explanations
python experiments/test_alternative_explanations.py
```

### Saving Results
```python
# Save model
torch.save(model.state_dict(), "results/models/cbt_model.pt")

# Save analysis results
with open("results/analysis/concept_analysis.json", "w") as f:
    json.dump(analysis_results, f)
```

## Key Files

### Core Library (`cbt/`)
- **`model.py`**: CBT model implementation
- **`trainer.py`**: Training loop and logic
- **`evaluator.py`**: Evaluation metrics
- **`analyzer.py`**: Concept analysis tools
- **`concept_layer.py`**: Concept layer implementation
- **`advanced_losses.py`**: Advanced loss functions
- **`llm_labeling.py`**: LLM-based concept labeling
- **`ablation_tools.py`**: Concept ablation utilities

### Configuration (`configs/`)
- **`training.yaml`**: Training parameters

### Experiments (`experiments/`)
- **`run_trained_experiments.py`**: Main experiment runner
- **`run_concept_analysis.py`**: Concept analysis
- **`test_alternative_explanations.py`**: Test alternative hypotheses
- **`check_baseline_perplexity.py`**: Baseline perplexity checks

### Examples (`examples/`)
- **`basic_training.py`**: Basic training example
- **`advanced_training.py`**: Advanced training with all features
- **`concept_analysis_demo.py`**: Concept analysis demonstration
- **`unmocked_demo.py`**: Full unmocked demonstration

### Results (`results/`)
- **`models/`**: Saved model checkpoints and weights
- **`logs/`**: Training and experiment logs
- **`analysis/`**: Concept analysis results and visualizations

## Benefits
- **Simple**: Easy to understand and navigate
- **Clean**: No duplicate or unnecessary directories
- **Practical**: Works for research
- **Maintainable**: Easy to modify and extend
- **Organized**: Clear separation of code, configs, and results 
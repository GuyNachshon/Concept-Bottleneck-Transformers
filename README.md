# Concept-Bottleneck Transformers (CBT)

A framework for adding sparse concept layers to transformer models to create human-auditable, steerable concepts.

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Run concept analysis on latest model
python scripts/analysis/analyze_concepts.py

# List available models
python scripts/management/list_models.py

# Run experiments
python experiments/run_trained_experiments.py
```

## 📁 Project Structure

```
cbt/                          # Core library
├── model.py                  # CBT model implementation
├── trainer.py                # Training logic
├── evaluator.py              # Evaluation metrics
├── analyzer.py               # Concept analysis tools
├── concept_layer.py          # Concept layer implementation
├── advanced_losses.py        # Advanced loss functions
├── llm_labeling.py           # LLM-based concept labeling
├── ablation_tools.py         # Concept ablation tools
├── experiments.py            # Experiment utilities
└── config.py                 # Configuration management

configs/                      # Configuration files
├── training.yaml             # Training configuration
└── *.yaml                    # Other configs

experiments/                  # Research scripts
├── run_trained_experiments.py
├── run_full_experiment.py
└── check_baseline_perplexity.py

scripts/                      # Utility scripts
├── analysis/                 # Analysis scripts
│   ├── analyze_concepts.py
│   ├── run_concept_analysis.py
│   └── test_alternative_explanations.py
├── management/               # Management scripts
│   ├── organize_analysis.py
│   ├── manage_results.py
│   └── list_models.py
└── utils/                    # Utility scripts
    ├── cbt_cli.py
    └── run_tests.py

examples/                     # Usage examples
├── basic_training.py
├── advanced_training.py
├── concept_analysis_demo.py
└── *.py

docs/                         # Documentation
├── README.md                 # Main documentation
├── PROJECT_STRUCTURE.md      # Structure guide
├── guides/                   # How-to guides
└── api/                      # API documentation

tests/                        # Test files
├── unit/                     # Unit tests
├── integration/              # Integration tests
└── fixtures/                 # Test data

results/                      # Experiment results
├── analysis/                 # Analysis results
├── models/                   # Model files
└── experiments/              # Experiment outputs
```

## 🔧 Usage

### Analysis
```bash
# Analyze concepts in latest model
python scripts/analysis/analyze_concepts.py

# Analyze specific model
python scripts/analysis/analyze_concepts.py cbt_model_stab_kl_m32_k4_a30.pt

# Test alternative explanations
python scripts/analysis/test_alternative_explanations.py
```

### Management
```bash
# List all models
python scripts/management/list_models.py

# Manage results
python scripts/management/manage_results.py summary

# Organize results
python scripts/management/organize_analysis.py
```

### Experiments
```bash
# Run full experiment pipeline
python experiments/run_trained_experiments.py

# Run concept analysis
python experiments/run_concept_analysis.py
```

## 📚 Documentation

- [Main Guide](docs/README.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [API Documentation](docs/api/)
- [Examples](examples/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](docs/LICENSE) file for details.

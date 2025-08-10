#!/usr/bin/env python3
"""
Organize the main CBT project directory structure.
"""

import os
import shutil
import glob
from pathlib import Path

def create_project_structure():
    """Create the proper project directory structure."""
    structure = {
        "cbt": {},  # Core library (already exists)
        "configs": {},  # Configuration files
        "experiments": {},  # Research scripts
        "examples": {},  # Usage examples
        "scripts": {  # Utility scripts
            "analysis": {},
            "management": {},
            "utils": {}
        },
        "docs": {  # Documentation
            "api": {},
            "guides": {},
            "examples": {}
        },
        "tests": {  # Test files
            "unit": {},
            "integration": {},
            "fixtures": {}
        },
        "results": {  # Results (already exists)
            "analysis": {},
            "models": {},
            "experiments": {}
        }
    }
    
    # Create directories
    for category, subcats in structure.items():
        if isinstance(subcats, dict):
            for subcat in subcats:
                os.makedirs(f"{category}/{subcat}", exist_ok=True)
        else:
            os.makedirs(category, exist_ok=True)
    
    print("✅ Created project directory structure")

def organize_scripts():
    """Organize utility scripts into proper directories."""
    print("🔧 Organizing scripts...")
    
    # Scripts to organize
    script_moves = {
        # Analysis scripts
        "analyze_concepts.py": "scripts/analysis/",
        "run_concept_analysis.py": "scripts/analysis/",
        "test_alternative_explanations.py": "scripts/analysis/",
        
        # Management scripts
        "organize_analysis.py": "scripts/management/",
        "organize_project.py": "scripts/management/",
        "manage_results.py": "scripts/management/",
        "list_models.py": "scripts/management/",
        
        # Utility scripts
        "cbt_cli.py": "scripts/utils/",
        "run_tests.py": "scripts/utils/",
        
        # Experiment scripts (move to experiments)
        "run_trained_experiments.py": "experiments/",
        "run_full_experiment.py": "experiments/",
        "check_baseline_perplexity.py": "experiments/",
    }
    
    for script, target_dir in script_moves.items():
        if os.path.exists(script):
            target_path = os.path.join(target_dir, script)
            if not os.path.exists(target_path):
                shutil.move(script, target_path)
                print(f"  ✅ Moved {script} to {target_dir}")
            else:
                print(f"  ⚠️  {target_path} already exists, skipping {script}")

def organize_configs():
    """Organize configuration files."""
    print("🔧 Organizing configs...")
    
    # Move config files
    config_moves = {
        "training.yaml": "configs/",
        "*.yaml": "configs/",
        "*.yml": "configs/",
    }
    
    for pattern, target_dir in config_moves.items():
        files = glob.glob(pattern)
        for file in files:
            if os.path.isfile(file) and not file.startswith("configs/"):
                target_path = os.path.join(target_dir, os.path.basename(file))
                if not os.path.exists(target_path):
                    shutil.move(file, target_path)
                    print(f"  ✅ Moved {file} to {target_dir}")
                else:
                    print(f"  ⚠️  {target_path} already exists, skipping {file}")

def organize_docs():
    """Organize documentation files."""
    print("🔧 Organizing documentation...")
    
    # Move documentation files
    doc_moves = {
        "README.md": "docs/",
        "PROJECT_STRUCTURE.md": "docs/",
        "LICENSE": "docs/",
        "*.md": "docs/guides/",
    }
    
    for pattern, target_dir in doc_moves.items():
        files = glob.glob(pattern)
        for file in files:
            if os.path.isfile(file) and not file.startswith("docs/"):
                target_path = os.path.join(target_dir, os.path.basename(file))
                if not os.path.exists(target_path):
                    shutil.move(file, target_path)
                    print(f"  ✅ Moved {file} to {target_dir}")
                else:
                    print(f"  ⚠️  {target_path} already exists, skipping {file}")

def organize_experiments():
    """Organize experiment directories."""
    print("🔧 Organizing experiments...")
    
    # Move experiment result directories
    exp_patterns = [
        "trained_cbt_experiment_results_*",
        "cbt_experiment_results_*",
        "experiment_results_*"
    ]
    
    for pattern in exp_patterns:
        dirs = glob.glob(pattern)
        for exp_dir in dirs:
            if os.path.isdir(exp_dir):
                target_path = os.path.join("results/experiments", os.path.basename(exp_dir))
                if not os.path.exists(target_path):
                    shutil.move(exp_dir, target_path)
                    print(f"  ✅ Moved {exp_dir} to results/experiments/")
                else:
                    print(f"  ⚠️  {target_path} already exists, skipping {exp_dir}")

def create_main_readme():
    """Create a main README that points to the organized structure."""
    readme_content = """# Concept-Bottleneck Transformers (CBT)

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
"""
    
    with open("README.md", 'w') as f:
        f.write(readme_content)
    
    print("✅ Created main README.md")

def create_gitignore():
    """Create a comprehensive .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# Results and data
results/
*.log
*.json
*.npz
*.npy

# Experiment results
trained_cbt_experiment_results_*/
cbt_experiment_results_*/
experiment_results_*/
concept_analysis_*/
alternative_explanations_*/

# Model files
*.pt
*.pth
*.safetensors

# Logs
logs/
*.log

# Temporary files
*.tmp
*.temp
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local
.env.*.local

# API keys
api_keys.txt
secrets.json
"""
    
    with open(".gitignore", 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore")

def main():
    """Main organization function."""
    print("🗂️  CBT Project Organization")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Organize files
    organize_scripts()
    organize_configs()
    organize_docs()
    organize_experiments()
    
    # Create main files
    create_main_readme()
    create_gitignore()
    
    print("\n✅ Project organization complete!")
    print("\n📁 New organized structure:")
    print("  cbt/                    # Core library")
    print("  configs/                # Configuration files")
    print("  experiments/            # Research scripts")
    print("  scripts/                # Utility scripts")
    print("  │   ├── analysis/       # Analysis scripts")
    print("  │   ├── management/     # Management scripts")
    print("  │   └── utils/          # Utility scripts")
    print("  docs/                   # Documentation")
    print("  examples/               # Usage examples")
    print("  tests/                  # Test files")
    print("  results/                # Experiment results")
    
    print("\n💡 Next steps:")
    print("  python scripts/analysis/analyze_concepts.py    # Run analysis")
    print("  python scripts/management/list_models.py       # List models")
    print("  python experiments/run_trained_experiments.py  # Run experiments")

if __name__ == "__main__":
    main() 
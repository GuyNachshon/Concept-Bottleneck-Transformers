# Results Directory

This directory stores experiment results and outputs.

## Structure
- `models/` - Saved model checkpoints and weights
- `logs/` - Training and experiment logs
- `analysis/` - Concept analysis results, visualizations, and metrics

## Usage
```python
# Save model
torch.save(model.state_dict(), "results/models/cbt_model.pt")

# Save logs
with open("results/logs/experiment.log", "w") as f:
    f.write(log_data)

# Save analysis results
with open("results/analysis/concept_analysis.json", "w") as f:
    json.dump(analysis_results, f)
```

## Gitignore
Actual result files are gitignored to keep the repository clean.
Only the directory structure and this README are tracked. 
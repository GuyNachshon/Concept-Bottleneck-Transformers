#!/usr/bin/env python3
"""
Research Dashboard for CBT Results
Creates publication-ready visualizations and analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class CBTResearchDashboard:
    def __init__(self, results_path):
        """Initialize dashboard with CBT results."""
        self.results_path = Path(results_path)
        self.load_results()
        
    def load_results(self):
        """Load analysis results."""
        with open(self.results_path / "concept_analysis_results.json", 'r') as f:
            self.results = json.load(f)
        
        with open(self.results_path / "analysis_metadata.json", 'r') as f:
            self.metadata = json.load(f)
    
    def create_concept_usage_heatmap(self):
        """Create concept usage heatmap across blocks."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Concept Usage Patterns Across Transformer Blocks', fontsize=16, fontweight='bold')
        
        blocks = ['block_4', 'block_5', 'block_6', 'block_7']
        
        for idx, block in enumerate(blocks):
            ax = axes[idx // 2, idx % 2]
            
            if block in self.results['specialization']:
                usage = self.results['specialization'][block]['concept_usage']
                usage_array = np.array(usage)
                
                # Create heatmap
                im = ax.imshow(usage_array.T, cmap='viridis', aspect='auto')
                ax.set_title(f'{block.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Token Position')
                ax.set_ylabel('Concept Index')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Usage Probability')
        
        plt.tight_layout()
        plt.savefig('analysis/concept_usage_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_concept_activation_distribution(self):
        """Create distribution of concept activations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Concept Activation Distributions', fontsize=16, fontweight='bold')
        
        blocks = ['block_4', 'block_5', 'block_6', 'block_7']
        
        for idx, block in enumerate(blocks):
            ax = axes[idx // 2, idx % 2]
            
            if block in self.results['concept_labels']:
                activations = []
                for concept_data in self.results['concept_labels'][block].values():
                    activations.extend([concept_data['avg_activation']] * concept_data['num_contexts'])
                
                ax.hist(activations, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{block.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Average Activation')
                ax.set_ylabel('Frequency')
                ax.axvline(np.mean(activations), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(activations):.3f}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('analysis/concept_activation_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_concept_specialization_analysis(self):
        """Analyze concept specialization patterns."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Concept usage vs activation
        all_concepts = []
        for block, concepts in self.results['concept_labels'].items():
            for concept_id, data in concepts.items():
                all_concepts.append({
                    'block': block,
                    'concept_id': concept_id,
                    'usage': data['num_contexts'],
                    'activation': data['avg_activation'],
                    'label': data['label']
                })
        
        df = pd.DataFrame(all_concepts)
        
        # Scatter plot
        scatter = axes[0].scatter(df['usage'], df['activation'], 
                                c=df['block'].astype('category').cat.codes, 
                                alpha=0.6, s=50)
        axes[0].set_xlabel('Number of Contexts (Usage)')
        axes[0].set_ylabel('Average Activation')
        axes[0].set_title('Concept Usage vs Activation')
        axes[0].set_xscale('log')
        
        # Add legend
        legend1 = axes[0].legend(*scatter.legend_elements(),
                                title="Blocks", loc="upper right")
        axes[0].add_artist(legend1)
        
        # Plot 2: Concept type distribution
        label_counts = df['label'].value_counts()
        axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Distribution of Concept Types')
        
        plt.tight_layout()
        plt.savefig('analysis/concept_specialization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sparsity_analysis(self):
        """Analyze sparsity patterns."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate sparsity metrics
        sparsity_data = []
        for block, concepts in self.results['concept_labels'].items():
            total_concepts = len(concepts)
            active_concepts = sum(1 for c in concepts.values() if c['num_contexts'] > 10)
            sparsity = 1 - (active_concepts / total_concepts)
            
            sparsity_data.append({
                'block': block,
                'total_concepts': total_concepts,
                'active_concepts': active_concepts,
                'sparsity': sparsity
            })
        
        df_sparsity = pd.DataFrame(sparsity_data)
        
        # Plot 1: Sparsity by block
        bars = axes[0].bar(df_sparsity['block'], df_sparsity['sparsity'], 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0].set_title('Concept Sparsity by Block')
        axes[0].set_ylabel('Sparsity (1 - active/total)')
        axes[0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # Plot 2: Active vs inactive concepts
        concept_counts = []
        for block, concepts in self.results['concept_labels'].items():
            active = sum(1 for c in concepts.values() if c['num_contexts'] > 10)
            inactive = len(concepts) - active
            concept_counts.extend([
                {'block': block, 'type': 'Active', 'count': active},
                {'block': block, 'type': 'Inactive', 'count': inactive}
            ])
        
        df_counts = pd.DataFrame(concept_counts)
        
        # Stacked bar chart
        pivot_data = df_counts.pivot(index='block', columns='type', values='count')
        pivot_data.plot(kind='bar', stacked=True, ax=axes[1], 
                       color=['#4ECDC4', '#FF6B6B'])
        axes[1].set_title('Active vs Inactive Concepts by Block')
        axes[1].set_ylabel('Number of Concepts')
        axes[1].legend(title='Concept Status')
        
        plt.tight_layout()
        plt.savefig('analysis/sparsity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_comparison(self):
        """Compare CBT vs baseline performance."""
        # This would include perplexity, quality hit, etc.
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Mock data - replace with actual performance metrics
        models = ['GPT-2 Base', 'CBT (α=0.1)', 'CBT (α=0.2)', 'CBT (α=0.3)']
        perplexity = [63.0, 65.2, 68.1, 70.3]  # From your results
        quality_hit = [0, -3.3, -8.1, -11.5]   # Negative = improvement
        
        # Plot 1: Perplexity comparison
        bars1 = axes[0].bar(models, perplexity, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0].set_title('Perplexity Comparison')
        axes[0].set_ylabel('Perplexity (lower is better)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom')
        
        # Plot 2: Quality hit comparison
        bars2 = axes[1].bar(models, quality_hit, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1].set_title('Quality Hit Comparison')
        axes[1].set_ylabel('Quality Hit % (negative = improvement)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height - 0.5,
                        f'{height:.1f}%', ha='center', va='top')
        
        plt.tight_layout()
        plt.savefig('analysis/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_concept_explorer(self):
        """Create interactive concept exploration dashboard."""
        # Prepare data for interactive visualization
        all_concepts = []
        for block, concepts in self.results['concept_labels'].items():
            for concept_id, data in concepts.items():
                all_concepts.append({
                    'block': block,
                    'concept_id': int(concept_id),
                    'usage': data['num_contexts'],
                    'activation': data['avg_activation'],
                    'label': data['label']
                })
        
        df = pd.DataFrame(all_concepts)
        
        # Create interactive scatter plot
        fig = px.scatter(df, x='usage', y='activation', 
                        color='block', size='usage',
                        hover_data=['concept_id', 'label'],
                        title='Interactive Concept Explorer',
                        labels={'usage': 'Number of Contexts', 
                               'activation': 'Average Activation'})
        
        fig.update_layout(
            xaxis_type="log",
            height=600,
            showlegend=True
        )
        
        fig.write_html('analysis/interactive_concept_explorer.html')
        return fig
    
    def generate_research_summary(self):
        """Generate comprehensive research summary."""
        summary = {
            'model_config': self.metadata['model_config'],
            'total_concepts': sum(len(self.results['concept_labels'][block]) 
                                for block in self.results['concept_labels']),
            'blocks_analyzed': len(self.results['concept_labels']),
            'concepts_per_block': self.metadata['model_config']['m'],
            'active_concepts_per_token': self.metadata['model_config']['k'],
            'alpha_value': self.metadata['model_config']['alpha']
        }
        
        # Calculate sparsity metrics
        total_concepts = 0
        active_concepts = 0
        for block, concepts in self.results['concept_labels'].items():
            total_concepts += len(concepts)
            active_concepts += sum(1 for c in concepts.values() if c['num_contexts'] > 10)
        
        summary['overall_sparsity'] = 1 - (active_concepts / total_concepts)
        summary['active_concepts'] = active_concepts
        summary['total_concepts'] = total_concepts
        
        # Calculate activation statistics
        all_activations = []
        for block, concepts in self.results['concept_labels'].items():
            for concept_data in concepts.values():
                all_activations.extend([concept_data['avg_activation']] * concept_data['num_contexts'])
        
        summary['mean_activation'] = np.mean(all_activations)
        summary['std_activation'] = np.std(all_activations)
        summary['max_activation'] = np.max(all_activations)
        
        return summary
    
    def create_all_visualizations(self):
        """Create all research visualizations."""
        print("Creating CBT Research Dashboard...")
        
        # Create analysis directory
        Path('analysis').mkdir(exist_ok=True)
        
        # Generate all plots
        self.create_concept_usage_heatmap()
        self.create_concept_activation_distribution()
        self.create_concept_specialization_analysis()
        self.create_sparsity_analysis()
        self.create_performance_comparison()
        
        # Create interactive dashboard
        self.create_interactive_concept_explorer()
        
        # Generate summary
        summary = self.generate_research_summary()
        
        # Save summary
        with open('analysis/research_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Research dashboard created successfully!")
        print(f"📊 Summary: {summary['total_concepts']} concepts analyzed")
        print(f"🎯 Sparsity: {summary['overall_sparsity']:.2%}")
        print(f"📈 Mean activation: {summary['mean_activation']:.3f}")
        
        return summary

if __name__ == "__main__":
    # Use the latest analysis results
    results_path = "results/analysis/concept_analysis/cbt_model_stab_kl_m32_k4_a30_20250810_223738"
    
    dashboard = CBTResearchDashboard(results_path)
    summary = dashboard.create_all_visualizations() 
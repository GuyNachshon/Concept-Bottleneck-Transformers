#!/usr/bin/env python3
"""
Cross-Domain Testing for CBT Concepts
Tests if learned concepts generalize beyond entertainment to other domains.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict

# Import CBT modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cbt.model import CBTModel


@dataclass
class DomainTest:
    """Represents a domain test case"""
    name: str
    category: str  # 'science', 'politics', 'sports', etc.
    test_texts: List[str]  # Texts from this domain
    expected_activation: str  # 'high', 'medium', 'low', 'none'


class CBTDomainTester:
    """Tests CBT concepts across different domains"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load_model(model_path)
        self.concept_labels = self._load_concept_labels()
        
    def _load_model(self, model_path: str) -> Tuple[CBTModel, GPT2Tokenizer]:
        """Load CBT model and tokenizer"""
        print(f"🔍 Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model (hardcode config for now)
        model = CBTModel(
            base_model_name="gpt2",
            concept_blocks=[4, 5, 6, 7],
            m=32, k=4, alpha=0.3
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Model loaded successfully")
        return model, tokenizer
    
    def _load_concept_labels(self) -> Dict[str, str]:
        """Load concept labels from LLM labeling results"""
        labels_file = Path("real_llm_labels.json")
        if not labels_file.exists():
            print("⚠️  No concept labels found, using default labels")
            return {}
        
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        concept_labels = {}
        for concept_key, data in labels_data.items():
            concept_labels[concept_key] = data['new_label'].strip('"')
        
        print(f"✅ Loaded {len(concept_labels)} concept labels")
        return concept_labels
    
    def _get_concept_activations(self, text: str) -> Dict[str, np.ndarray]:
        """Get concept activations for a given text"""
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=256,
            padding=True
        ).to(self.device)
        
        # Get activations
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'], return_concepts=True)
            concept_activations = outputs['concept_activations']
        
        # Convert to numpy and organize by block
        activations_by_block = {}
        for block_name, activations in concept_activations.items():
            activations_by_block[block_name] = activations.cpu().numpy()
        
        return activations_by_block
    
    def _compute_concept_statistics(self, activations: np.ndarray) -> Dict[str, float]:
        """Compute statistics for concept activations"""
        return {
            'mean_activation': float(np.mean(activations)),
            'max_activation': float(np.max(activations)),
            'std_activation': float(np.std(activations)),
            'sparsity': float(np.mean(activations == 0)),
            'active_concepts': int(np.sum(activations > 0.1)),
            'strong_activations': int(np.sum(activations > 0.5))
        }
    
    def create_domain_tests(self) -> List[DomainTest]:
        """Create comprehensive domain test cases"""
        tests = []
        
        # Entertainment domain (baseline)
        entertainment_texts = [
            "Robert Boulter is an English film, television and theatre actor.",
            "He had a guest starring role on the television series The Bill in 2000.",
            "She is a talented actress who appeared in many films.",
            "The actor performed brilliantly in the theatre production.",
            "He is a film and television actor with many credits.",
            "She starred in the popular television series.",
            "The theatre actor received critical acclaim.",
            "He appeared in several guest roles on television.",
        ]
        tests.append(DomainTest("entertainment", "entertainment", entertainment_texts, "high"))
        
        # Science domain
        science_texts = [
            "The scientist conducted experiments in the laboratory.",
            "Einstein developed the theory of relativity.",
            "The research team published their findings in Nature.",
            "The physicist discovered a new particle.",
            "The chemist synthesized a novel compound.",
            "The biologist studied cellular mechanisms.",
            "The astronomer observed distant galaxies.",
            "The mathematician proved a complex theorem.",
        ]
        tests.append(DomainTest("science", "science", science_texts, "low"))
        
        # Politics domain
        politics_texts = [
            "The politician delivered a speech to the parliament.",
            "The president announced new policies today.",
            "The senator voted on the healthcare bill.",
            "The mayor addressed the city council.",
            "The governor signed the new law.",
            "The congressman proposed legislation.",
            "The diplomat negotiated the peace treaty.",
            "The candidate campaigned across the state.",
        ]
        tests.append(DomainTest("politics", "politics", politics_texts, "low"))
        
        # Sports domain
        sports_texts = [
            "The athlete won the gold medal at the Olympics.",
            "The quarterback threw a touchdown pass.",
            "The basketball player scored thirty points.",
            "The soccer player kicked the winning goal.",
            "The tennis player served an ace.",
            "The runner broke the world record.",
            "The swimmer won the championship.",
            "The golfer hit a perfect drive.",
        ]
        tests.append(DomainTest("sports", "sports", sports_texts, "low"))
        
        # Business domain
        business_texts = [
            "The CEO announced quarterly earnings.",
            "The entrepreneur started a new company.",
            "The investor purchased shares in the startup.",
            "The manager led the team meeting.",
            "The consultant advised the client.",
            "The analyst reviewed the financial data.",
            "The executive presented the business plan.",
            "The director oversaw the project.",
        ]
        tests.append(DomainTest("business", "business", business_texts, "low"))
        
        # Technology domain
        technology_texts = [
            "The programmer wrote the software code.",
            "The engineer designed the new system.",
            "The developer created the mobile app.",
            "The architect built the database.",
            "The technician fixed the computer.",
            "The designer created the user interface.",
            "The analyst processed the data.",
            "The administrator managed the network.",
        ]
        tests.append(DomainTest("technology", "technology", technology_texts, "low"))
        
        # Education domain
        education_texts = [
            "The teacher explained the lesson to the students.",
            "The professor lectured on quantum physics.",
            "The instructor taught the workshop.",
            "The tutor helped the student with homework.",
            "The principal addressed the school assembly.",
            "The librarian organized the book collection.",
            "The researcher studied educational methods.",
            "The administrator managed the university.",
        ]
        tests.append(DomainTest("education", "education", education_texts, "low"))
        
        return tests
    
    def run_domain_test(self, test: DomainTest) -> Dict[str, Any]:
        """Run a single domain test"""
        print(f"\n🔍 Running domain test: {test.name}")
        print(f"   Category: {test.category}")
        print(f"   Expected activation: {test.expected_activation}")
        
        results = {
            'test_name': test.name,
            'category': test.category,
            'expected_activation': test.expected_activation,
            'texts': [],
            'overall_stats': {}
        }
        
        # Test each text
        for i, text in enumerate(test.test_texts):
            print(f"   Testing text {i+1}/{len(test.test_texts)}")
            
            # Get activations
            activations = self._get_concept_activations(text)
            
            # Analyze each block
            text_result = {
                'text': text,
                'activations': {},
                'concept_usage': {}
            }
            
            for block_name in activations.keys():
                stats = self._compute_concept_statistics(activations[block_name])
                text_result['activations'][block_name] = stats
                
                # Identify which concepts are most active
                block_activations = activations[block_name]
                active_concepts = []
                for concept_idx in range(block_activations.shape[-1]):
                    max_activation = np.max(block_activations[:, concept_idx])
                    if max_activation > 0.1:
                        concept_key = f"{block_name}_{concept_idx}"
                        concept_label = self.concept_labels.get(concept_key, f"concept_{concept_idx}")
                        active_concepts.append({
                            'concept_idx': concept_idx,
                            'concept_label': concept_label,
                            'max_activation': float(max_activation)
                        })
                
                # Sort by activation strength
                active_concepts.sort(key=lambda x: x['max_activation'], reverse=True)
                text_result['concept_usage'][block_name] = active_concepts
            
            results['texts'].append(text_result)
        
        # Compute overall statistics
        self._compute_overall_domain_stats(results)
        
        return results
    
    def _compute_overall_domain_stats(self, results: Dict[str, Any]):
        """Compute overall statistics for the domain"""
        # Aggregate statistics across all texts and blocks
        all_stats = defaultdict(list)
        all_concept_usage = defaultdict(list)
        
        for text_result in results['texts']:
            for block_name, stats in text_result['activations'].items():
                for metric, value in stats.items():
                    all_stats[f"{block_name}_{metric}"].append(value)
            
            for block_name, concepts in text_result['concept_usage'].items():
                for concept in concepts:
                    all_concept_usage[f"{block_name}_{concept['concept_label']}"].append(concept['max_activation'])
        
        # Compute overall statistics
        overall_stats = {}
        for metric, values in all_stats.items():
            overall_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # Compute concept usage statistics
        concept_usage_stats = {}
        for concept_key, activations in all_concept_usage.items():
            concept_usage_stats[concept_key] = {
                'mean_activation': float(np.mean(activations)),
                'max_activation': float(np.max(activations)),
                'usage_count': len(activations),
                'usage_rate': len(activations) / len(results['texts'])
            }
        
        results['overall_stats'] = overall_stats
        results['concept_usage_stats'] = concept_usage_stats
    
    def run_all_domain_tests(self) -> Dict[str, Any]:
        """Run all domain tests"""
        print("🌍 CBT CONCEPT CROSS-DOMAIN TESTING")
        print("=" * 60)
        
        tests = self.create_domain_tests()
        all_results = {
            'model_info': {
                'concept_blocks': [4, 5, 6, 7],
                'm': 32,
                'k': 4,
                'alpha': 0.3
            },
            'concept_labels': self.concept_labels,
            'tests': {}
        }
        
        for test in tests:
            results = self.run_domain_test(test)
            all_results['tests'][test.name] = results
        
        return all_results
    
    def analyze_domain_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-domain testing results"""
        print("\n📊 DOMAIN ANALYSIS")
        print("=" * 60)
        
        analysis = {
            'domain_summary': {},
            'concept_generalization': {},
            'domain_specialization': {},
            'recommendations': []
        }
        
        # Analyze each domain
        for test_name, test_results in results['tests'].items():
            print(f"\n🔍 {test_name.upper()} DOMAIN ANALYSIS:")
            
            # Get overall activation statistics
            mean_activations = []
            for metric, stats in test_results['overall_stats'].items():
                if 'mean_activation' in metric:
                    mean_activations.append(stats['mean'])
            
            avg_activation = np.mean(mean_activations)
            print(f"   Average activation: {avg_activation:.4f}")
            
            # Determine activation level
            if avg_activation > 0.05:
                activation_level = "high"
            elif avg_activation > 0.02:
                activation_level = "medium"
            elif avg_activation > 0.01:
                activation_level = "low"
            else:
                activation_level = "none"
            
            print(f"   Activation level: {activation_level}")
            
            # Check if matches expectation
            expected = test_results['expected_activation']
            matches_expectation = (
                (expected == "high" and activation_level in ["high", "medium"]) or
                (expected == "medium" and activation_level in ["medium", "low"]) or
                (expected == "low" and activation_level in ["low", "none"]) or
                (expected == "none" and activation_level == "none")
            )
            
            print(f"   Matches expectation: {matches_expectation}")
            
            # Analyze concept usage
            top_concepts = []
            for concept_key, stats in test_results['concept_usage_stats'].items():
                if stats['usage_rate'] > 0.3:  # Used in >30% of texts
                    top_concepts.append({
                        'concept': concept_key,
                        'usage_rate': stats['usage_rate'],
                        'mean_activation': stats['mean_activation']
                    })
            
            # Sort by usage rate
            top_concepts.sort(key=lambda x: x['usage_rate'], reverse=True)
            
            print(f"   Top concepts: {len(top_concepts)}")
            for concept in top_concepts[:3]:
                print(f"     - {concept['concept']}: {concept['usage_rate']:.1%} usage")
            
            analysis['domain_summary'][test_name] = {
                'avg_activation': float(avg_activation),
                'activation_level': activation_level,
                'matches_expectation': matches_expectation,
                'top_concepts': top_concepts[:5]
            }
            
            # Generate recommendations
            if test_name == "entertainment" and activation_level != "high":
                analysis['recommendations'].append(
                    "Entertainment domain should have high activation - check model loading"
                )
            elif test_name != "entertainment" and activation_level == "high":
                analysis['recommendations'].append(
                    f"{test_name} domain has unexpectedly high activation - concepts may be too general"
                )
            elif test_name != "entertainment" and activation_level == "none":
                analysis['recommendations'].append(
                    f"{test_name} domain has no activation - concepts are entertainment-specific"
                )
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], analysis: Dict[str, Any], filename: str = "cross_domain_results.json"):
        """Save cross-domain testing results"""
        output = {
            'results': results,
            'analysis': analysis,
            'timestamp': str(Path().cwd())
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def print_summary(self, analysis: Dict[str, Any]):
        """Print cross-domain testing summary"""
        print("\n🎯 CROSS-DOMAIN TESTING SUMMARY")
        print("=" * 60)
        
        print("Domain | Activation | Level | Matches Expectation")
        print("-------|------------|-------|-------------------")
        
        for domain, summary in analysis['domain_summary'].items():
            activation = summary['avg_activation']
            level = summary['activation_level']
            matches = "✅" if summary['matches_expectation'] else "❌"
            print(f"{domain:8} | {activation:10.4f} | {level:5} | {matches}")
        
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        if not analysis['recommendations']:
            print("  ✅ All domains behave as expected")


def main():
    """Main cross-domain testing function"""
    # Find latest model
    results_dir = Path("results/experiments_20250810_160233")
    model_file = "cbt_model_stab_kl_m32_k4_a30.pt"
    model_path = results_dir / model_file
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Run cross-domain testing
    tester = CBTDomainTester(str(model_path))
    results = tester.run_all_domain_tests()
    analysis = tester.analyze_domain_results(results)
    
    # Save and summarize
    tester.save_results(results, analysis)
    tester.print_summary(analysis)


if __name__ == "__main__":
    main() 
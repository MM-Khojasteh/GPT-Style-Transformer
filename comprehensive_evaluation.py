"""Comprehensive evaluation script with multiple metrics and prediction tests."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import argparse
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

from src.models.gpt import AdvancedGPT
from src.utils.tokenizer import CharTokenizer
from src.utils.checkpointing import load_checkpoint
from src.utils.device import get_device, print_gpu_info
from src.utils.config_presets import get_model_config
from src.utils.generation import generate_text, generate_with_beam_search


class ModelEvaluator:
    """Comprehensive model evaluator with multiple metrics."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def compute_perplexity(self, text: str, seq_len: int = 256) -> Tuple[float, float]:
        """Compute perplexity and average loss.
        
        Returns:
            (perplexity, average_loss)
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 2:
            return float('inf'), float('inf')
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(tokens) - 1, seq_len):
                chunk = tokens[i:i + seq_len + 1]
                if len(chunk) < 2:
                    break
                
                input_ids = torch.tensor([chunk[:-1]], device=self.device)
                targets = torch.tensor([chunk[1:]], device=self.device)
                
                logits = self.model(input_ids)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                loss = torch.nn.functional.cross_entropy(
                    logits_flat, targets_flat, reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += len(chunk) - 1
        
        if total_tokens == 0:
            return float('inf'), float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity, avg_loss
    
    def compute_accuracy(self, text: str, seq_len: int = 256, top_k: int = 1) -> float:
        """Compute next-token prediction accuracy (top-k)."""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 2:
            return 0.0
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(tokens) - 1, seq_len):
                chunk = tokens[i:i + seq_len + 1]
                if len(chunk) < 2:
                    break
                
                input_ids = torch.tensor([chunk[:-1]], device=self.device)
                targets = torch.tensor([chunk[1:]], device=self.device)
                
                logits = self.model(input_ids)
                
                if top_k == 1:
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == targets).sum().item()
                else:
                    top_k_preds = torch.topk(logits, k=top_k, dim=-1)[1]
                    for j in range(targets.size(1)):
                        if targets[0, j].item() in top_k_preds[0, j].cpu().numpy():
                            correct += 1
                
                total += targets.size(1)
        
        return correct / total if total > 0 else 0.0
    
    def compute_entropy(self, text: str, seq_len: int = 256) -> Dict[str, float]:
        """Compute prediction entropy metrics.
        
        Returns:
            Dictionary with average entropy, max entropy, min entropy
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 2:
            return {'avg_entropy': 0.0, 'max_entropy': 0.0, 'min_entropy': 0.0}
        
        entropies = []
        
        with torch.no_grad():
            for i in range(0, len(tokens) - 1, seq_len):
                chunk = tokens[i:i + seq_len + 1]
                if len(chunk) < 2:
                    break
                
                input_ids = torch.tensor([chunk[:-1]], device=self.device)
                logits = self.model(input_ids)
                
                # Compute entropy for each position
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum(dim=-1)
                entropies.extend(entropy.cpu().numpy().flatten().tolist())
        
        if not entropies:
            return {'avg_entropy': 0.0, 'max_entropy': 0.0, 'min_entropy': 0.0}
        
        return {
            'avg_entropy': np.mean(entropies),
            'max_entropy': np.max(entropies),
            'min_entropy': np.min(entropies),
            'std_entropy': np.std(entropies)
        }
    
    def compute_repetition_metrics(self, generated_text: str) -> Dict[str, float]:
        """Compute repetition and diversity metrics."""
        tokens = self.tokenizer.encode(generated_text)
        
        if len(tokens) < 2:
            return {
                'unique_tokens_ratio': 0.0,
                'avg_repetition_length': 0.0,
                'max_repetition_length': 0.0
            }
        
        # Unique tokens ratio
        unique_tokens = len(set(tokens))
        unique_ratio = unique_tokens / len(tokens) if len(tokens) > 0 else 0.0
        
        # Find repetition patterns
        max_repetition = 1
        current_repetition = 1
        
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                current_repetition += 1
                max_repetition = max(max_repetition, current_repetition)
            else:
                current_repetition = 1
        
        # Average repetition length
        repetition_lengths = []
        current_rep = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                current_rep += 1
            else:
                if current_rep > 1:
                    repetition_lengths.append(current_rep)
                current_rep = 1
        if current_rep > 1:
            repetition_lengths.append(current_rep)
        
        avg_repetition = np.mean(repetition_lengths) if repetition_lengths else 1.0
        
        return {
            'unique_tokens_ratio': unique_ratio,
            'avg_repetition_length': avg_repetition,
            'max_repetition_length': max_repetition,
            'total_tokens': len(tokens),
            'unique_tokens': unique_tokens
        }
    
    def compute_coherence_score(self, text: str, context_window: int = 10) -> Dict[str, float]:
        """Compute coherence score based on prediction confidence.
        
        Returns normalized score (0-1) where 1 is perfect coherence.
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < context_window + 1:
            return {'coherence_score': 0.0, 'avg_confidence': 0.0, 'normalized_score': 0.0}
        
        confidences = []
        vocab_size = self.tokenizer.vocab_size
        
        with torch.no_grad():
            for i in range(context_window, len(tokens)):
                context = tokens[i-context_window:i]
                target = tokens[i]
                
                input_ids = torch.tensor([context], device=self.device)
                logits = self.model(input_ids)
                
                # Get probability of correct token
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                confidence = probs[target].item()
                confidences.append(confidence)
        
        if not confidences:
            return {'coherence_score': 0.0, 'avg_confidence': 0.0, 'normalized_score': 0.0}
        
        avg_confidence = np.mean(confidences)
        # Normalize: random baseline is 1/vocab_size, perfect is 1.0
        random_baseline = 1.0 / vocab_size
        normalized_score = (avg_confidence - random_baseline) / (1.0 - random_baseline)
        normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
        
        return {
            'coherence_score': normalized_score,
            'avg_confidence': avg_confidence,
            'normalized_score': normalized_score,
            'random_baseline': random_baseline
        }
    
    def test_generation_quality(
        self, 
        prompts: List[str], 
        num_samples: int = 5,
        max_tokens: int = 100,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, any]:
        """Test generation quality with multiple prompts.
        
        Args:
            prompts: List of prompts to generate from
            num_samples: Number of samples per prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (auto if None)
            top_k: Top-k sampling (auto if None)
            top_p: Top-p sampling (auto if None)
        """
        results = {
            'generations': [],
            'avg_length': 0.0,
            'avg_unique_ratio': 0.0,
            'successful_generations': 0
        }
        
        # Adaptive generation parameters based on vocabulary size
        vocab_size = self.tokenizer.vocab_size
        if temperature is None:
            # For small vocabularies, use higher temperature to increase diversity
            if vocab_size < 100:
                temperature = 0.9
            elif vocab_size < 1000:
                temperature = 0.8
            else:
                temperature = 0.7
        
        if top_k is None:
            # For small vocabularies, use smaller top_k
            if vocab_size < 50:
                top_k = min(20, vocab_size)
            elif vocab_size < 200:
                top_k = min(40, vocab_size)
            else:
                top_k = min(50, vocab_size)
        
        if top_p is None:
            # For small vocabularies, use slightly higher top_p
            if vocab_size < 100:
                top_p = 0.95
            else:
                top_p = 0.9
        
        total_length = 0
        total_unique_ratio = 0
        
        for prompt in prompts:
            try:
                prompt_tokens = self.tokenizer.encode(prompt)
                if len(prompt_tokens) == 0:
                    continue
                
                generated_tokens = generate_text(
                    self.model,
                    prompt_tokens,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    device=str(self.device),
                    use_cache=True
                )
                
                generated_text = self.tokenizer.decode(generated_tokens)
                rep_metrics = self.compute_repetition_metrics(generated_text)
                
                results['generations'].append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'length': len(generated_tokens),
                    'unique_ratio': rep_metrics['unique_tokens_ratio']
                })
                
                total_length += len(generated_tokens)
                total_unique_ratio += rep_metrics['unique_tokens_ratio']
                results['successful_generations'] += 1
                
            except Exception as e:
                results['generations'].append({
                    'prompt': prompt,
                    'error': str(e)
                })
        
        if results['successful_generations'] > 0:
            results['avg_length'] = total_length / results['successful_generations']
            results['avg_unique_ratio'] = total_unique_ratio / results['successful_generations']
        
        return results
    
    def test_prediction_distribution(self, text: str, num_samples: int = 100) -> Dict[str, any]:
        """Analyze prediction distribution characteristics."""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < num_samples + 1:
            num_samples = len(tokens) - 1
        
        top1_confidences = []
        top3_confidences = []
        top5_confidences = []
        prediction_entropies = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(tokens) - 1)):
                context = tokens[max(0, i-50):i+1]
                if len(context) < 2:
                    continue
                
                target = tokens[i+1] if i+1 < len(tokens) else tokens[i]
                
                input_ids = torch.tensor([context[:-1]], device=self.device)
                logits = self.model(input_ids)
                
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                
                # Top-1 confidence
                top1_conf = probs[target].item()
                top1_confidences.append(top1_conf)
                
                # Top-k confidences
                top_probs, top_indices = torch.topk(probs, k=min(5, len(probs)))
                if target in top_indices:
                    idx = (top_indices == target).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        top3_confidences.append(top_probs[idx[0]].item() if idx[0] < 3 else 0.0)
                        top5_confidences.append(top_probs[idx[0]].item())
                    else:
                        top3_confidences.append(0.0)
                        top5_confidences.append(0.0)
                else:
                    top3_confidences.append(0.0)
                    top5_confidences.append(0.0)
                
                # Entropy
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                prediction_entropies.append(entropy)
        
        return {
            'top1_confidence': {
                'mean': np.mean(top1_confidences) if top1_confidences else 0.0,
                'std': np.std(top1_confidences) if top1_confidences else 0.0,
                'min': np.min(top1_confidences) if top1_confidences else 0.0,
                'max': np.max(top1_confidences) if top1_confidences else 0.0
            },
            'top3_confidence': {
                'mean': np.mean(top3_confidences) if top3_confidences else 0.0,
                'std': np.std(top3_confidences) if top3_confidences else 0.0
            },
            'top5_confidence': {
                'mean': np.mean(top5_confidences) if top5_confidences else 0.0,
                'std': np.std(top5_confidences) if top5_confidences else 0.0
            },
            'prediction_entropy': {
                'mean': np.mean(prediction_entropies) if prediction_entropies else 0.0,
                'std': np.std(prediction_entropies) if prediction_entropies else 0.0
            }
        }
    
    def comprehensive_evaluation(
        self, 
        test_text: str,
        prompts: Optional[List[str]] = None,
        seq_len: int = 256
    ) -> Dict[str, any]:
        """Run comprehensive evaluation with all metrics."""
        
        print("\n" + "=" * 70)
        print("Running Comprehensive Evaluation")
        print("=" * 70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_data_length': len(test_text),
            'test_tokens': len(self.tokenizer.encode(test_text)),
            'vocab_size': self.tokenizer.vocab_size
        }
        
        # 1. Perplexity and Loss
        print("\n[1/7] Computing Perplexity...")
        perplexity, avg_loss = self.compute_perplexity(test_text, seq_len)
        results['perplexity'] = perplexity
        results['average_loss'] = avg_loss
        print(f"  ✓ Perplexity: {perplexity:.4f}")
        print(f"  ✓ Average Loss: {avg_loss:.4f}")
        
        # 2. Accuracy Metrics
        print("\n[2/7] Computing Accuracy Metrics...")
        top1_acc = self.compute_accuracy(test_text, seq_len, top_k=1)
        top3_acc = self.compute_accuracy(test_text, seq_len, top_k=3)
        top5_acc = self.compute_accuracy(test_text, seq_len, top_k=5)
        top10_acc = self.compute_accuracy(test_text, seq_len, top_k=10)
        results['accuracy'] = {
            'top1': top1_acc,
            'top3': top3_acc,
            'top5': top5_acc,
            'top10': top10_acc
        }
        print(f"  ✓ Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"  ✓ Top-3 Accuracy: {top3_acc*100:.2f}%")
        print(f"  ✓ Top-5 Accuracy: {top5_acc*100:.2f}%")
        print(f"  ✓ Top-10 Accuracy: {top10_acc*100:.2f}%")
        
        # 3. Entropy Metrics
        print("\n[3/7] Computing Entropy Metrics...")
        entropy_metrics = self.compute_entropy(test_text, seq_len)
        results['entropy'] = entropy_metrics
        print(f"  ✓ Average Entropy: {entropy_metrics['avg_entropy']:.4f}")
        print(f"  ✓ Max Entropy: {entropy_metrics['max_entropy']:.4f}")
        print(f"  ✓ Min Entropy: {entropy_metrics['min_entropy']:.4f}")
        
        # 4. Coherence Score
        print("\n[4/7] Computing Coherence Score...")
        coherence_metrics = self.compute_coherence_score(test_text)
        results['coherence'] = coherence_metrics
        coherence_score = coherence_metrics['coherence_score']
        print(f"  ✓ Coherence Score (normalized): {coherence_score:.4f}")
        print(f"  ✓ Average Confidence: {coherence_metrics['avg_confidence']:.4f}")
        print(f"  ✓ Random Baseline: {coherence_metrics['random_baseline']:.4f}")
        
        # 5. Prediction Distribution Analysis
        print("\n[5/7] Analyzing Prediction Distribution...")
        pred_dist = self.test_prediction_distribution(test_text, num_samples=100)
        results['prediction_distribution'] = pred_dist
        print(f"  ✓ Top-1 Confidence: {pred_dist['top1_confidence']['mean']:.4f} ± {pred_dist['top1_confidence']['std']:.4f}")
        print(f"  ✓ Top-5 Confidence: {pred_dist['top5_confidence']['mean']:.4f} ± {pred_dist['top5_confidence']['std']:.4f}")
        
        # 6. Generation Quality Tests
        if prompts:
            print("\n[6/7] Testing Generation Quality...")
            # Use adaptive parameters for small vocabularies
            vocab_size = self.tokenizer.vocab_size
            gen_results = self.test_generation_quality(
                prompts, 
                num_samples=len(prompts),
                temperature=None,  # Auto-adapt
                top_k=None,  # Auto-adapt
                top_p=None  # Auto-adapt
            )
            results['generation_quality'] = gen_results
            print(f"  ✓ Successful Generations: {gen_results['successful_generations']}/{len(prompts)}")
            print(f"  ✓ Average Length: {gen_results['avg_length']:.1f} tokens")
            print(f"  ✓ Average Unique Ratio: {gen_results['avg_unique_ratio']:.4f}")
            if vocab_size < 100:
                print(f"  ℹ Using adaptive generation parameters for small vocabulary (size={vocab_size})")
        else:
            results['generation_quality'] = None
        
        # 7. Repetition Analysis (on generated text if available)
        print("\n[7/7] Computing Repetition Metrics...")
        if prompts and results.get('generation_quality'):
            all_generated = ' '.join([
                g['generated'] for g in results['generation_quality']['generations']
                if 'generated' in g
            ])
            if all_generated:
                rep_metrics = self.compute_repetition_metrics(all_generated)
                results['repetition_metrics'] = rep_metrics
                print(f"  ✓ Unique Tokens Ratio: {rep_metrics['unique_tokens_ratio']:.4f}")
                print(f"  ✓ Max Repetition Length: {rep_metrics['max_repetition_length']}")
        
        return results


def print_evaluation_report(results: Dict[str, any]):
    """Print comprehensive evaluation report."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\n📊 Core Metrics:")
    perplexity = results['perplexity']
    print(f"  • Perplexity: {perplexity:.4f}", end="")
    if perplexity < 20:
        print(" ✓ (Excellent)")
    elif perplexity < 50:
        print(" ✓ (Good)")
    elif perplexity < 100:
        print(" ⚠ (Fair)")
    else:
        print(" ✗ (Needs improvement)")
    print(f"  • Average Loss: {results['average_loss']:.4f}")
    
    print(f"\n🎯 Accuracy Metrics:")
    acc = results['accuracy']
    vocab_size = results.get('vocab_size', 36)
    random_baseline = 1.0 / vocab_size * 100
    
    print(f"  • Top-1 Accuracy: {acc['top1']*100:.2f}% (Random: {random_baseline:.2f}%)", end="")
    if acc['top1'] > random_baseline / 100 * 2:
        print(" ✓")
    else:
        print(" ⚠")
    print(f"  • Top-3 Accuracy: {acc['top3']*100:.2f}%")
    print(f"  • Top-5 Accuracy: {acc['top5']*100:.2f}%")
    print(f"  • Top-10 Accuracy: {acc['top10']*100:.2f}%")
    
    print(f"\n📈 Prediction Quality:")
    pred_dist = results['prediction_distribution']
    print(f"  • Top-1 Confidence: {pred_dist['top1_confidence']['mean']:.4f} ± {pred_dist['top1_confidence']['std']:.4f}")
    print(f"  • Top-5 Confidence: {pred_dist['top5_confidence']['mean']:.4f} ± {pred_dist['top5_confidence']['std']:.4f}")
    print(f"  • Prediction Entropy: {pred_dist['prediction_entropy']['mean']:.4f} ± {pred_dist['prediction_entropy']['std']:.4f}")
    
    print(f"\n🔍 Model Characteristics:")
    entropy = results['entropy']
    max_entropy = np.log(results.get('vocab_size', 36))  # Maximum entropy for uniform distribution
    entropy_ratio = entropy['avg_entropy'] / max_entropy if max_entropy > 0 else 0
    print(f"  • Average Entropy: {entropy['avg_entropy']:.4f} (Max: {max_entropy:.4f}, Ratio: {entropy_ratio:.2%})")
    
    coherence = results.get('coherence', {})
    if isinstance(coherence, dict):
        coherence_score = coherence.get('coherence_score', 0.0)
        avg_confidence = coherence.get('avg_confidence', 0.0)
        random_baseline = coherence.get('random_baseline', 0.0)
        vocab_size = results.get('vocab_size', 36)
        
        print(f"  • Coherence Score (normalized): {coherence_score:.4f}", end="")
        if coherence_score > 0.5:
            print(" ✓ (High coherence)")
        elif coherence_score > 0.2:
            print(" ✓ (Moderate coherence)")
        else:
            print(" ⚠ (Low coherence)")
        print(f"  • Average Confidence: {avg_confidence:.4f}")
        print(f"  • Random Baseline: {random_baseline:.4f}")
        
        # Add explanation for small vocabularies
        if vocab_size < 100:
            improvement_ratio = avg_confidence / random_baseline if random_baseline > 0 else 0
            print(f"  ℹ Note: With small vocabulary (size={vocab_size}), low coherence is expected.")
            print(f"    Model confidence ({avg_confidence:.4f}) is {improvement_ratio:.1f}x better than random baseline.")
    else:
        print(f"  • Coherence Score: {coherence:.4f}")
    
    if results.get('generation_quality'):
        gen_qual = results['generation_quality']
        print(f"\n✨ Generation Quality:")
        print(f"  • Successful Generations: {gen_qual['successful_generations']}")
        print(f"  • Average Length: {gen_qual['avg_length']:.1f} tokens")
        print(f"  • Average Unique Ratio: {gen_qual['avg_unique_ratio']:.4f}")
    
    if results.get('repetition_metrics'):
        rep = results['repetition_metrics']
        vocab_size = results.get('vocab_size', 36)
        print(f"\n🔄 Repetition Analysis:")
        unique_ratio = rep['unique_tokens_ratio']
        print(f"  • Unique Tokens Ratio: {unique_ratio:.4f}", end="")
        if unique_ratio > 0.5:
            print(" ✓ (Good diversity)")
        elif unique_ratio > 0.2:
            print(" ⚠ (Moderate diversity)")
        else:
            print(" ✗ (High repetition)")
        print(f"  • Max Repetition Length: {rep['max_repetition_length']}")
        print(f"  • Average Repetition: {rep['avg_repetition_length']:.2f}")
        
        # Add explanation for small vocabularies
        if vocab_size < 100:
            expected_ratio = rep.get('unique_tokens', 0) / vocab_size if vocab_size > 0 else 0
            print(f"  ℹ Note: With vocabulary size {vocab_size}, maximum possible unique ratio is ~{expected_ratio:.2f}")
            print(f"    Consider using larger dataset or different tokenization for better diversity.")
    
    # Add summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    vocab_size = results.get('vocab_size', 36)
    test_tokens = results.get('test_tokens', 0)
    
    # Dataset size analysis
    if test_tokens < 1000:
        print("\n📊 Dataset Analysis:")
        print(f"  • Small dataset detected ({test_tokens} tokens)")
        print("  • Recommendations:")
        print("    - Use larger training dataset for better generalization")
        print("    - Consider data augmentation techniques")
        print("    - Split data into train/validation/test sets")
    
    # Vocabulary size analysis
    if vocab_size < 100:
        print("\n📚 Vocabulary Analysis:")
        print(f"  • Small vocabulary detected ({vocab_size} tokens)")
        print("  • This is expected for character-level tokenization")
        print("  • Recommendations:")
        print("    - Consider BPE or word-level tokenization for larger vocabularies")
        print("    - Low coherence scores are normal with small vocabularies")
        print("    - Repetition metrics may be limited by vocabulary size")
    
    # Model performance summary
    perplexity = results['perplexity']
    top1_acc = results['accuracy']['top1']
    
    print("\n🎯 Model Performance Summary:")
    if perplexity < 20:
        print("  ✓ Excellent perplexity - model is well-trained")
    elif perplexity < 50:
        print("  ✓ Good perplexity - model shows solid performance")
    else:
        print("  ⚠ Perplexity could be improved with more training or better hyperparameters")
    
    random_baseline = 1.0 / vocab_size
    if top1_acc > random_baseline * 5:
        print("  ✓ Strong accuracy improvement over random baseline")
    elif top1_acc > random_baseline * 2:
        print("  ✓ Moderate accuracy improvement over random baseline")
    else:
        print("  ⚠ Accuracy improvement over baseline could be better")
    
    # Add note about evaluation on unseen data
    test_tokens = results.get('test_tokens', 0)
    if test_tokens > 0:
        print("\n📝 Evaluation Notes:")
        print("  • Lower metrics on test set are expected and indicate realistic evaluation")
        print("  • If test set contains unseen data, metrics will be lower than training metrics")
        print("  • This is normal and shows the model's generalization ability")
        print("  • For best results, use a separate test set that wasn't seen during training")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive model evaluation with multiple metrics'
    )
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                        help='Model preset')
    parser.add_argument('--test-data', type=str, default='data/sample_text.txt',
                        help='Path to test data')
    parser.add_argument('--seq-len', type=int, default=256,
                        help='Sequence length for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save JSON results (optional)')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Custom prompts for generation testing')
    parser.add_argument('--train-data', type=str, default=None,
                        help='Path to training data (for tokenizer consistency and overlap detection)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Comprehensive Model Evaluation")
    print("=" * 70)
    print()
    
    # Load test data
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    print(f"Test Data: {args.test_data}")
    print(f"  Length: {len(test_text)} characters")
    
    # Check for potential overlap with training data
    if args.train_data:
        try:
            with open(args.train_data, 'r', encoding='utf-8') as f:
                train_text = f.read()
            
            # Simple overlap detection: check if test data is subset of training data
            test_tokens = set(test_text.split())
            train_tokens = set(train_text.split())
            overlap_ratio = len(test_tokens & train_tokens) / len(test_tokens) if len(test_tokens) > 0 else 0
            
            if overlap_ratio > 0.8:
                print(f"  ⚠ Warning: High overlap detected with training data ({overlap_ratio:.1%})")
                print(f"    This may lead to overly optimistic evaluation results.")
                print(f"    For realistic evaluation, use a separate test set.")
            elif overlap_ratio > 0.5:
                print(f"  ⚠ Note: Moderate overlap with training data ({overlap_ratio:.1%})")
        except FileNotFoundError:
            print(f"  ℹ Training data file not found: {args.train_data}")
            print(f"    Skipping overlap detection.")
    
    print()
    
    # Create tokenizer - use training data if provided for consistency
    if args.train_data:
        try:
            with open(args.train_data, 'r', encoding='utf-8') as f:
                train_text = f.read()
            # Combine train and test for tokenizer to ensure vocabulary consistency
            combined_text = train_text + "\n" + test_text
            tokenizer = CharTokenizer(combined_text)
            print(f"Tokenizer (created from training + test data for consistency):")
        except FileNotFoundError:
            print(f"Warning: Training data file not found, using test data for tokenizer")
            tokenizer = CharTokenizer(test_text)
            print(f"Tokenizer:")
    else:
        tokenizer = CharTokenizer(test_text)
        print(f"Tokenizer:")
    
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Total tokens: {len(tokenizer.encode(test_text))}")
    print()
    
    # Get model configuration
    model_config = get_model_config(preset=args.preset, vocab_size=tokenizer.vocab_size)
    
    # Create model
    model = AdvancedGPT(**model_config)
    
    # Load checkpoint
    try:
        load_checkpoint(args.checkpoint, model)
        print(f"✓ Loaded checkpoint: {args.checkpoint}")
    except FileNotFoundError:
        print(f"✗ Error: Checkpoint not found: {args.checkpoint}")
        return
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return
    
    # Move to device
    device = get_device()
    model = model.to(device)
    print_gpu_info()
    print()
    
    # Prepare prompts
    if args.prompts:
        prompts = args.prompts
    else:
        # Default prompts from test data
        prompts = [
            "The transformer",
            "GPT models",
            "Training large",
            "Modern transformers",
            "The attention"
        ]
        # Filter prompts that exist in text
        prompts = [p for p in prompts if p.lower() in test_text.lower()]
    
    # Create evaluator
    evaluator = ModelEvaluator(model, tokenizer, device)
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(
        test_text=test_text,
        prompts=prompts,
        seq_len=args.seq_len
    )
    
    # Print report
    print_evaluation_report(results)
    
    # Save results if requested
    if args.output:
        # Convert numpy types to native Python types for JSON
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results_json = convert_to_native(results)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {args.output}")
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()


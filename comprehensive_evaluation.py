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
    
    def compute_coherence_score(self, text: str, context_window: int = 10) -> float:
        """Compute coherence score based on prediction confidence."""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < context_window + 1:
            return 0.0
        
        confidences = []
        
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
        
        return np.mean(confidences) if confidences else 0.0
    
    def test_generation_quality(
        self, 
        prompts: List[str], 
        num_samples: int = 5,
        max_tokens: int = 100
    ) -> Dict[str, any]:
        """Test generation quality with multiple prompts."""
        results = {
            'generations': [],
            'avg_length': 0.0,
            'avg_unique_ratio': 0.0,
            'successful_generations': 0
        }
        
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
                    temperature=0.7,
                    top_k=min(50, self.tokenizer.vocab_size),
                    top_p=0.9,
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
            'test_tokens': len(self.tokenizer.encode(test_text))
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
        coherence = self.compute_coherence_score(test_text)
        results['coherence_score'] = coherence
        print(f"  ✓ Coherence Score: {coherence:.4f}")
        
        # 5. Prediction Distribution Analysis
        print("\n[5/7] Analyzing Prediction Distribution...")
        pred_dist = self.test_prediction_distribution(test_text, num_samples=100)
        results['prediction_distribution'] = pred_dist
        print(f"  ✓ Top-1 Confidence: {pred_dist['top1_confidence']['mean']:.4f} ± {pred_dist['top1_confidence']['std']:.4f}")
        print(f"  ✓ Top-5 Confidence: {pred_dist['top5_confidence']['mean']:.4f} ± {pred_dist['top5_confidence']['std']:.4f}")
        
        # 6. Generation Quality Tests
        if prompts:
            print("\n[6/7] Testing Generation Quality...")
            gen_results = self.test_generation_quality(prompts, num_samples=len(prompts))
            results['generation_quality'] = gen_results
            print(f"  ✓ Successful Generations: {gen_results['successful_generations']}/{len(prompts)}")
            print(f"  ✓ Average Length: {gen_results['avg_length']:.1f} tokens")
            print(f"  ✓ Average Unique Ratio: {gen_results['avg_unique_ratio']:.4f}")
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
    print(f"  • Perplexity: {results['perplexity']:.4f}")
    print(f"  • Average Loss: {results['average_loss']:.4f}")
    
    print(f"\n🎯 Accuracy Metrics:")
    acc = results['accuracy']
    print(f"  • Top-1 Accuracy: {acc['top1']*100:.2f}%")
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
    print(f"  • Average Entropy: {entropy['avg_entropy']:.4f}")
    print(f"  • Coherence Score: {results['coherence_score']:.4f}")
    
    if results.get('generation_quality'):
        gen_qual = results['generation_quality']
        print(f"\n✨ Generation Quality:")
        print(f"  • Successful Generations: {gen_qual['successful_generations']}")
        print(f"  • Average Length: {gen_qual['avg_length']:.1f} tokens")
        print(f"  • Average Unique Ratio: {gen_qual['avg_unique_ratio']:.4f}")
    
    if results.get('repetition_metrics'):
        rep = results['repetition_metrics']
        print(f"\n🔄 Repetition Analysis:")
        print(f"  • Unique Tokens Ratio: {rep['unique_tokens_ratio']:.4f}")
        print(f"  • Max Repetition Length: {rep['max_repetition_length']}")
        print(f"  • Average Repetition: {rep['avg_repetition_length']:.2f}")
    
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
    print()
    
    # Create tokenizer
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


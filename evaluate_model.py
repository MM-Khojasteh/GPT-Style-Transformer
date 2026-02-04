"""Evaluate trained GPT model on sample data."""

import torch
import argparse
import numpy as np
from src.models.gpt import AdvancedGPT
from src.utils.tokenizer import CharTokenizer
from src.utils.checkpointing import load_checkpoint
from src.utils.device import get_device, print_gpu_info
from src.utils.config_presets import get_model_config
from src.utils.generation import generate_text


def compute_perplexity(model, tokenizer, text, device, seq_len=256):
    """Compute perplexity on given text.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        text: Text to evaluate
        device: Device to run on
        
    Returns:
        Perplexity score
    """
    model.eval()
    tokens = tokenizer.encode(text)
    
    if len(tokens) < 2:
        return float('inf')
    
    total_loss = 0.0
    total_tokens = 0
    
    # Process in chunks
    with torch.no_grad():
        for i in range(0, len(tokens) - 1, seq_len):
            chunk = tokens[i:i + seq_len + 1]
            if len(chunk) < 2:
                break
            
            input_ids = torch.tensor([chunk[:-1]], device=device)
            targets = torch.tensor([chunk[1:]], device=device)
            
            # Forward pass
            logits = model(input_ids)  # (1, seq_len, vocab_size)
            
            # Compute loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction='sum')
            
            total_loss += loss.item()
            total_tokens += len(chunk) - 1
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def compute_accuracy(model, tokenizer, text, device, seq_len=256, top_k=1):
    """Compute next-token prediction accuracy.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        text: Text to evaluate
        device: Device to run on
        top_k: Consider top-k predictions as correct
        
    Returns:
        Accuracy score
    """
    model.eval()
    tokens = tokenizer.encode(text)
    
    if len(tokens) < 2:
        return 0.0
    
    correct = 0
    total = 0
    
    # Process in chunks
    with torch.no_grad():
        for i in range(0, len(tokens) - 1, seq_len):
            chunk = tokens[i:i + seq_len + 1]
            if len(chunk) < 2:
                break
            
            input_ids = torch.tensor([chunk[:-1]], device=device)
            targets = torch.tensor([chunk[1:]], device=device)
            
            # Forward pass
            logits = model(input_ids)  # (1, seq_len, vocab_size)
            
            # Get predictions
            if top_k == 1:
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == targets).sum().item()
            else:
                # Top-k accuracy
                top_k_preds = torch.topk(logits, k=top_k, dim=-1)[1]  # (1, seq_len, top_k)
                for j in range(targets.size(1)):
                    if targets[0, j].item() in top_k_preds[0, j].cpu().numpy():
                        correct += 1
            
            total += targets.size(1)
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    return accuracy


def show_predictions(model, tokenizer, text, device, num_examples=5, context_len=50):
    """Show example predictions from the model.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        text: Text to sample from
        device: Device to run on
        num_examples: Number of examples to show
        context_len: Length of context to use
    """
    model.eval()
    tokens = tokenizer.encode(text)
    
    print("\n" + "=" * 70)
    print("Example Predictions")
    print("=" * 70)
    
    with torch.no_grad():
        for i in range(min(num_examples, len(tokens) - context_len - 1)):
            # Get context
            context = tokens[i:i + context_len]
            target_token = tokens[i + context_len]
            
            # Predict
            input_ids = torch.tensor([context], device=device)
            logits = model(input_ids)
            
            # Get top predictions
            last_logits = logits[0, -1, :]  # (vocab_size,)
            probs = torch.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5)
            
            # Decode
            context_text = tokenizer.decode(context)
            target_char = tokenizer.decode([target_token])
            predicted_char = tokenizer.decode([top_indices[0].item()])
            
            print(f"\nExample {i+1}:")
            print(f"  Context: '{context_text[-30:]}'")  # Last 30 chars
            print(f"  Target: '{target_char}'")
            print(f"  Predicted: '{predicted_char}' (prob: {top_probs[0].item():.4f})")
            print(f"  Top 5 predictions:")
            for j in range(5):
                char = tokenizer.decode([top_indices[j].item()])
                print(f"    {j+1}. '{char}' ({top_probs[j].item():.4f})")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained GPT model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                        help='Model preset (should match training preset)')
    parser.add_argument('--data', type=str, default='data/sample_text.txt',
                        help='Path to evaluation data')
    parser.add_argument('--seq-len', type=int, default=256,
                        help='Sequence length for evaluation')
    parser.add_argument('--show-examples', type=int, default=5,
                        help='Number of example predictions to show')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Model Evaluation on Sample Data")
    print("=" * 70)
    print()
    
    # Load data
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Evaluation Data:")
    print(f"  File: {args.data}")
    print(f"  Length: {len(text)} characters")
    print()
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    tokens = tokenizer.encode(text)
    
    print(f"Tokenizer:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Total tokens: {len(tokens)}")
    print()
    
    # Get model configuration
    model_config = get_model_config(preset=args.preset, vocab_size=tokenizer.vocab_size)
    
    print(f"Model Configuration ({args.preset}):")
    print(f"  embed_dim: {model_config['embed_dim']}")
    print(f"  num_heads: {model_config['num_heads']}")
    print(f"  num_layers: {model_config['num_layers']}")
    print()
    
    # Create model
    model = AdvancedGPT(**model_config)
    
    # Load checkpoint
    try:
        load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from {args.checkpoint}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Move to device
    device = get_device()
    model = model.to(device)
    model.eval()
    
    print_gpu_info()
    print()
    
    # Compute metrics
    print("=" * 70)
    print("Computing Evaluation Metrics")
    print("=" * 70)
    print()
    
    # Perplexity
    print("Computing perplexity...")
    perplexity, avg_loss = compute_perplexity(model, tokenizer, text, device, args.seq_len)
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")
    print()
    
    # Accuracy
    print("Computing accuracy...")
    top1_accuracy = compute_accuracy(model, tokenizer, text, device, args.seq_len, top_k=1)
    top5_accuracy = compute_accuracy(model, tokenizer, text, device, args.seq_len, top_k=5)
    print(f"  Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
    print()
    
    # Show example predictions
    show_predictions(model, tokenizer, text, device, num_examples=args.show_examples)
    
    # Test text generation
    print("\n" + "=" * 70)
    print("Text Generation Test")
    print("=" * 70)
    
    # Generate from different prompts
    prompts = [
        "The transformer",
        "GPT models",
        "Training large",
        "Modern transformers",
        "The attention"
    ]
    
    for prompt in prompts:
        if prompt.lower() in text.lower():
            prompt_tokens = tokenizer.encode(prompt)
            if len(prompt_tokens) > 0:
                print(f"\nPrompt: '{prompt}'")
                # Adjust top_k to not exceed vocab size
                top_k = min(50, tokenizer.vocab_size)
                generated_tokens = generate_text(
                    model,
                    prompt_tokens,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_k=top_k,
                    top_p=0.9,
                    device=str(device),
                    use_cache=True
                )
                generated_text = tokenizer.decode(generated_tokens)
                print(f"Generated: '{generated_text}'")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
    print()


if __name__ == '__main__':
    main()


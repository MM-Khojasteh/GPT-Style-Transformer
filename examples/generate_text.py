"""Example text generation script with optimized configurations."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from src.models.gpt import AdvancedGPT
from src.utils.tokenizer import CharTokenizer
from src.utils.generation import generate_text
from src.utils.checkpointing import load_checkpoint
from src.utils.device import get_device, print_gpu_info
from src.utils.config_presets import get_model_config
from src.utils.env_config import env_str, env_int, env_float


def main():
    parser = argparse.ArgumentParser(description='Generate text with GPT model')
    parser.add_argument('--checkpoint', type=str,
                        default=env_str('CHECKPOINT_PATH', 'Trained model weights to load'),
                        help='Path to model checkpoint')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                        help='Model preset (should match training preset)')
    parser.add_argument('--prompt', type=str, default='The future of',
                        help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int,
                        default=env_int('GEN_MAX_NEW_TOKENS', 'Maximum tokens to generate'),
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float,
                        default=env_float('GEN_TEMPERATURE', 'Sampling temperature (0 = greedy)'),
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int,
                        default=env_int('GEN_TOP_K', 'Top-k sampling (0 disables)'),
                        help='Top-k sampling')
    parser.add_argument('--top-p', type=float,
                        default=env_float('GEN_TOP_P', 'Nucleus sampling (1.0 disables)'),
                        help='Top-p (nucleus) sampling')
    args = parser.parse_args()
    
    # Load tokenizer
    with open(env_str('TOKENIZER_CORPUS_PATH',
                      'Corpus used to rebuild the character vocabulary'),
              'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    
    # Get model configuration from preset
    model_config = get_model_config(preset=args.preset, vocab_size=tokenizer.vocab_size)
    
    print(f"Loading model with {args.preset.upper()} configuration...")
    print(f"  embed_dim: {model_config['embed_dim']}")
    print(f"  num_heads: {model_config['num_heads']}")
    print(f"  num_layers: {model_config['num_layers']}")
    print()
    
    # Create model
    model = AdvancedGPT(**model_config)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model)
    model.eval()
    
    # Check GPU availability
    device = get_device()
    model = model.to(device)
    print_gpu_info()
    print()
    
    # Generate text
    prompt_tokens = tokenizer.encode(args.prompt)
    
    print(f"Prompt: {args.prompt}")
    print(f"Generation settings:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print("\nGenerating...\n")
    
    generated_tokens = generate_text(
        model,
        prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=str(device),
        use_cache=True
    )
    
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text:\n{generated_text}")


if __name__ == '__main__':
    main()


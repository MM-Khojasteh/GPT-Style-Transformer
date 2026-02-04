"""Test text generation on GPU."""

import torch
from src.models.gpt import AdvancedGPT
from src.utils.tokenizer import CharTokenizer
from src.utils.generation import generate_text
from src.utils.device import get_device, print_gpu_info
from src.utils.checkpointing import load_checkpoint


def main():
    print("=" * 70)
    print("Testing Text Generation on GPU")
    print("=" * 70)
    print()
    
    # Load sample text
    with open('data/sample_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    
    # Create model
    print("Creating GPT model...")
    model = AdvancedGPT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout_prob=0.1,
        pos_emb_type='learned',
        use_rope=True,
        weight_tying=True
    )
    
    # Try to load checkpoint if available
    try:
        checkpoint_path = './checkpoints/best_model.pt'
        load_checkpoint(checkpoint_path, model)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except:
        print("No checkpoint found, using randomly initialized model")
    
    # Move to GPU
    device = get_device()
    print_gpu_info()
    print()
    
    model = model.to(device)
    model.eval()
    
    # Generate text
    prompts = [
        "The transformer",
        "GPT models",
        "Training large"
    ]
    
    print("=" * 70)
    print("Generating Text")
    print("=" * 70)
    print()
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        prompt_tokens = tokenizer.encode(prompt)
        
        generated_tokens = generate_text(
            model,
            prompt_tokens,
            max_new_tokens=30,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            device=str(device),
            use_cache=True
        )
        
        generated_text = tokenizer.decode(generated_tokens)
        print(f"Generated: '{generated_text}'")
        print()
    
    print("=" * 70)
    print("Generation Test Completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()


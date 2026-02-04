"""Example text generation script."""

import torch
from src.models.gpt import AdvancedGPT
from src.utils.tokenizer import CharTokenizer
from src.utils.generation import generate_text
from src.utils.checkpointing import load_checkpoint
from src.utils.device import get_device, print_gpu_info


def main():
    # Load tokenizer
    with open('data/sample_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    
    # Create model (same config as training)
    model = AdvancedGPT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        dropout_prob=0.1,
        pos_emb_type='learned',
        use_rope=True,
        weight_tying=True
    )
    
    # Load checkpoint
    checkpoint_path = './checkpoints/best_model.pt'
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    # Generate text
    prompt = "The future of"
    prompt_tokens = tokenizer.encode(prompt)
    
    print(f"Prompt: {prompt}")
    print("\nGenerating...\n")
    
    # Check GPU availability
    device = get_device()
    print_gpu_info()
    
    generated_tokens = generate_text(
        model,
        prompt_tokens,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        device=str(device),
        use_cache=True
    )
    
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text:\n{generated_text}")


if __name__ == '__main__':
    main()


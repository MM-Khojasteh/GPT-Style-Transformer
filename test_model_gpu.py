"""Test script to run GPT model on GPU and verify it works."""

import torch
from src.models.gpt import AdvancedGPT
from src.utils.device import get_device, print_gpu_info
from src.utils.tokenizer import CharTokenizer


def main():
    print("=" * 70)
    print("Testing GPT Model on GPU")
    print("=" * 70)
    
    # Check GPU availability
    print_gpu_info()
    print()
    
    # Create sample text
    sample_text = """
    The transformer architecture has revolutionized natural language processing. 
    GPT models use self-attention mechanisms to understand context and relationships.
    These models can generate coherent text by learning patterns from large datasets.
    The attention mechanism allows the model to focus on relevant parts of the input.
    Modern transformers use techniques like rotary position embeddings for efficiency.
    """
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = CharTokenizer(sample_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    
    # Create model
    print("Creating GPT model...")
    model = AdvancedGPT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=256,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout_prob=0.1,
        pos_emb_type='learned',
        use_rope=True,
        weight_tying=True
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    print()
    
    # Move model to GPU
    device = get_device()
    print(f"Moving model to device: {device}")
    model = model.to(device)
    print()
    
    # Test forward pass
    print("Testing forward pass on GPU...")
    print("-" * 70)
    
    # Encode some text
    test_text = "The transformer architecture"
    input_ids = tokenizer.encode(test_text)
    print(f"Input text: '{test_text}'")
    print(f"Input token IDs: {input_ids}")
    print(f"Input length: {len(input_ids)}")
    print()
    
    # Convert to tensor and move to GPU
    input_tensor = torch.tensor([input_ids], device=device)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor device: {input_tensor.device}")
    print()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits device: {logits.device}")
    print()
    
    # Check predictions
    print("Top 5 predicted tokens for last position:")
    last_logits = logits[0, -1, :]
    top_k_values, top_k_indices = torch.topk(last_logits, k=5)
    
    for i, (idx, val) in enumerate(zip(top_k_indices, top_k_values)):
        token = tokenizer.idx_to_char.get(idx.item(), '<UNK>')
        print(f"  {i+1}. Token '{token}' (ID: {idx.item()}) - Logit: {val.item():.4f}")
    
    print()
    print("=" * 70)
    print("GPU Test Completed Successfully!")
    print("=" * 70)
    
    # Memory info
    if device.type == 'cuda':
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")


if __name__ == '__main__':
    main()


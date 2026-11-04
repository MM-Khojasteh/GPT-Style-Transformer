"""Test training loop on GPU with a small dataset."""

import torch
from src.models.gpt import AdvancedGPT
from src.training.trainer import Trainer
from src.training.optimizers import get_optimizer
from src.training.schedulers import get_scheduler
from src.utils.tokenizer import CharTokenizer
from src.utils.device import get_device, print_gpu_info


def main():
    print("=" * 70)
    print("Testing GPT Training on GPU")
    print("=" * 70)
    print()
    
    # Load sample text
    try:
        with open('data/sample_text.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        # Create sample text if file doesn't exist
        text = """
        The transformer architecture has revolutionized natural language processing. 
        GPT models use self-attention mechanisms to understand context and relationships between words.
        These models can generate coherent text by learning patterns from large datasets.
        The attention mechanism allows the model to focus on relevant parts of the input sequence.
        Modern transformers use techniques like rotary position embeddings and flash attention for efficiency.
        Training large language models requires significant computational resources and careful optimization.
        Fine-tuning allows us to adapt pre-trained models for specific tasks and domains.
        The success of transformers has led to breakthroughs in machine translation, text generation, and understanding.
        Researchers continue to improve these architectures with innovations like mixture of experts and grouped query attention.
        Practical applications include chatbots, code generation, content creation, and automated reasoning systems.
        """
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(data)}")
    print()
    
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
    
    print(f"Model parameters: {model.get_num_params():,}")
    print()
    
    # Check GPU availability
    print_gpu_info()
    print()
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        optimizer_type='adamw',
        lr=3e-4,
        weight_decay=0.1
    )
    
    # Create scheduler
    total_steps = max(100, len(data) // 4)  # At least 100 steps
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        max_lr=3e-4,
        warmup_steps=10,
        total_steps=total_steps
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_data=data,
        optimizer=optimizer,
        device=get_device(),
        seq_len=64,
        batch_size=4,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        use_amp=True,
        checkpoint_dir='./checkpoints',
        checkpoint_interval=50,
        log_interval=5,
        use_wandb=False,
        use_tensorboard=False,
        project_name='gpt-transformer-test'
    )
    
    print()
    print("=" * 70)
    print("Starting Training (2 epochs for testing)")
    print("=" * 70)
    print()
    
    # Train for 2 epochs
    trainer.train(epochs=2, scheduler=scheduler)
    
    print()
    print("=" * 70)
    print("Training Test Completed!")
    print("=" * 70)
    
    # Memory info
    device = get_device()
    if device.type == 'cuda':
        print(f"\nFinal GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")


if __name__ == '__main__':
    main()


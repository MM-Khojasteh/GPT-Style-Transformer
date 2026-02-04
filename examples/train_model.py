"""Example training script for GPT model."""

import torch
from src.models.gpt import AdvancedGPT
from src.training.trainer import Trainer
from src.training.optimizers import get_optimizer
from src.training.schedulers import get_scheduler
from src.utils.tokenizer import CharTokenizer
from src.utils.device import get_device, print_gpu_info


def main():
    # Load and prepare data
    with open('data/sample_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(data)}")
    
    # Create model
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
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        optimizer_type='adamw',
        lr=3e-4,
        weight_decay=0.1
    )
    
    # Create scheduler
    total_steps = len(data) // 16 * 10  # Approximate steps for 10 epochs
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        max_lr=3e-4,
        warmup_steps=total_steps // 10,
        total_steps=total_steps
    )
    
    # Check GPU availability
    print_gpu_info()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=data,
        optimizer=optimizer,
        device=get_device(),  # Auto-detect best device
        seq_len=512,
        batch_size=16,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        use_amp=True,
        checkpoint_dir='./checkpoints',
        checkpoint_interval=1000,
        log_interval=10,
        use_wandb=False,
        use_tensorboard=True,
        project_name='gpt-transformer'
    )
    
    # Train
    trainer.train(epochs=10, scheduler=scheduler)
    
    print("Training completed!")


if __name__ == '__main__':
    main()


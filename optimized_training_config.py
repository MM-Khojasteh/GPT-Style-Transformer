"""Optimized training configuration for RTX 3050 6GB GPU."""

import torch
import argparse
from src.models.gpt import AdvancedGPT
from src.training.trainer import Trainer
from src.training.optimizers import get_optimizer
from src.training.schedulers import get_scheduler
from src.utils.tokenizer import CharTokenizer
from src.utils.device import get_device, print_gpu_info
from src.utils.config_presets import get_model_config, get_training_config, estimate_model_size
from src.optimizations.gradient_checkpoint import apply_gradient_checkpointing


def main():
    parser = argparse.ArgumentParser(description='Optimized training for RTX 3050 6GB GPU')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                        help='Model size preset')
    parser.add_argument('--data', type=str, default='data/sample_text.txt',
                        help='Path to training data file')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Optimized Training Configuration - {args.preset.upper()} Preset")
    print("=" * 70)
    print()
    
    # Load data
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(data)}")
    print()
    
    # Get model configuration from preset
    model_config = get_model_config(preset=args.preset, vocab_size=tokenizer.vocab_size)
    
    print(f"Model Configuration ({args.preset}):")
    print(f"  embed_dim: {model_config['embed_dim']}")
    print(f"  num_heads: {model_config['num_heads']}")
    print(f"  num_layers: {model_config['num_layers']}")
    print(f"  max_seq_len: {model_config['max_seq_len']}")
    if model_config.get('use_gqa'):
        print(f"  GQA enabled: {model_config.get('num_kv_heads', model_config['num_heads'])} KV heads")
    print()
    
    # Create model
    model = AdvancedGPT(**model_config)
    
    # Estimate model size
    size_info = estimate_model_size(model_config)
    actual_params = model.get_num_params()
    
    print(f"Model Size:")
    print(f"  Parameters: {actual_params:,}")
    print(f"  Estimated Memory: {size_info['total_memory_mb']:.2f} MB ({size_info['total_memory_gb']:.3f} GB)")
    print()
    
    # Get training configuration
    train_cfg = get_training_config(preset=args.preset)
    
    print("Training Configuration:")
    print(f"  Sequence length: {train_cfg['seq_len']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Gradient accumulation: {train_cfg['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")
    if train_cfg.get('use_gradient_checkpointing'):
        print(f"  Gradient checkpointing: Enabled")
    print()
    
    # Check GPU
    print_gpu_info()
    print()
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        optimizer_type='adamw',
        lr=args.lr,
        weight_decay=0.1
    )
    
    # Create scheduler
    total_steps = len(data) // train_cfg['batch_size'] * args.epochs
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        max_lr=args.lr,
        warmup_steps=total_steps // 10,
        total_steps=total_steps
    )
    
    # Apply gradient checkpointing if needed
    if train_cfg.get('use_gradient_checkpointing', False):
        print("Applying gradient checkpointing...")
        apply_gradient_checkpointing(model)
        print("Gradient checkpointing enabled")
        print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=data,
        optimizer=optimizer,
        device=get_device(),
        seq_len=train_cfg['seq_len'],
        batch_size=train_cfg['batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        max_grad_norm=1.0,
        use_amp=True,
        checkpoint_dir='./checkpoints',
        checkpoint_interval=500,
        log_interval=10,
        use_wandb=False,
        use_tensorboard=True,
        project_name=f'gpt-transformer-{args.preset}'
    )
    
    print("=" * 70)
    print("Starting Optimized Training")
    print("=" * 70)
    print()
    
    # Train
    trainer.train(epochs=args.epochs, scheduler=scheduler)
    
    print()
    print("=" * 70)
    print("Training Completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()


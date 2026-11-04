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
from src.utils.env_config import env_str, env_int, env_float, env_bool
from src.optimizations.gradient_checkpoint import apply_gradient_checkpointing


def main():
    parser = argparse.ArgumentParser(description='Optimized training for RTX 3050 6GB GPU')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                        help='Model size preset')
    parser.add_argument('--data', type=str,
                        default=env_str('TRAIN_DATA_PATH', 'Path to the training corpus'),
                        help='Path to training data file')
    parser.add_argument('--epochs', type=int,
                        default=env_int('OPT_TRAIN_EPOCHS', 'Epochs for the optimized profile'),
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float,
                        default=env_float('TRAIN_LEARNING_RATE', 'Peak learning rate for AdamW'),
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
        weight_decay=env_float('TRAIN_WEIGHT_DECAY', 'Decoupled weight decay for AdamW')
    )
    
    # Create scheduler
    total_steps = len(data) // train_cfg['batch_size'] * args.epochs
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        max_lr=args.lr,
        warmup_steps=int(total_steps * env_float(
            'TRAIN_WARMUP_FRACTION', 'Fraction of total steps used for LR warmup')),
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
        max_grad_norm=env_float('TRAIN_MAX_GRAD_NORM', 'Gradient clipping threshold'),
        use_amp=env_bool('TRAIN_USE_AMP', 'Mixed-precision (AMP) training'),
        checkpoint_dir=env_str('CHECKPOINT_DIR', 'Directory for training checkpoints'),
        checkpoint_interval=env_int('OPT_CHECKPOINT_INTERVAL', 'Steps between checkpoints'),
        log_interval=env_int('LOG_INTERVAL', 'Steps between console log lines'),
        use_wandb=env_bool('USE_WANDB', 'Weights & Biases logging'),
        use_tensorboard=env_bool('USE_TENSORBOARD', 'TensorBoard logging'),
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


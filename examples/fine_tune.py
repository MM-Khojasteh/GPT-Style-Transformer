"""Example fine-tuning script with optimized configurations."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from src.models.gpt import AdvancedGPT
from src.training.trainer import Trainer
from src.training.optimizers import get_optimizer
from src.training.schedulers import get_scheduler
from src.utils.tokenizer import CharTokenizer
from src.utils.checkpointing import load_checkpoint
from src.utils.device import get_device, print_gpu_info
from src.utils.config_presets import get_model_config, get_training_config
from src.utils.env_config import env_str, env_int, env_float, env_bool
from src.optimizations.gradient_checkpoint import apply_gradient_checkpointing


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT model')
    parser.add_argument('--base-model', type=str,
                        default=env_str('CHECKPOINT_PATH', 'Base model weights to fine-tune'),
                        help='Path to base model checkpoint')
    parser.add_argument('--data', type=str,
                        default=env_str('FINETUNE_DATA_PATH', 'Path to the fine-tuning corpus'),
                        help='Path to fine-tuning data')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                        help='Model preset (should match base model)')
    parser.add_argument('--epochs', type=int,
                        default=env_int('FINETUNE_EPOCHS', 'Number of fine-tuning epochs'),
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float,
                        default=env_float('FINETUNE_LEARNING_RATE', 'Peak learning rate for fine-tuning'),
                        help='Learning rate for fine-tuning (typically lower than pretraining)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Fine-tuning GPT Model")
    print("=" * 70)
    print()
    
    # Load fine-tuning data
    try:
        with open(args.data, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Fine-tuning data file not found: {args.data}")
        print("Please create the file or specify a different path with --data")
        return
    
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    
    print(f"Fine-tuning data:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Total tokens: {len(data)}")
    print()
    
    # Get model configuration from preset
    model_config = get_model_config(preset=args.preset, vocab_size=tokenizer.vocab_size)
    
    print(f"Model Configuration ({args.preset}):")
    print(f"  embed_dim: {model_config['embed_dim']}")
    print(f"  num_heads: {model_config['num_heads']}")
    print(f"  num_layers: {model_config['num_layers']}")
    print()
    
    # Create model
    model = AdvancedGPT(**model_config)
    
    # Load pretrained weights
    try:
        load_checkpoint(args.base_model, model)
        print(f"Loaded pretrained model from {args.base_model}")
    except FileNotFoundError:
        print(f"Warning: Base model checkpoint not found: {args.base_model}")
        print("Starting fine-tuning from randomly initialized weights")
    except Exception as e:
        print(f"Warning: Error loading checkpoint: {e}")
        print("Starting fine-tuning from randomly initialized weights")
    
    print()
    
    # Get training configuration
    train_config = get_training_config(preset=args.preset)
    
    # Apply gradient checkpointing if needed
    if train_config.get('use_gradient_checkpointing', False):
        print("Applying gradient checkpointing...")
        apply_gradient_checkpointing(model)
        print()
    
    # Create optimizer with lower learning rate for fine-tuning
    optimizer = get_optimizer(
        model,
        optimizer_type='adamw',
        lr=args.lr,
        weight_decay=env_float('TRAIN_WEIGHT_DECAY', 'Decoupled weight decay for AdamW')
    )
    
    # Create scheduler
    total_steps = len(data) // train_config['batch_size'] * args.epochs
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        max_lr=args.lr,
        warmup_steps=int(total_steps * env_float(
            'TRAIN_WARMUP_FRACTION', 'Fraction of total steps used for LR warmup')),
        total_steps=total_steps
    )
    
    # Check GPU availability
    print_gpu_info()
    print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=data,
        optimizer=optimizer,
        device=get_device(),
        seq_len=train_config['seq_len'],
        batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        max_grad_norm=env_float('TRAIN_MAX_GRAD_NORM', 'Gradient clipping threshold'),
        use_amp=env_bool('TRAIN_USE_AMP', 'Mixed-precision (AMP) training'),
        checkpoint_dir=env_str('FINETUNE_CHECKPOINT_DIR', 'Directory for fine-tuned checkpoints'),
        checkpoint_interval=env_int('FINETUNE_CHECKPOINT_INTERVAL', 'Steps between checkpoints'),
        log_interval=env_int('LOG_INTERVAL', 'Steps between console log lines'),
        use_wandb=env_bool('USE_WANDB', 'Weights & Biases logging'),
        use_tensorboard=env_bool('USE_TENSORBOARD', 'TensorBoard logging'),
        project_name='gpt-finetune',
        enable_memory_profiling=True
    )
    
    print("=" * 70)
    print("Starting Fine-tuning")
    print("=" * 70)
    print()
    
    # Fine-tune
    trainer.train(epochs=args.epochs, scheduler=scheduler)
    
    print()
    print("=" * 70)
    print("Fine-tuning completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()


"""Example training script for GPT model with optimized configurations."""

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
from src.utils.device import get_device, print_gpu_info
from src.utils.config_presets import get_model_config, get_training_config, estimate_model_size
from src.utils.env_config import env_str, env_int, env_float, env_bool
from src.optimizations.gradient_checkpoint import apply_gradient_checkpointing
import random


def main():
    parser = argparse.ArgumentParser(description='Train GPT model with optimized configurations')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                        help='Model size preset')
    parser.add_argument('--data', type=str,
                        default=env_str('TRAIN_DATA_PATH', 'Path to the training corpus'),
                        help='Path to training data file')
    parser.add_argument('--epochs', type=int,
                        default=env_int('TRAIN_EPOCHS', 'Number of training epochs'),
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float,
                        default=env_float('TRAIN_LEARNING_RATE', 'Peak learning rate for AdamW'),
                        help='Learning rate')
    parser.add_argument('--val-split', type=float,
                        default=env_float('TRAIN_VAL_SPLIT', 'Validation split ratio'),
                        help='Validation split ratio (0.0 to disable)')
    parser.add_argument('--test-split', type=float,
                        default=env_float('TRAIN_TEST_SPLIT', 'Test split ratio'),
                        help='Test split ratio (0.0 to disable)')
    parser.add_argument('--seed', type=int,
                        default=env_int('TRAIN_RANDOM_SEED', 'Reproducibility seed'),
                        help='Random seed for reproducibility')
    parser.add_argument('--save-splits', action='store_true',
                        help='Save train/validation/test splits to separate files')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Training GPT Model - {args.preset.upper()} Configuration")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load and prepare data
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(data)}")
    
    # Split data into train/validation/test
    if args.val_split > 0 or args.test_split > 0:
        total_split = args.val_split + args.test_split
        if total_split >= 1.0:
            print("Warning: val_split + test_split >= 1.0, adjusting...")
            args.val_split = args.val_split / total_split * 0.2
            args.test_split = args.test_split / total_split * 0.2
            total_split = 0.2
        
        # Shuffle data
        data_shuffled = data.copy()
        random.shuffle(data_shuffled)
        
        # Calculate split indices
        n = len(data_shuffled)
        val_start = int(n * (1 - total_split))
        test_start = int(n * (1 - args.test_split)) if args.test_split > 0 else n
        
        train_data = data_shuffled[:val_start]
        val_data = data_shuffled[val_start:test_start] if args.val_split > 0 else []
        test_data = data_shuffled[test_start:] if args.test_split > 0 else []
        
        print(f"Data split:")
        print(f"  Train: {len(train_data)} tokens ({len(train_data)/n*100:.1f}%)")
        if args.val_split > 0:
            print(f"  Validation: {len(val_data)} tokens ({len(val_data)/n*100:.1f}%)")
        if args.test_split > 0:
            print(f"  Test: {len(test_data)} tokens ({len(test_data)/n*100:.1f}%)")
        
        # Save splits to files if requested
        if args.save_splits:
            data_dir = Path(args.data).parent
            base_name = Path(args.data).stem
            
            # Decode tokens back to text
            train_text = tokenizer.decode(train_data)
            if args.val_split > 0:
                val_text = tokenizer.decode(val_data)
            if args.test_split > 0:
                test_text = tokenizer.decode(test_data)
            
            # Save train data
            train_file = data_dir / f"{base_name}_train.txt"
            with open(train_file, 'w', encoding='utf-8') as f:
                f.write(train_text)
            print(f"  ✓ Saved train data to: {train_file}")
            
            # Save validation data
            if args.val_split > 0:
                val_file = data_dir / f"{base_name}_val.txt"
                with open(val_file, 'w', encoding='utf-8') as f:
                    f.write(val_text)
                print(f"  ✓ Saved validation data to: {val_file}")
            
            # Save test data
            if args.test_split > 0:
                test_file = data_dir / f"{base_name}_test.txt"
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(test_text)
                print(f"  ✓ Saved test data to: {test_file}")
                print(f"  ℹ Use this file for evaluation: --test-data {test_file}")
            print()
        
        # Use train data for training
        data = train_data
    else:
        print("No data split (using all data for training)")
    
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
    train_config = get_training_config(preset=args.preset)
    
    print(f"Training Configuration:")
    print(f"  Sequence length: {train_config['seq_len']}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {train_config['batch_size'] * train_config['gradient_accumulation_steps']}")
    if train_config.get('use_gradient_checkpointing'):
        print(f"  Gradient checkpointing: Enabled")
    print()
    
    # Apply gradient checkpointing if needed
    if train_config.get('use_gradient_checkpointing', False):
        print("Applying gradient checkpointing...")
        apply_gradient_checkpointing(model)
        print("Gradient checkpointing enabled")
        print()
    
    # Create optimizer
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
        checkpoint_dir=env_str('CHECKPOINT_DIR', 'Directory for training checkpoints'),
        checkpoint_interval=env_int('TRAIN_CHECKPOINT_INTERVAL', 'Steps between checkpoints'),
        log_interval=env_int('LOG_INTERVAL', 'Steps between console log lines'),
        use_wandb=env_bool('USE_WANDB', 'Weights & Biases logging'),
        use_tensorboard=env_bool('USE_TENSORBOARD', 'TensorBoard logging'),
        project_name=f'gpt-transformer-{args.preset}'
    )
    
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)
    print()
    
    # Train
    trainer.train(epochs=args.epochs, scheduler=scheduler)
    
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()


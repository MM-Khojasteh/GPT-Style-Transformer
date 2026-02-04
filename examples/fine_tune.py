"""Example fine-tuning script."""

import torch
from src.models.gpt import AdvancedGPT
from src.training.trainer import Trainer
from src.training.optimizers import get_optimizer
from src.training.schedulers import get_scheduler
from src.utils.tokenizer import CharTokenizer
from src.utils.checkpointing import load_checkpoint


def main():
    # Load base model checkpoint
    base_model_path = './checkpoints/base_model.pt'
    
    # Create model
    model = AdvancedGPT(
        vocab_size=50000,  # Adjust based on your tokenizer
        max_seq_len=1024,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        dropout_prob=0.1
    )
    
    # Load pretrained weights
    if base_model_path:
        load_checkpoint(base_model_path, model)
        print(f"Loaded pretrained model from {base_model_path}")
    
    # Load fine-tuning data
    with open('data/fine_tune_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    
    # Create optimizer with lower learning rate for fine-tuning
    optimizer = get_optimizer(
        model,
        optimizer_type='adamw',
        lr=1e-5,  # Lower LR for fine-tuning
        weight_decay=0.1
    )
    
    # Create scheduler
    total_steps = len(data) // 16 * 3  # 3 epochs
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        max_lr=1e-5,
        warmup_steps=total_steps // 10,
        total_steps=total_steps
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=data,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seq_len=1024,
        batch_size=8,
        gradient_accumulation_steps=8,
        checkpoint_dir='./checkpoints_finetuned',
        use_wandb=True,
        project_name='gpt-finetune'
    )
    
    # Fine-tune for a few epochs
    trainer.train(epochs=3, scheduler=scheduler)
    
    print("Fine-tuning completed!")


if __name__ == '__main__':
    main()


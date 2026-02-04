"""Training loop for GPT model."""

import os
import time
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils.checkpointing import save_checkpoint, load_checkpoint
from ..utils.logging import setup_logging, log_metrics


class GPTDataset(Dataset):
    """Dataset for GPT training.
    
    Args:
        data: List of token sequences or single long sequence
        seq_len: Sequence length for training
    """
    
    def __init__(self, data: Union[List[List[int]], List[int]], seq_len: int):
        if isinstance(data[0], int):
            # Single long sequence
            self.data = data
            self.is_long_sequence = True
        else:
            # List of sequences
            self.data = data
            self.is_long_sequence = False
        self.seq_len = seq_len
    
    def __len__(self):
        if self.is_long_sequence:
            return len(self.data) - self.seq_len
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_long_sequence:
            # Return sequence starting at idx
            return torch.tensor(self.data[idx:idx + self.seq_len + 1], dtype=torch.long)
        else:
            # Return sequence at idx (pad or truncate if needed)
            seq = self.data[idx]
            if len(seq) < self.seq_len + 1:
                # Pad
                seq = seq + [0] * (self.seq_len + 1 - len(seq))
            else:
                # Truncate
                seq = seq[:self.seq_len + 1]
            return torch.tensor(seq, dtype=torch.long)


class Trainer:
    """Trainer for GPT model.
    
    Args:
        model: GPT model to train
        train_data: Training data (list of token sequences or long sequence)
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        seq_len: Sequence length for training
        batch_size: Batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        use_amp: Whether to use automatic mixed precision
        amp_dtype: AMP dtype ('float16' or 'bfloat16')
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N steps
        log_interval: Log metrics every N steps
        use_wandb: Whether to use WandB logging
        use_tensorboard: Whether to use TensorBoard logging
        project_name: Project name for logging
        resume_from: Path to checkpoint to resume from
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_data: Union[List[List[int]], List[int]],
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        seq_len: int = 512,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        amp_dtype: str = 'float16',
        checkpoint_dir: str = './checkpoints',
        checkpoint_interval: int = 1000,
        log_interval: int = 10,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        project_name: str = 'gpt-transformer',
        resume_from: Optional[str] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup mixed precision training
        self.use_amp = use_amp and device == 'cuda'
        if self.use_amp:
            if amp_dtype == 'bfloat16':
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == 'float16'))
        else:
            self.scaler = None
        
        # Create dataset and dataloader
        dataset = GPTDataset(train_data, seq_len)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to >0 if you have multiple workers
            pin_memory=True if device == 'cuda' else False
        )
        
        # Setup logging
        self.logger = setup_logging(
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            project_name=project_name
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        return_logits: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """Compute cross-entropy loss.
        
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len + 1)
            return_logits: Whether to return logits
            
        Returns:
            Loss tensor, or (loss, logits) if return_logits=True
        """
        # Split into input and target
        inputs = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()
        
        # Forward pass
        logits = self.model(inputs)  # (batch_size, seq_len, vocab_size)
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        if return_logits:
            return loss, logits
        return loss
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Batch of token sequences
            
        Returns:
            Dictionary of metrics
        """
        batch = batch.to(self.device)
        
        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                loss = self.compute_loss(batch)
        else:
            loss = self.compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights (only after accumulating gradients)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
            
            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        # Compute perplexity
        perplexity = torch.exp(loss * self.gradient_accumulation_steps).item()
        
        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'perplexity': perplexity,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            metrics = self.train_step(batch)
            
            total_loss += metrics['loss']
            total_perplexity += metrics['perplexity']
            num_batches += 1
            self.global_step += 1
            
            # Log metrics
            if self.global_step % self.log_interval == 0:
                log_metrics(
                    self.logger,
                    metrics,
                    step=self.global_step,
                    epoch=self.epoch
                )
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'ppl': f"{metrics['perplexity']:.2f}"
            })
            
            # Save checkpoint
            if self.global_step % self.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity
        }
    
    def train(self, epochs: int, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """Train the model.
        
        Args:
            epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            metrics = self.train_epoch()
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            print(f"Epoch {epoch} completed:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Perplexity: {metrics['perplexity']:.2f}")
            
            # Save best model
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self.save_checkpoint("best_model.pt")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        save_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scaler,
            self.global_step,
            self.epoch,
            self.best_loss
        )
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        state = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scaler
        )
        self.global_step = state.get('global_step', 0)
        self.epoch = state.get('epoch', 0)
        self.best_loss = state.get('best_loss', float('inf'))
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")


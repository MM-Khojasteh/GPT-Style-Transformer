"""Learning rate scheduler utilities."""

from typing import Optional
import torch
import math


class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing learning rate scheduler with warmup.
    
    Linearly increases learning rate from 0 to max_lr during warmup,
    then follows cosine annealing schedule.
    
    Args:
        optimizer: Optimizer
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate (default: max_lr / 10)
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: Optional[float] = None,
        warmup_steps: int = 0,
        total_steps: int = 1000
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr / 10.0
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer)
    
    def get_lr(self):
        """Get learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.max_lr * (self.last_epoch + 1) / self.warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress)) / 2
        
        return [lr for _ in self.optimizer.param_groups]


class LinearDecayWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Linear decay learning rate scheduler with warmup.
    
    Linearly increases learning rate from 0 to max_lr during warmup,
    then linearly decreases to min_lr.
    
    Args:
        optimizer: Optimizer
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
        total_steps: int = 1000
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer)
    
    def get_lr(self):
        """Get learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.max_lr * (self.last_epoch + 1) / self.warmup_steps
        else:
            # Linear decay phase
            progress = (self.last_epoch - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            lr = self.max_lr - (self.max_lr - self.min_lr) * progress
        
        return [lr for _ in self.optimizer.param_groups]


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    max_lr: float = 3e-4,
    min_lr: Optional[float] = None,
    warmup_steps: int = 0,
    total_steps: int = 1000,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type.lower() == 'cosine':
        return CosineAnnealingWithWarmup(
            optimizer=optimizer,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
    elif scheduler_type.lower() == 'linear':
        min_lr = min_lr if min_lr is not None else 0.0
        return LinearDecayWithWarmup(
            optimizer=optimizer,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
    elif scheduler_type.lower() == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


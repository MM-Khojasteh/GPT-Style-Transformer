"""Optimizer utilities for training."""

from typing import Optional, Dict, Any
import torch
import torch.optim as optim


def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    **kwargs
) -> torch.optim.Optimizer:
    """Get optimizer for model.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay
        betas: Beta parameters for Adam/AdamW
        eps: Epsilon for numerical stability
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    params = model.parameters()
    
    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


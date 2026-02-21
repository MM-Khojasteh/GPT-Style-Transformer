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
    """Get optimizer for model with proper weight decay grouping.

    Separates parameters into decay and no-decay groups. Weight decay is NOT
    applied to biases, LayerNorm/RMSNorm parameters, or embeddings, following
    best practices from GPT-2/3 and LLaMA.

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay (only applied to certain parameters)
        betas: Beta parameters for Adam/AdamW
        eps: Epsilon for numerical stability
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to:
        # - Biases (any parameter with 'bias' in name)
        # - LayerNorm/RMSNorm parameters (any parameter with 'norm' in name)
        # - Embedding parameters (any parameter with 'embedding' or 'embed' in name)
        # - Positional encoding parameters
        if any(key in name.lower() for key in ['bias', 'norm', 'embed', 'pos', 'rope']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


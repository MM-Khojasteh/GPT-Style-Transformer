"""Gradient checkpointing utilities for memory-efficient training."""

from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_fn: Optional[Callable] = None
):
    """Apply gradient checkpointing to model.
    
    This replaces forward passes with checkpointed versions that
    recompute activations during backward pass, trading compute for memory.
    
    Args:
        model: Model to apply checkpointing to
        checkpoint_fn: Custom checkpoint function (default: torch.utils.checkpoint.checkpoint)
    """
    if checkpoint_fn is None:
        checkpoint_fn = checkpoint
    
    # Find transformer blocks and wrap their forward passes
    for module in model.modules():
        if isinstance(module, nn.ModuleList):
            # Check if this is a list of transformer blocks
            for i, block in enumerate(module):
                if hasattr(block, 'forward'):
                    # Wrap forward pass
                    original_forward = block.forward
                    
                    def make_checkpointed_forward(block, orig_forward):
                        def checkpointed_forward(*args, **kwargs):
                            return checkpoint_fn(orig_forward, *args, **kwargs)
                        return checkpointed_forward
                    
                    block.forward = make_checkpointed_forward(block, original_forward)


def checkpoint_sequential(
    functions: list,
    segments: int,
    input: torch.Tensor,
    *args,
    **kwargs
) -> torch.Tensor:
    """Checkpoint sequential functions.
    
    Args:
        functions: List of functions to apply sequentially
        segments: Number of segments to split into
        input: Input tensor
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Output tensor
    """
    if segments <= 1:
        # No checkpointing needed
        for func in functions:
            input = func(input, *args, **kwargs)
        return input
    
    # Split functions into segments
    segment_size = len(functions) // segments
    segments_list = [
        functions[i:i + segment_size]
        for i in range(0, len(functions), segment_size)
    ]
    
    # Apply segments with checkpointing
    for segment in segments_list:
        def run_segment(x):
            for func in segment:
                x = func(x, *args, **kwargs)
            return x
        
        input = checkpoint(run_segment, input)
    
    return input


class GradientCheckpointingMixin:
    """Mixin class for modules that support gradient checkpointing."""
    
    def __init__(self, *args, use_checkpointing: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpointing = use_checkpointing
    
    def _checkpointed_forward(self, *args, **kwargs):
        """Checkpointed forward pass."""
        return checkpoint(self._original_forward, *args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward pass with optional checkpointing."""
        if self.use_checkpointing and self.training:
            return self._checkpointed_forward(*args, **kwargs)
        else:
            return self._original_forward(*args, **kwargs)


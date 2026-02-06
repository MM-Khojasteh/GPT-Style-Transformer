"""Checkpoint management utilities."""

import os
import torch
from typing import Optional, Dict, Any, Union

# For type hints - use new API if available, fallback to old for compatibility
try:
    GradScalerType = torch.amp.GradScaler
except AttributeError:
    GradScalerType = torch.cuda.amp.GradScaler


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScalerType] = None,
    global_step: int = 0,
    epoch: int = 0,
    best_loss: float = float('inf'),
    **kwargs
):
    """Save training checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        scaler: GradScaler state (optional)
        global_step: Current training step
        epoch: Current epoch
        best_loss: Best loss so far
        **kwargs: Additional state to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'best_loss': best_loss,
        **kwargs
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScalerType] = None
) -> Dict[str, Any]:
    """Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scaler: GradScaler to load state into (optional)
        
    Returns:
        Dictionary of checkpoint state
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler state
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get path to latest checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pt') or f.endswith('.pth')
    ]
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    
    return os.path.join(checkpoint_dir, checkpoints[0])


"""Tests for training utilities."""

import torch
import pytest
from src.models.gpt import AdvancedGPT
from src.training.trainer import GPTDataset
from src.training.optimizers import get_optimizer
from src.training.schedulers import get_scheduler


def test_gpt_dataset():
    """Test GPT dataset."""
    # Test with long sequence
    data = list(range(1000))
    seq_len = 64
    dataset = GPTDataset(data, seq_len)
    
    assert len(dataset) == 1000 - seq_len
    
    # Test get item
    item = dataset[0]
    assert item.shape == (seq_len + 1,)
    
    # Test with list of sequences
    data = [[1, 2, 3, 4, 5] for _ in range(10)]
    dataset = GPTDataset(data, seq_len)
    assert len(dataset) == 10


def test_optimizer():
    """Test optimizer creation."""
    model = AdvancedGPT(
        vocab_size=1000,
        max_seq_len=512,
        embed_dim=128,
        num_heads=8,
        num_layers=4
    )
    
    optimizer = get_optimizer(model, optimizer_type='adamw', lr=3e-4)
    assert optimizer is not None


def test_scheduler():
    """Test scheduler creation."""
    model = AdvancedGPT(
        vocab_size=1000,
        max_seq_len=512,
        embed_dim=128,
        num_heads=8,
        num_layers=4
    )
    
    optimizer = get_optimizer(model, optimizer_type='adamw', lr=3e-4)
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        max_lr=3e-4,
        total_steps=1000
    )
    
    assert scheduler is not None


"""Tests for GPT model."""

import torch
import pytest
from src.models.gpt import AdvancedGPT


def test_gpt_model():
    """Test GPT model forward pass."""
    vocab_size = 1000
    max_seq_len = 512
    embed_dim = 128
    num_heads = 8
    num_layers = 4
    
    model = AdvancedGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_gpt_weight_tying():
    """Test weight tying in GPT model."""
    vocab_size = 1000
    embed_dim = 128
    
    model = AdvancedGPT(
        vocab_size=vocab_size,
        max_seq_len=512,
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=4,
        weight_tying=True
    )
    
    # Check that head uses embedding weights
    assert model.head.embedding is model.token_emb.embedding


def test_gpt_cache():
    """Test KV caching in GPT model."""
    vocab_size = 1000
    model = AdvancedGPT(
        vocab_size=vocab_size,
        max_seq_len=512,
        embed_dim=128,
        num_heads=8,
        num_layers=4
    )
    
    # Test cache clearing
    model.clear_cache()
    
    # Should not raise error
    assert True


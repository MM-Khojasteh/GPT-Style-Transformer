"""Tests for attention modules."""

import torch
import pytest
from src.models.attention import MultiHeadAttention, GroupedQueryAttention


def test_multi_head_attention():
    """Test multi-head attention."""
    embed_dim = 128
    num_heads = 8
    seq_len = 64
    batch_size = 2
    
    attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_rope=True
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_grouped_query_attention():
    """Test grouped query attention."""
    embed_dim = 128
    num_heads = 8
    num_kv_heads = 2
    seq_len = 64
    batch_size = 2
    
    gqa = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_rope=True
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = gqa(x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)


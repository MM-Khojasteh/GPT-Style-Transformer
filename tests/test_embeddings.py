"""Tests for embedding modules."""

import torch
import pytest
from src.models.embeddings import (
    TokenEmbedding,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RotaryPositionEmbedding,
    ALiBiPositionalEncoding
)


def test_token_embedding():
    """Test token embedding."""
    vocab_size = 1000
    embed_dim = 128
    embedding = TokenEmbedding(vocab_size, embed_dim)
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (2, 10))
    output = embedding(input_ids)
    
    assert output.shape == (2, 10, embed_dim)


def test_learned_positional_embedding():
    """Test learned positional embedding."""
    max_seq_len = 512
    embed_dim = 128
    pos_emb = LearnedPositionalEmbedding(max_seq_len, embed_dim)
    
    # Test forward pass
    output = pos_emb(100)
    assert output.shape == (100, embed_dim)
    
    # Test extension beyond max_seq_len
    output = pos_emb(600)
    assert output.shape == (600, embed_dim)


def test_sinusoidal_positional_embedding():
    """Test sinusoidal positional embedding."""
    max_seq_len = 512
    embed_dim = 128
    pos_emb = SinusoidalPositionalEmbedding(max_seq_len, embed_dim)
    
    # Test forward pass
    output = pos_emb(100)
    assert output.shape == (100, embed_dim)


def test_rotary_position_embedding():
    """Test rotary position embedding."""
    dim = 64
    max_seq_len = 1024
    rope = RotaryPositionEmbedding(dim, max_seq_len)
    
    # Test forward pass
    batch_size, num_heads, seq_len = 2, 8, 128
    Q = torch.randn(batch_size, num_heads, seq_len, dim)
    K = torch.randn(batch_size, num_heads, seq_len, dim)
    
    Q_rot, K_rot = rope(Q, K, seq_len=seq_len)
    
    assert Q_rot.shape == Q.shape
    assert K_rot.shape == K.shape


def test_alibi_positional_encoding():
    """Test ALiBi positional encoding."""
    num_heads = 8
    max_seq_len = 1024
    alibi = ALiBiPositionalEncoding(num_heads, max_seq_len)
    
    # Test bias generation
    bias = alibi.get_bias(128, torch.device('cpu'))
    assert bias.shape == (num_heads, 128, 128)
    
    # Test forward pass
    attention_scores = torch.randn(2, num_heads, 128, 128)
    output = alibi(attention_scores, seq_len=128)
    assert output.shape == attention_scores.shape


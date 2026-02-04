"""Configuration presets for different model sizes and use cases."""

from typing import Dict, Any, Optional


def get_model_config(
    preset: str = 'medium',
    vocab_size: int = 50000,
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get model configuration preset.
    
    Args:
        preset: Preset name ('tiny', 'small', 'medium', 'large', 'xl', 'xxl')
        vocab_size: Vocabulary size
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Dictionary of model configuration parameters
    """
    presets = {
        'tiny': {
            'max_seq_len': 128,
            'embed_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout_prob': 0.1,
            'pos_emb_type': 'learned',
            'pre_norm': True,
            'use_rope': False,
            'use_alibi': False,
            'use_gqa': False,
            'use_flash_attn': False,
            'ffn_dim_mult': 4,
            'ffn_type': 'swiglu',
            'use_moe': False,
            'norm_type': 'layernorm',
            'weight_tying': True,
            'residual_scale': 1.0
        },
        'small': {
            'max_seq_len': 128,
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 4,
            'dropout_prob': 0.1,
            'pos_emb_type': 'learned',
            'pre_norm': True,
            'use_rope': True,
            'use_alibi': False,
            'use_gqa': False,
            'use_flash_attn': False,
            'ffn_dim_mult': 4,
            'ffn_type': 'swiglu',
            'use_moe': False,
            'norm_type': 'layernorm',
            'weight_tying': True,
            'residual_scale': 1.0
        },
        'medium': {
            'max_seq_len': 512,
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'dropout_prob': 0.1,
            'pos_emb_type': 'learned',
            'pre_norm': True,
            'use_rope': True,
            'use_alibi': False,
            'use_gqa': False,
            'use_flash_attn': False,
            'ffn_dim_mult': 4,
            'ffn_type': 'swiglu',
            'use_moe': False,
            'norm_type': 'layernorm',
            'weight_tying': True,
            'residual_scale': 1.0
        },
        'large': {
            'max_seq_len': 1024,
            'embed_dim': 512,
            'num_heads': 16,
            'num_layers': 10,
            'dropout_prob': 0.1,
            'pos_emb_type': 'learned',
            'pre_norm': True,
            'use_rope': True,
            'use_alibi': False,
            'use_gqa': True,
            'num_kv_heads': 8,  # Half of num_heads for GQA
            'use_flash_attn': False,
            'ffn_dim_mult': 4,
            'ffn_type': 'swiglu',
            'use_moe': False,
            'norm_type': 'layernorm',
            'weight_tying': True,
            'residual_scale': 1.0
        },
        'xl': {
            'max_seq_len': 1024,
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 12,
            'dropout_prob': 0.1,
            'pos_emb_type': 'learned',
            'pre_norm': True,
            'use_rope': True,
            'use_alibi': False,
            'use_gqa': True,
            'num_kv_heads': 6,  # Half of num_heads for GQA
            'use_flash_attn': False,
            'ffn_dim_mult': 4,
            'ffn_type': 'swiglu',
            'use_moe': False,
            'norm_type': 'rmsnorm',  # More efficient
            'weight_tying': True,
            'residual_scale': 1.0
        },
        'xxl': {
            'max_seq_len': 2048,
            'embed_dim': 1024,
            'num_heads': 16,
            'num_layers': 16,
            'dropout_prob': 0.1,
            'pos_emb_type': 'learned',
            'pre_norm': True,
            'use_rope': True,
            'use_alibi': False,
            'use_gqa': True,
            'num_kv_heads': 8,  # Half of num_heads for GQA
            'use_flash_attn': False,
            'ffn_dim_mult': 4,
            'ffn_type': 'swiglu',
            'use_moe': False,
            'norm_type': 'rmsnorm',
            'weight_tying': True,
            'residual_scale': 1.0
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
    
    config = presets[preset].copy()
    config['vocab_size'] = vocab_size
    
    # Apply custom overrides
    if custom_config:
        config.update(custom_config)
    
    return config


def get_training_config(
    preset: str = 'medium',
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get training configuration preset.
    
    Args:
        preset: Preset name ('tiny', 'small', 'medium', 'large', 'xl', 'xxl')
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Dictionary of training configuration parameters
    """
    presets = {
        'tiny': {
            'seq_len': 64,
            'batch_size': 8,
            'gradient_accumulation_steps': 2,
            'use_gradient_checkpointing': False
        },
        'small': {
            'seq_len': 64,
            'batch_size': 4,
            'gradient_accumulation_steps': 2,
            'use_gradient_checkpointing': False
        },
        'medium': {
            'seq_len': 256,
            'batch_size': 16,
            'gradient_accumulation_steps': 4,
            'use_gradient_checkpointing': False
        },
        'large': {
            'seq_len': 512,
            'batch_size': 8,
            'gradient_accumulation_steps': 8,
            'use_gradient_checkpointing': False
        },
        'xl': {
            'seq_len': 512,
            'batch_size': 4,
            'gradient_accumulation_steps': 16,
            'use_gradient_checkpointing': True
        },
        'xxl': {
            'seq_len': 1024,
            'batch_size': 2,
            'gradient_accumulation_steps': 32,
            'use_gradient_checkpointing': True
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
    
    config = presets[preset].copy()
    
    # Apply custom overrides
    if custom_config:
        config.update(custom_config)
    
    return config


def estimate_model_size(config: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate model size and memory usage.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dictionary with parameter count and memory estimates
    """
    vocab_size = config['vocab_size']
    embed_dim = config['embed_dim']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    max_seq_len = config['max_seq_len']
    ffn_dim_mult = config.get('ffn_dim_mult', 4)
    
    # Estimate parameters
    # Token embeddings
    token_emb_params = vocab_size * embed_dim
    
    # Positional embeddings
    pos_emb_params = max_seq_len * embed_dim
    
    # Per layer: attention + FFN
    # Attention: Q, K, V, O projections
    attn_params_per_layer = 4 * embed_dim * embed_dim
    # FFN: up, gate, down (for SwiGLU/GeGLU)
    ffn_params_per_layer = 3 * embed_dim * (ffn_dim_mult * embed_dim)
    # Layer norms (2 per layer)
    norm_params_per_layer = 2 * embed_dim
    
    params_per_layer = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    total_layer_params = num_layers * params_per_layer
    
    # Final layer norm
    final_norm_params = embed_dim
    
    # Output head (if not weight tied)
    if config.get('weight_tying', True):
        output_head_params = 0  # Shared with embeddings
    else:
        output_head_params = vocab_size * embed_dim
    
    total_params = (
        token_emb_params +
        pos_emb_params +
        total_layer_params +
        final_norm_params +
        output_head_params
    )
    
    # Memory estimates (FP16 for model, FP32 for optimizer)
    model_memory_mb = (total_params * 2) / (1024**2)  # FP16
    optimizer_memory_mb = (total_params * 2 * 2) / (1024**2)  # AdamW: 2 states, FP32
    
    return {
        'total_parameters': total_params,
        'model_memory_mb': model_memory_mb,
        'optimizer_memory_mb': optimizer_memory_mb,
        'total_memory_mb': model_memory_mb + optimizer_memory_mb,
        'total_memory_gb': (model_memory_mb + optimizer_memory_mb) / 1024
    }


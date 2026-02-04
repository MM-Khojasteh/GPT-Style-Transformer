"""Text generation utilities with various sampling strategies."""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Sample from top-k tokens.
    
    Args:
        logits: Logits tensor of shape (vocab_size,)
        k: Number of top tokens to consider
        
    Returns:
        Filtered logits
    """
    if k <= 0:
        return logits
    
    # Get top-k values
    top_k_values, _ = torch.topk(logits, k)
    
    # Create mask
    mask = logits < top_k_values[-1]
    logits = logits.clone()
    logits[mask] = float('-inf')
    
    return logits


def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Sample using nucleus (top-p) sampling.
    
    Args:
        logits: Logits tensor of shape (vocab_size,)
        p: Cumulative probability threshold
        
    Returns:
        Filtered logits
    """
    if p >= 1.0:
        return logits
    
    # Sort logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # Compute cumulative probabilities
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    # Find cutoff
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Remove tokens
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits = logits.clone()
    logits[indices_to_remove] = float('-inf')
    
    return logits


def generate_text(
    model: torch.nn.Module,
    prompt: List[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: str = 'cuda',
    use_cache: bool = True
) -> List[int]:
    """Generate text from prompt.
    
    Args:
        model: GPT model
        prompt: Prompt token IDs
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling (0 to disable)
        top_p: Top-p (nucleus) sampling (1.0 to disable)
        device: Device to run on
        use_cache: Whether to use KV caching
        
    Returns:
        Generated token IDs (including prompt)
    """
    model.eval()
    tokens = prompt.copy()
    
    # Clear cache
    if hasattr(model, 'clear_cache'):
        model.clear_cache()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get context (truncate if too long)
            max_len = model.config.get('max_seq_len', 1024)
            context = tokens[-max_len:] if len(tokens) > max_len else tokens
            
            # Convert to tensor
            input_ids = torch.tensor([context], device=device)
            
            # Forward pass
            logits = model(input_ids, use_cache=use_cache)
            
            # Get logits for last token
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            else:
                # Greedy decoding
                next_token = next_token_logits.argmax().item()
                tokens.append(next_token)
                continue
            
            # Apply top-k filtering
            if top_k > 0:
                next_token_logits = sample_top_k(next_token_logits, top_k)
            
            # Apply top-p filtering
            if top_p < 1.0:
                next_token_logits = sample_top_p(next_token_logits, top_p)
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(next_token)
    
    return tokens


def generate_with_beam_search(
    model: torch.nn.Module,
    prompt: List[int],
    max_new_tokens: int = 100,
    beam_width: int = 5,
    length_penalty: float = 1.0,
    device: str = 'cuda'
) -> List[int]:
    """Generate text using beam search.
    
    Args:
        model: GPT model
        prompt: Prompt token IDs
        max_new_tokens: Maximum number of tokens to generate
        beam_width: Beam width
        length_penalty: Length penalty factor
        device: Device to run on
        
    Returns:
        Generated token IDs (including prompt)
    """
    model.eval()
    
    # Initialize beams
    beams = [(prompt.copy(), 0.0)]  # (sequence, score)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            candidates = []
            
            for sequence, score in beams:
                # Get context
                max_len = model.config.get('max_seq_len', 1024)
                context = sequence[-max_len:] if len(sequence) > max_len else sequence
                
                # Forward pass
                input_ids = torch.tensor([context], device=device)
                logits = model(input_ids)
                
                # Get logits for last token
                next_token_logits = logits[0, -1, :]
                
                # Get top-k candidates
                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits,
                    min(beam_width, next_token_logits.size(0))
                )
                
                # Add candidates
                for token_id, logit in zip(top_k_indices, top_k_logits):
                    new_sequence = sequence + [token_id.item()]
                    new_score = score + logit.item()
                    candidates.append((new_sequence, new_score))
            
            # Select top beam_width candidates
            candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
            beams = candidates[:beam_width]
    
    # Return best sequence
    best_sequence = max(beams, key=lambda x: x[1] / (len(x[0]) ** length_penalty))[0]
    return best_sequence


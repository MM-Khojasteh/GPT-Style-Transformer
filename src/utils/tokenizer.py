"""Tokenization utilities including character-level and BPE tokenizers."""

from typing import List, Dict, Optional
import re
from collections import Counter


class CharTokenizer:
    """Character-level tokenizer.
    
    Simple tokenizer that maps characters to token IDs.
    
    Args:
        text: Text corpus to build vocabulary from
    """
    
    def __init__(self, text: str):
        # Build vocabulary from unique characters
        unique_chars = sorted(set(text))
        self.vocab = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        return [self.vocab[char] for char in text if char in self.vocab]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return ''.join([self.idx_to_char[idx] for idx in token_ids if idx in self.idx_to_char])


class BPETokenizer:
    """Byte Pair Encoding (BPE) tokenizer.
    
    Simplified BPE implementation. For production use, consider using
    Hugging Face tokenizers library.
    
    Args:
        vocab_size: Target vocabulary size
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: List[tuple] = []
    
    def train(self, text: str):
        """Train BPE tokenizer on text.
        
        Args:
            text: Training text
        """
        # Initialize vocabulary with bytes
        vocab = {chr(i): i for i in range(256)}
        
        # Tokenize text into bytes
        tokens = [chr(ord(c)) for c in text]
        
        # BPE training
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            # Count pairs
            pairs = Counter()
            for j in range(len(tokens) - 1):
                pairs[(tokens[j], tokens[j + 1])] += 1
            
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            
            # Merge
            new_token = ''.join(best_pair)
            vocab[new_token] = 256 + i
            self.merges.append(best_pair)
            
            # Update tokens
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == best_pair:
                    new_tokens.append(new_token)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        
        self.vocab = vocab
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        # Start with bytes
        tokens = [chr(ord(c)) for c in text]
        
        # Apply merges
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(''.join(pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Convert to IDs
        return [self.vocab.get(token, 0) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Reverse vocab lookup
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(idx, '') for idx in token_ids]
        return ''.join(tokens)


def load_hf_tokenizer(tokenizer_name: str):
    """Load Hugging Face tokenizer.
    
    Args:
        tokenizer_name: Name of tokenizer (e.g., 'gpt2')
        
    Returns:
        Hugging Face tokenizer
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except ImportError:
        raise ImportError(
            "transformers library not installed. "
            "Install it with: pip install transformers"
        )


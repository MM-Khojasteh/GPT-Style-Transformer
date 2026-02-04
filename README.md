# GPT-Style Transformer in Python

A comprehensive, production-ready implementation of GPT-style transformer architectures in PyTorch, based on the Julia implementation with modern enhancements.

## Features

### Core Architecture
- **Advanced GPT Model** with configurable architecture
- **Multiple Positional Encodings**: Learned, Sinusoidal, RoPE (Rotary Position Embedding), ALiBi (Attention with Linear Biases)
- **Advanced Attention Mechanisms**:
  - Multi-Head Attention
  - Flash Attention (memory-efficient)
  - Grouped Query Attention (GQA)
  - KV Caching for efficient inference
- **Modern Feedforward Networks**:
  - SwiGLU (Swish-Gated Linear Unit)
  - GeGLU (GELU-Gated Linear Unit)
  - Mixture of Experts (MoE) with GShard routing
- **Flexible Normalization**: LayerNorm and RMSNorm options
- **Pre/Post Normalization**: Configurable transformer block architecture

### Training Infrastructure
- Mixed precision training (FP16/BF16)
- Gradient accumulation and clipping
- Distributed training support (DDP)
- Gradient checkpointing for memory efficiency
- Comprehensive checkpointing system
- WandB and TensorBoard integration
- Custom optimizers and learning rate schedulers

### Utilities
- Character-level and BPE tokenizers
- Advanced text generation with multiple sampling strategies
- Hugging Face integration for datasets and tokenizers
- Model checkpoint management

### Optimizations
- Flash Attention implementation
- Gradient checkpointing
- Model quantization support
- TorchScript compilation support

## Installation

```bash
# Clone the repository
cd python_implementation

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.models.gpt import AdvancedGPT

# Create model
model = AdvancedGPT(
    vocab_size=50000,
    max_seq_len=1024,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    dropout_prob=0.1,
    pos_emb_type="learned",
    use_rope=True,
    weight_tying=True
)

# Forward pass
import torch
input_ids = torch.randint(0, 50000, (2, 128))  # (batch_size, seq_len)
logits = model(input_ids)  # (batch_size, vocab_size, seq_len)
```

### Training

```python
from src.training.trainer import Trainer
from src.utils.tokenizer import CharTokenizer

# Initialize tokenizer and model
tokenizer = CharTokenizer(text_corpus)
model = AdvancedGPT(...)

# Create trainer
trainer = Trainer(
    model=model,
    train_data=train_tokens,
    optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4),
    device="cuda"
)

# Train
trainer.train(epochs=10, batch_size=16, bptt=64)
```

### Text Generation

```python
from src.utils.generation import generate_text

prompt = "The future of AI"
prompt_tokens = tokenizer.encode(prompt)

generated_tokens = generate_text(
    model,
    prompt_tokens,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)

generated_text = tokenizer.decode(generated_tokens)
print(generated_text)
```

## Project Structure

```
python_implementation/
├── src/
│   ├── models/          # Model components
│   ├── training/        # Training infrastructure
│   ├── utils/          # Utilities (tokenizers, generation, etc.)
│   └── optimizations/  # Performance optimizations
├── notebooks/          # Educational notebooks
├── examples/           # Example scripts
└── tests/              # Unit tests
```

## Educational Notebooks

The project includes educational notebooks mirroring the Julia implementation:

1. **0_bigram_model.ipynb**: Simple bigram language model
2. **1_basic_transformer.ipynb**: Basic GPT implementation
3. **2_advanced_transformer.ipynb**: Advanced features (RoPE, Flash Attention, etc.)
4. **3_huggingface_integration.ipynb**: Hugging Face integration

## Key Enhancements Beyond Julia Code

1. **Modern Architecture**:
   - Grouped Query Attention (GQA)
   - Mixture of Experts (MoE)
   - RMSNorm support
   - Better initialization schemes

2. **Training Tools**:
   - Comprehensive checkpointing
   - WandB/TensorBoard logging
   - Distributed training (DDP)
   - Mixed precision training

3. **Performance Optimizations**:
   - Optimized Flash Attention
   - Gradient checkpointing
   - Model quantization
   - Better memory management

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


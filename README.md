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

### Step 1: Install PyTorch with CUDA 12.1 Support (Required for GPU)

**IMPORTANT:** For GPU training, you must install PyTorch with CUDA support. The CPU-only version will not work with GPUs.

**This project uses PyTorch 2.5.1 with CUDA 12.1 support.**

**Quick Setup (Recommended):**
```bash
# Run the GPU setup helper script
python setup_gpu.py

# This will:
# - Check for NVIDIA GPU
# - Verify PyTorch CUDA installation
# - Offer to install PyTorch 2.5.1 with CUDA 12.1 if needed
```

**Manual Installation:**
```bash
# Check your CUDA version first (if you have NVIDIA drivers installed)
nvidia-smi

# Install PyTorch 2.5.1 with CUDA 12.1 support
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation:
python check_gpu.py
# Or manually:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}')"
```

**Note:** If you see `+cpu` in the PyTorch version (e.g., `2.10.0+cpu`), you have the CPU-only version. Uninstall it and reinstall with CUDA support:
```bash
pip uninstall torch torchvision torchaudio
# Then install with CUDA 12.1 as shown above, or run: python setup_gpu.py
```

### Step 2: Install Other Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd GPT-Style-Transformer

# Install all dependencies (including PyTorch with CUDA 12.1)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Optional: Install as a package (recommended for development)
# This allows importing from anywhere, but example scripts work without it
pip install -e .
```

**Note:** The example scripts (`examples/*.py`) automatically handle Python path configuration, so you can run them directly without installing the package. However, installing with `pip install -e .` is recommended for development and allows you to import the modules from any location.

## Running This Project

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0 (with CUDA support recommended for GPU training)
- CUDA-capable GPU (recommended for training, optional for inference)

### Step 1: Prepare Your Data

Place your training text data in the `data/` directory. The project includes a sample file at `data/sample_text.txt` that you can use for testing.

```bash
# The data file should be plain text
# You can use your own text file:
cp your_text_file.txt data/sample_text.txt
```

### Step 2: Train a Model

Use the example training script with different model size presets:

```bash
# Train with medium-sized model (default)
python examples/train_model.py --preset medium --epochs 10 --data data/sample_text.txt

# Train with smaller model (faster, less memory)
python examples/train_model.py --preset small --epochs 10

# Train with larger model (slower, more memory)
python examples/train_model.py --preset large --epochs 10

# Available presets: tiny, small, medium, large, xl, xxl
```

**Training Options:**
- `--preset`: Model size preset (tiny, small, medium, large, xl, xxl)
- `--data`: Path to training data file (default: `data/sample_text.txt`)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 3e-4)

**Training Output:**
- Model checkpoints are saved to `./checkpoints/`
- TensorBoard logs are saved to `./runs/` (view with `tensorboard --logdir runs`)
- The best model is saved as `checkpoints/best_model.pt`

### Step 3: Evaluate Model Performance

Evaluate your trained model with comprehensive metrics:

```bash
# Comprehensive evaluation with all metrics
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --test-data data/sample_text.txt

# Save results to JSON file
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --test-data data/sample_text.txt \
    --output evaluation_results.json

# Use custom prompts for generation testing
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --test-data data/sample_text.txt \
    --prompts "The transformer" "GPT models" "Training large"
```

**Evaluation Metrics Included:**
- **Perplexity**: Model's prediction uncertainty
- **Accuracy**: Top-1, Top-3, Top-5, Top-10 next-token prediction accuracy
- **Entropy Metrics**: Prediction distribution characteristics
- **Coherence Score**: Context understanding quality
- **Prediction Distribution**: Confidence analysis
- **Generation Quality**: Text generation diversity and quality
- **Repetition Analysis**: Repetition patterns in generated text

**Evaluation Options:**
- `--checkpoint`: Path to model checkpoint (default: `./checkpoints/best_model.pt`)
- `--preset`: Model preset (should match training preset)
- `--test-data`: Path to test data file (default: `data/sample_text.txt`)
- `--seq-len`: Sequence length for evaluation (default: 256)
- `--output`: Path to save JSON results (optional)
- `--prompts`: Custom prompts for generation testing (optional)

### Step 4: Generate Text

After training, generate text using a trained checkpoint:

```bash
# Generate text with default settings
python examples/generate_text.py --checkpoint ./checkpoints/best_model.pt --prompt "The future of"

# Customize generation parameters
python examples/generate_text.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --prompt "Artificial intelligence" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9
```

**Generation Options:**
- `--checkpoint`: Path to model checkpoint (default: `./checkpoints/best_model.pt`)
- `--preset`: Model preset (should match the preset used during training)
- `--prompt`: Text prompt for generation
- `--max-tokens`: Maximum number of tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.7, higher = more creative)
- `--top-k`: Top-k sampling (default: 50)
- `--top-p`: Top-p (nucleus) sampling (default: 0.9)

### Step 5: Run Educational Notebooks

Explore the implementation through interactive Jupyter notebooks:

```bash
# Start Jupyter notebook server
jupyter notebook

# Then open notebooks in the notebooks/ directory:
# - 0_bigram_model.ipynb: Simple bigram language model
# - 1_basic_transformer.ipynb: Basic GPT implementation
# - 2_advanced_transformer.ipynb: Advanced features (RoPE, Flash Attention, etc.)
# - 3_huggingface_integration.ipynb: Hugging Face integration
```

### Step 6: Monitor Training (Optional)

**TensorBoard:**
```bash
# View training metrics in real-time
tensorboard --logdir runs
# Then open http://localhost:6006 in your browser
```

**WandB (Optional):**
To use Weights & Biases for experiment tracking, set `use_wandb=True` in the trainer configuration and ensure you have logged in:
```bash
wandb login
```

### Example Workflow

```bash
# 1. Verify GPU setup (if using GPU)
python setup_gpu.py
# Or manually check:
python check_gpu.py

# 2. Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Train a small model for testing
python examples/train_model.py --preset small --epochs 5 --data data/sample_text.txt

# 4. Evaluate model performance comprehensively
python comprehensive_evaluation.py --checkpoint ./checkpoints/best_model.pt --preset small

# 5. Generate text from the trained model
python examples/generate_text.py --checkpoint ./checkpoints/best_model.pt --prompt "The transformer"

# 6. Monitor training progress (in another terminal)
tensorboard --logdir runs
```

### Troubleshooting

**GPU Not Detected / CUDA Not Available:**

If you see "CUDA Available: False" or "No CUDA-capable GPU detected":

1. **Check PyTorch Installation:**
   ```bash
   python check_gpu.py
   # Or manually:
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

2. **If PyTorch shows `+cpu` version:**
   - You have the CPU-only version installed
   - Uninstall: `pip uninstall torch torchvision torchaudio`
   - Reinstall with CUDA 12.1 support:
     ```bash
     pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
         --extra-index-url https://download.pytorch.org/whl/cu121
     ```
   - Or run: `python setup_gpu.py`

3. **Verify NVIDIA GPU and Drivers:**
   ```bash
   # Check if GPU is detected by system
   nvidia-smi
   ```
   - If `nvidia-smi` fails, install/update NVIDIA drivers from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)

4. **Check CUDA Toolkit:**
   - This project requires CUDA 12.1 support
   - Ensure CUDA toolkit 12.1 or compatible version is installed
   - PyTorch 2.5.1+cu121 requires CUDA 12.1 drivers
   - Check CUDA version: `nvidia-smi` (look for "CUDA Version" in the output)

5. **Common Issues:**
   - **Virtual environment:** Make sure you're using the correct virtual environment where PyTorch with CUDA is installed
   - **Multiple PyTorch installations:** Check for conflicting installations: `pip list | grep torch`
   - **CUDA version mismatch:** PyTorch CUDA version must match or be compatible with your system's CUDA toolkit

**Out of Memory Errors:**
- Use a smaller model preset (`--preset tiny` or `--preset small`)
- Reduce batch size in the training configuration
- Enable gradient checkpointing (automatically enabled for larger models)
- Check GPU memory: `nvidia-smi`

**Checkpoint Loading:**
- Ensure the `--preset` argument matches the preset used during training
- Verify the checkpoint file exists at the specified path

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


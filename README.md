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
- `--val-split`: Validation split ratio (default: 0.1, set to 0.0 to disable)
- `--test-split`: Test split ratio (default: 0.1, set to 0.0 to disable)
- `--save-splits`: Save train/validation/test splits to separate files
- `--seed`: Random seed for reproducibility (default: 42)

**Recommended Training with Data Split:**
```bash
# Train with data split and save splits to files
python examples/train_model.py \
    --preset medium \
    --epochs 10 \
    --data data/sample_text.txt \
    --val-split 0.1 \
    --test-split 0.1 \
    --save-splits

# This will create:
# - data/sample_text_train.txt (80% of data)
# - data/sample_text_val.txt (10% of data)
# - data/sample_text_test.txt (10% of data)
```

**Training Output:**
- Model checkpoints are saved to `./checkpoints/`
- TensorBoard logs are saved to `./runs/` (view with `tensorboard --logdir runs`)
- The best model is saved as `checkpoints/best_model.pt`

### Step 3: Evaluate Model Performance

Evaluate your trained model with comprehensive metrics:

**Recommended: Evaluate on Separate Test Set**

```bash
# If you used --save-splits during training, evaluate on test set:
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --test-data data/sample_text_test.txt \
    --train-data data/sample_text_train.txt

# This ensures:
# - Realistic evaluation on unseen data
# - Tokenizer consistency with training
# - Overlap detection between train/test sets
```

**Other Evaluation Options:**

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
    --test-data data/sample_text_test.txt \
    --train-data data/sample_text_train.txt \
    --output evaluation_results.json

# Use custom prompts for generation testing
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --test-data data/sample_text_test.txt \
    --prompts "The transformer" "GPT models" "Training large"
```

**Note:** If you evaluate on data that includes training examples, metrics will be higher but less realistic. For proper evaluation, always use a separate test set that wasn't seen during training.

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
- `--train-data`: Path to training data file (optional, for tokenizer consistency and overlap detection)
- `--seq-len`: Sequence length for evaluation (default: 256)
- `--output`: Path to save JSON results (optional)
- `--prompts`: Custom prompts for generation testing (optional)

**Understanding Evaluation Results:**
- Lower metrics on test set are **expected** and indicate realistic evaluation
- If test set contains unseen data, metrics will be lower than training metrics
- This is normal and shows the model's generalization ability
- For best results, always use a separate test set that wasn't seen during training

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

# 3. Train a model with data split (recommended)
python examples/train_model.py \
    --preset small \
    --epochs 5 \
    --data data/sample_text.txt \
    --val-split 0.1 \
    --test-split 0.1 \
    --save-splits

# This creates: data/sample_text_train.txt, data/sample_text_val.txt, data/sample_text_test.txt

# 4. Evaluate model performance on test set (realistic evaluation)
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset small \
    --test-data data/sample_text_test.txt \
    --train-data data/sample_text_train.txt

# 5. Generate text from the trained model
python examples/generate_text.py --checkpoint ./checkpoints/best_model.pt --prompt "The transformer"

# 6. Monitor training progress (in another terminal)
tensorboard --logdir runs
```

## Complete Training and Evaluation Guide

This comprehensive guide covers everything you need to know about training and evaluating GPT models with this project.

### Table of Contents
1. [Data Preparation](#data-preparation)
2. [Training Process](#training-process)
3. [Model Evaluation](#model-evaluation)
4. [Understanding Results](#understanding-results)
5. [Best Practices](#best-practices)
6. [Advanced Topics](#advanced-topics)

### Data Preparation

#### Data Format
- **Format**: Plain text files (`.txt`)
- **Encoding**: UTF-8
- **Location**: Place data files in the `data/` directory

#### Data Requirements
- **Minimum Size**: For meaningful results, use at least 1,000-5,000 tokens
- **Recommended Size**: 10,000+ tokens for better generalization
- **Quality**: Clean, consistent text data works best

#### Data Split Strategy

**Why Split Data?**
- **Training Set**: Used to train the model (typically 70-80%)
- **Validation Set**: Used for hyperparameter tuning and early stopping (typically 10-15%)
- **Test Set**: Used for final evaluation on unseen data (typically 10-15%)

**Recommended Split:**
```bash
# 80% training, 10% validation, 10% test
--val-split 0.1 --test-split 0.1
```

**Important Notes:**
- Always use a separate test set that the model never sees during training
- Shuffle data before splitting to ensure random distribution
- Keep the same split for consistent comparisons across experiments

### Training Process

#### Basic Training

**Quick Start:**
```bash
python examples/train_model.py \
    --preset medium \
    --epochs 10 \
    --data data/sample_text.txt
```

#### Model Presets

Choose a preset based on your resources and needs:

| Preset | Parameters | Memory | Use Case |
|--------|-----------|--------|----------|
| `tiny` | ~100K | <1GB | Quick testing, learning |
| `small` | ~500K | 1-2GB | Small datasets, prototyping |
| `medium` | ~6M | 2-4GB | Medium datasets, balanced |
| `large` | ~50M | 4-8GB | Large datasets, better quality |
| `xl` | ~200M | 8-16GB | Very large datasets |
| `xxl` | ~500M+ | 16GB+ | Production models |

#### Training with Data Split (Recommended)

**Complete Training Command:**
```bash
python examples/train_model.py \
    --preset medium \
    --epochs 10 \
    --data data/sample_text.txt \
    --val-split 0.1 \
    --test-split 0.1 \
    --save-splits \
    --seed 42 \
    --lr 3e-4
```

**What This Does:**
1. Splits data into train/validation/test sets
2. Saves splits to separate files for later use
3. Trains model only on training set
4. Uses consistent random seed for reproducibility

**Output Files:**
- `data/sample_text_train.txt` - Training data (80%)
- `data/sample_text_val.txt` - Validation data (10%)
- `data/sample_text_test.txt` - Test data (10%)
- `checkpoints/best_model.pt` - Best model checkpoint
- `checkpoints/checkpoint_epoch_N.pt` - Per-epoch checkpoints

#### Training Hyperparameters

**Learning Rate:**
- **Default**: `3e-4` (0.0003)
- **Small models**: Can use higher rates (5e-4 to 1e-3)
- **Large models**: Use lower rates (1e-4 to 3e-4)
- **Too high**: Training becomes unstable, loss spikes
- **Too low**: Training is slow, may get stuck in local minima

**Epochs:**
- **Small datasets**: 10-20 epochs
- **Medium datasets**: 5-10 epochs
- **Large datasets**: 3-5 epochs
- **Monitor**: Stop if validation loss stops improving

**Batch Size:**
- Automatically set based on preset
- Larger batch = more stable gradients, more memory
- Smaller batch = less memory, more gradient noise

#### Monitoring Training

**TensorBoard (Recommended):**
```bash
# Start TensorBoard in a separate terminal
tensorboard --logdir runs

# Open browser to http://localhost:6006
# View:
# - Training loss over time
# - Perplexity trends
# - Learning rate schedule
```

**Key Metrics to Watch:**
- **Loss**: Should decrease steadily
- **Perplexity**: Lower is better (aim for <50 for good models)
- **Learning Rate**: Should follow cosine schedule
- **Gradient Norm**: Should be stable (clipped at 1.0)

**Early Stopping:**
- Monitor validation loss
- Stop if validation loss doesn't improve for 3-5 epochs
- Save best model automatically (already implemented)

### Model Evaluation

#### Evaluation on Test Set (Recommended)

**Complete Evaluation Command:**
```bash
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --test-data data/sample_text_test.txt \
    --train-data data/sample_text_train.txt \
    --output evaluation_results.json
```

**Why Use Separate Test Set?**
- **Realistic Metrics**: Shows how model performs on unseen data
- **Generalization**: Tests model's ability to generalize
- **Fair Comparison**: Allows fair comparison between different models

#### Evaluation Metrics Explained

**1. Perplexity**
- **What it is**: Average uncertainty in next-token prediction
- **Lower is better**: Lower perplexity = better predictions
- **Interpretation**:
  - < 20: Excellent
  - 20-50: Good
  - 50-100: Fair
  - > 100: Needs improvement
- **Note**: Test perplexity will be higher than training perplexity (this is normal!)

**2. Accuracy Metrics**
- **Top-1 Accuracy**: Percentage of correct next-token predictions
- **Top-K Accuracy**: Percentage where correct token is in top-K predictions
- **Interpretation**:
  - Compare to random baseline (1/vocab_size)
  - 2-5x better than random = learning
  - 10x+ better = strong learning

**3. Coherence Score**
- **What it is**: How confident the model is in its predictions
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - > 0.5: High coherence
  - 0.2-0.5: Moderate coherence
  - < 0.2: Low coherence (normal for small vocabularies)

**4. Repetition Metrics**
- **Unique Tokens Ratio**: Percentage of unique tokens in generated text
- **Interpretation**:
  - > 0.5: Good diversity
  - 0.2-0.5: Moderate diversity
  - < 0.2: High repetition

**5. Generation Quality**
- **Average Length**: Average tokens per generation
- **Unique Ratio**: Diversity in generated text
- **Success Rate**: Percentage of successful generations

#### Understanding Evaluation Results

**Expected Behavior:**

1. **Test Metrics vs Training Metrics:**
   - Test perplexity > Training perplexity ✓ (Normal)
   - Test accuracy < Training accuracy ✓ (Normal)
   - This indicates realistic evaluation on unseen data

2. **Small Vocabulary Effects:**
   - Low coherence scores are normal (< 0.2)
   - Repetition metrics may be limited
   - Consider BPE or word-level tokenization for improvement

3. **Small Dataset Effects:**
   - Higher variance in metrics
   - May show overfitting (large train/test gap)
   - Use larger datasets for better results

**Red Flags:**
- Test perplexity much higher than training (>2x) = Overfitting
- Accuracy near random baseline = Model not learning
- Coherence score = 0 = Model completely uncertain

### Best Practices

#### Training Best Practices

1. **Always Use Data Split:**
   ```bash
   --val-split 0.1 --test-split 0.1 --save-splits
   ```

2. **Use Consistent Random Seed:**
   ```bash
   --seed 42  # For reproducibility
   ```

3. **Monitor Training Progress:**
   - Use TensorBoard to track metrics
   - Watch for overfitting (validation loss stops decreasing)
   - Save checkpoints regularly

4. **Start Small:**
   - Begin with `tiny` or `small` preset
   - Verify everything works
   - Scale up gradually

5. **Hyperparameter Tuning:**
   - Adjust learning rate if training is unstable
   - Increase epochs if loss is still decreasing
   - Use validation set for hyperparameter selection

#### Evaluation Best Practices

1. **Always Evaluate on Separate Test Set:**
   ```bash
   --test-data data/sample_text_test.txt
   --train-data data/sample_text_train.txt
   ```

2. **Use Tokenizer Consistency:**
   - Provide `--train-data` to ensure vocabulary matches
   - Prevents vocabulary mismatch errors

3. **Save Evaluation Results:**
   ```bash
   --output evaluation_results.json
   ```

4. **Compare Multiple Models:**
   - Use same test set for fair comparison
   - Document hyperparameters used
   - Track results over time

5. **Interpret Results Correctly:**
   - Lower test metrics are expected and good
   - Indicates realistic evaluation
   - Shows generalization ability

### Advanced Topics

#### Custom Training Configuration

For advanced users, you can modify training parameters directly in the code:

```python
from src.training.trainer import Trainer
from src.utils.config_presets import get_model_config, get_training_config

# Get base config
model_config = get_model_config(preset='medium', vocab_size=50000)
train_config = get_training_config(preset='medium')

# Customize
train_config['batch_size'] = 32
train_config['seq_len'] = 512
```

#### Distributed Training

For multi-GPU training (future feature):
```python
# Will be available in future versions
trainer = Trainer(..., use_ddp=True, num_gpus=4)
```

#### Fine-tuning Pre-trained Models

```bash
# Load pre-trained checkpoint and continue training
python examples/train_model.py \
    --preset medium \
    --data new_data.txt \
    --resume-from ./checkpoints/pretrained_model.pt
```

#### Evaluation on Custom Metrics

You can extend the evaluation script to add custom metrics:

```python
# In comprehensive_evaluation.py
def compute_custom_metric(self, text: str) -> float:
    # Your custom metric here
    pass
```

### Common Workflows

#### Workflow 1: Quick Test
```bash
# 1. Train small model
python examples/train_model.py --preset tiny --epochs 5

# 2. Evaluate
python comprehensive_evaluation.py --checkpoint ./checkpoints/best_model.pt --preset tiny

# 3. Generate sample
python examples/generate_text.py --checkpoint ./checkpoints/best_model.pt --prompt "Hello"
```

#### Workflow 2: Production Training
```bash
# 1. Prepare data and split
python examples/train_model.py \
    --preset medium \
    --epochs 20 \
    --data data/large_corpus.txt \
    --val-split 0.1 \
    --test-split 0.1 \
    --save-splits \
    --seed 42

# 2. Monitor training (separate terminal)
tensorboard --logdir runs

# 3. Evaluate on test set
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --test-data data/large_corpus_test.txt \
    --train-data data/large_corpus_train.txt \
    --output final_evaluation.json

# 4. Generate production samples
python examples/generate_text.py \
    --checkpoint ./checkpoints/best_model.pt \
    --preset medium \
    --prompt "Your prompt here" \
    --max-tokens 200 \
    --temperature 0.8
```

#### Workflow 3: Hyperparameter Tuning
```bash
# Train multiple models with different hyperparameters
for lr in 1e-4 3e-4 5e-4; do
    python examples/train_model.py \
        --preset small \
        --epochs 10 \
        --lr $lr \
        --data data/sample_text.txt \
        --val-split 0.1 \
        --test-split 0.1 \
        --save-splits
done

# Evaluate all models
for checkpoint in checkpoints/best_model_*.pt; do
    python comprehensive_evaluation.py \
        --checkpoint $checkpoint \
        --preset small \
        --test-data data/sample_text_test.txt \
        --output "eval_$(basename $checkpoint).json"
done
```

### Troubleshooting Training and Evaluation

**Training Issues:**

1. **Loss Not Decreasing:**
   - Check learning rate (may be too low)
   - Verify data is loaded correctly
   - Check if model is too small for dataset

2. **Loss Exploding:**
   - Learning rate too high
   - Reduce learning rate or use gradient clipping
   - Check for NaN values in data

3. **Out of Memory:**
   - Use smaller preset (`tiny` or `small`)
   - Reduce batch size
   - Enable gradient checkpointing

4. **Overfitting:**
   - Increase dataset size
   - Add more dropout
   - Use early stopping
   - Regularize more (weight decay)

**Evaluation Issues:**

1. **Vocabulary Mismatch:**
   - Always provide `--train-data` when evaluating
   - Ensures tokenizer uses same vocabulary

2. **Metrics Too Low:**
   - This is normal for test set!
   - Lower metrics indicate realistic evaluation
   - Compare to random baseline, not training metrics

3. **File Not Found Errors:**
   - Use `--save-splits` during training
   - Check file paths are correct
   - Verify files exist before evaluation

### Summary

**Training Checklist:**
- [ ] Data prepared and placed in `data/` directory
- [ ] Appropriate model preset selected
- [ ] Data split configured (`--val-split`, `--test-split`)
- [ ] `--save-splits` used to save test set
- [ ] Training monitored with TensorBoard
- [ ] Best checkpoint saved

**Evaluation Checklist:**
- [ ] Using separate test set (`--test-data`)
- [ ] Providing training data for tokenizer (`--train-data`)
- [ ] Results saved to JSON (`--output`)
- [ ] Metrics interpreted correctly (lower is expected)
- [ ] Results documented for comparison

**Remember:**
- Always use a separate test set for final evaluation
- Lower test metrics are expected and indicate realistic evaluation
- Monitor training to catch issues early
- Save all results for future comparison

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


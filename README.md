# GPT-Style Transformer

A from-scratch GPT-style transformer in PyTorch — modern architecture features with a complete training, evaluation, and generation pipeline.

## Features

- **Architecture**: learned / sinusoidal / RoPE / ALiBi positional encodings; multi-head attention, Flash Attention (via `scaled_dot_product_attention`), Grouped Query Attention (GQA), KV caching; SwiGLU / GeGLU feed-forward blocks; Mixture-of-Experts (GShard routing); LayerNorm / RMSNorm; pre-norm blocks; weight tying
- **Training**: mixed precision (AMP), gradient accumulation + clipping, cosine schedule with warmup, validation with best-model selection, TensorBoard / WandB logging
- **Utilities**: character-level and BPE tokenizers, top-k / top-p / beam-search generation, checkpointing, quantization, model-size presets (`tiny` → `xxl`)
- **Notebooks**: bigram baseline, basic transformer, advanced features, Hugging Face integration

## Setup

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Optional helpers
python setup_gpu.py    # verify/install PyTorch 2.5.1 + CUDA 12.1
python check_gpu.py    # verify CUDA availability

cp .env.example .env   # required before running anything (see below)
```

Requires Python ≥ 3.8 and PyTorch ≥ 2.0. A CUDA GPU is recommended for training.

## Configuration

All runtime paths, hyperparameters, sampling thresholds, and toggles are read
from environment variables loaded from a local `.env` file, which is
intentionally **not committed**. Copy the template, then fill in values:

```bash
cp .env.example .env
```

Every variable is documented in [`.env.example`](.env.example); CLI flags
(e.g. `--epochs 20`) override the environment.

## Usage

```bash
# Train (splits data and saves the split files)
python examples/train_model.py --preset medium --epochs 10 \
    --val-split 0.1 --test-split 0.1 --save-splits

# Evaluate on the held-out test set
python comprehensive_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt --preset medium \
    --test-data data/sample_text_test.txt --train-data data/sample_text_train.txt \
    --output evaluation_results.json

# Generate text
python examples/generate_text.py --prompt "The future of" \
    --temperature 0.8 --max-tokens 200

# Fine-tune from a checkpoint
python examples/fine_tune.py --epochs 3

# Monitor training
tensorboard --logdir runs
```

Checkpoints are written to `./checkpoints/` (best model by validation loss).
Training data goes in `data/` — see `data/README.md`. WandB is optional:
set `USE_WANDB=true` and `WANDB_API_KEY` in `.env`, or run `wandb login`.

### Model presets

| Preset | Use case |
|---|---|
| `tiny` / `small` | quick tests, prototyping |
| `medium` *(default)* | balanced quality / cost |
| `large` / `xl` / `xxl` | larger data, more GPU memory |

Presets are defined in `src/utils/config_presets.py`.

### Library usage

```python
import torch
from src.models.gpt import AdvancedGPT

model = AdvancedGPT(vocab_size=65, max_seq_len=512, embed_dim=256,
                    num_heads=8, num_layers=6)
logits = model(torch.randint(0, 65, (2, 128)))
```

## Project Structure

```
├── src/
│   ├── models/          # GPT model, attention, embeddings, feed-forward, blocks
│   ├── training/        # trainer, optimizers, LR schedulers
│   ├── optimizations/   # flash attention, gradient checkpointing, quantization
│   └── utils/           # tokenizers, generation, checkpointing, logging, config
├── examples/            # train / generate / fine-tune entry points
├── notebooks/           # educational notebooks
├── tests/               # unit tests (pytest)
├── comprehensive_evaluation.py
├── evaluate_model.py
└── benchmark_configs.py
```

## Testing

```bash
pytest tests/
```

## License

MIT

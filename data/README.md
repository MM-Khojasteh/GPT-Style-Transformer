# Data Directory

Training, validation, and fine-tuning corpora live here. **All files in this
directory are git-ignored** — datasets are intentionally kept out of the
public repository.

Expected files (see `.env.example` for the matching variables):

| File | Environment variable |
|---|---|
| `sample_text.txt` — training corpus | `TRAIN_DATA_PATH` |
| `sample_text_train.txt` / `_val.txt` / `_test.txt` — produced by `examples/train_model.py --save-splits` | `TRAIN_DATA_PATH` (source) |
| `fine_tune_text.txt` — fine-tuning corpus | `FINETUNE_DATA_PATH` |

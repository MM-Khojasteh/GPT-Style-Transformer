# Checkpoints Directory

Model weights produced during training are written here and are
**git-ignored** — binaries are never committed.

Point `CHECKPOINT_PATH` in your local `.env` at a checkpoint (for example
`best_model.pt`) to load it for generation, fine-tuning, or evaluation.
Fine-tuning runs write to the separate `checkpoints_finetuned/` directory
(`FINETUNE_CHECKPOINT_DIR`).

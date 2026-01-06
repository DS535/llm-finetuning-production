"""
Training utilities including custom trainers and callbacks for Colab environment.
"""

from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Optional, Dict
import os
import shutil


class ColabCheckpointCallback(TrainerCallback):
    """
    Callback to save checkpoints to Google Drive periodically.
    Prevents loss of progress if Colab disconnects.
    """

    def __init__(self, drive_path: str = "/content/drive/MyDrive/checkpoints", save_freq: int = 500):
        """
        Args:
            drive_path: Path to Google Drive checkpoint directory
            save_freq: Save to Drive every N steps
        """
        self.drive_path = drive_path
        self.save_freq = save_freq
        os.makedirs(drive_path, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        """Copy checkpoint to Drive after saving."""
        if state.global_step % self.save_freq == 0:
            checkpoint_folder = f"checkpoint-{state.global_step}"
            source = os.path.join(args.output_dir, checkpoint_folder)
            destination = os.path.join(self.drive_path, checkpoint_folder)

            if os.path.exists(source):
                try:
                    if os.path.exists(destination):
                        shutil.rmtree(destination)
                    shutil.copytree(source, destination)
                    print(f"✓ Checkpoint saved to Drive: {destination}")
                except Exception as e:
                    print(f"✗ Failed to save to Drive: {e}")


class MemoryEfficientTrainer(Trainer):
    """
    Custom trainer with memory optimization features.
    """

    def __init__(self, *args, enable_memory_tracking: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_memory_tracking = enable_memory_tracking

    def training_step(self, model, inputs):
        """Override training step with memory tracking."""
        if self.enable_memory_tracking:
            import torch
            torch.cuda.empty_cache()

        return super().training_step(model, inputs)


def create_training_args(
    output_dir: str,
    task_type: str = "causal_lm",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    fp16: bool = True,
    bf16: bool = False,
    **kwargs
) -> TrainingArguments:
    """
    Create training arguments with sensible defaults for 8GB GPU.

    Args:
        output_dir: Output directory for checkpoints
        task_type: Type of task
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        fp16: Use FP16 mixed precision (for T4)
        bf16: Use BF16 mixed precision (for A100)
        **kwargs: Additional TrainingArguments parameters

    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=fp16,
        bf16=bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        save_total_limit=2,  # Keep only 2 checkpoints
        remove_unused_columns=False,
        report_to=["wandb", "tensorboard"],
        **kwargs
    )

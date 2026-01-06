"""
Memory optimization utilities for LLM finetuning on limited GPU memory.

This module provides functions to monitor GPU memory usage, find optimal batch sizes,
and manage memory during training.
"""

import torch
import gc
from typing import Optional, Callable


def print_gpu_utilization(device_id: int = 0):
    """
    Print current GPU memory utilization.

    Args:
        device_id: CUDA device ID to check (default: 0)
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return

    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3

    print(f"GPU {device_id} Memory Status:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")

    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    print(f"  Total:     {total_memory:.2f} GB")
    print(f"  Usage:     {(allocated/total_memory*100):.1f}%")


def clear_memory_cache():
    """
    Clear GPU and Python memory cache.
    Call this between experiments or when switching models.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("Memory cache cleared.")


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """
    Get GPU memory information as a dictionary.

    Args:
        device_id: CUDA device ID to check

    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {
            "allocated": 0,
            "reserved": 0,
            "max_allocated": 0,
            "total": 0,
            "available": 0
        }

    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    available = total - allocated

    return {
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "max_allocated": round(max_allocated, 2),
        "total": round(total, 2),
        "available": round(available, 2)
    }


def find_optimal_batch_size(
    model: torch.nn.Module,
    sample_input: dict,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    device: str = "cuda"
) -> int:
    """
    Find the maximum batch size that fits in GPU memory.

    Uses binary search to efficiently find the optimal batch size.

    Args:
        model: PyTorch model
        sample_input: Sample input dictionary (tokenized)
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        device: Device to test on

    Returns:
        Maximum batch size that fits in memory
    """
    if not torch.cuda.is_available():
        print("CUDA not available, returning min_batch_size")
        return min_batch_size

    model.eval()
    model.to(device)

    def test_batch_size(batch_size: int) -> bool:
        """Test if batch size fits in memory."""
        try:
            clear_memory_cache()

            # Create batch
            batch = {
                k: v.repeat(batch_size, 1).to(device) if v.dim() > 1 else v.repeat(batch_size).to(device)
                for k, v in sample_input.items()
            }

            # Forward pass
            with torch.no_grad():
                _ = model(**batch)

            # Backward pass simulation (doubles memory usage)
            # This is conservative but safer
            clear_memory_cache()
            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                clear_memory_cache()
                return False
            raise e

    # Binary search for optimal batch size
    left, right = min_batch_size, max_batch_size
    optimal = min_batch_size

    while left <= right:
        mid = (left + right) // 2

        if test_batch_size(mid):
            optimal = mid
            left = mid + 1
        else:
            right = mid - 1

    clear_memory_cache()
    print(f"Optimal batch size found: {optimal}")
    return optimal


def setup_gradient_checkpointing(model: torch.nn.Module) -> torch.nn.Module:
    """
    Enable gradient checkpointing to reduce memory usage.

    Trade-off: ~20-30% slower training for 30-50% less memory.

    Args:
        model: Model to enable checkpointing on

    Returns:
        Model with gradient checkpointing enabled
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    else:
        print("Warning: Model doesn't support gradient checkpointing.")

    return model


def estimate_model_memory(
    num_parameters: int,
    precision: str = "fp32",
    include_gradients: bool = True,
    include_optimizer: bool = True
) -> float:
    """
    Estimate GPU memory required for a model.

    Args:
        num_parameters: Number of model parameters
        precision: Model precision (fp32, fp16, bf16, int8, int4)
        include_gradients: Whether to include gradient memory
        include_optimizer: Whether to include optimizer state memory (Adam uses 2x params)

    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }

    param_bytes = bytes_per_param.get(precision, 4)

    # Model parameters
    memory = num_parameters * param_bytes

    # Gradients (same size as parameters)
    if include_gradients:
        memory += num_parameters * param_bytes

    # Optimizer state (Adam stores momentum and variance, 2x parameters in fp32)
    if include_optimizer:
        memory += num_parameters * 2 * 4  # Always fp32 for optimizer

    memory_gb = memory / 1024**3

    print(f"Estimated memory for {num_parameters/1e6:.1f}M params ({precision}):")
    print(f"  Model: {num_parameters * param_bytes / 1024**3:.2f} GB")
    if include_gradients:
        print(f"  Gradients: {num_parameters * param_bytes / 1024**3:.2f} GB")
    if include_optimizer:
        print(f"  Optimizer: {num_parameters * 2 * 4 / 1024**3:.2f} GB")
    print(f"  Total: {memory_gb:.2f} GB")

    return memory_gb


class MemoryTracker:
    """Context manager to track memory usage of code blocks."""

    def __init__(self, name: str = "Operation", device_id: int = 0):
        self.name = name
        self.device_id = device_id
        self.start_allocated = 0
        self.start_reserved = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            self.start_allocated = torch.cuda.memory_allocated(self.device_id)
            self.start_reserved = torch.cuda.memory_reserved(self.device_id)
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated(self.device_id)
            end_reserved = torch.cuda.memory_reserved(self.device_id)
            peak_allocated = torch.cuda.max_memory_allocated(self.device_id)

            delta_allocated = (end_allocated - self.start_allocated) / 1024**3
            delta_reserved = (end_reserved - self.start_reserved) / 1024**3
            peak_gb = peak_allocated / 1024**3

            print(f"\n{self.name} Memory Usage:")
            print(f"  Delta Allocated: {delta_allocated:+.2f} GB")
            print(f"  Delta Reserved:  {delta_reserved:+.2f} GB")
            print(f"  Peak Allocated:  {peak_gb:.2f} GB")

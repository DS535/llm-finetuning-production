"""
Model loading utilities with quantization support for memory-efficient finetuning.

This module provides functions to load pretrained models with various quantization
configurations optimized for 8GB GPU memory.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from typing import Optional, Union, Tuple
import warnings


def load_model_with_quantization(
    model_name: str,
    task_type: str = "causal_lm",
    quantization: Optional[str] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    **kwargs
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a pretrained model with optional quantization.

    Args:
        model_name: HuggingFace model name or path
        task_type: Type of task (causal_lm, sequence_classification, question_answering)
        quantization: Quantization type (None, "8bit", "4bit", "4bit_double")
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for model loading

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"Task type: {task_type}")
    print(f"Quantization: {quantization}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    # Configure quantization
    quantization_config = None
    if quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        print("Using 8-bit quantization")

    elif quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )
        print("Using 4-bit NF4 quantization")

    elif quantization == "4bit_double":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True  # Extra memory savings
        )
        print("Using 4-bit NF4 quantization with double quantization")

    # Select model class based on task type
    model_class_map = {
        "causal_lm": AutoModelForCausalLM,
        "sequence_classification": AutoModelForSequenceClassification,
        "question_answering": AutoModelForQuestionAnswering
    }

    model_class = model_class_map.get(task_type, AutoModelForCausalLM)

    # Load model
    model = model_class.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **kwargs
    )

    # Print model info
    print(f"Model loaded successfully!")
    print(f"  Total parameters: {get_model_parameters(model):,}")
    print(f"  Trainable parameters: {get_trainable_parameters(model):,}")

    return model, tokenizer


def get_model_parameters(model: torch.nn.Module) -> int:
    """
    Get total number of model parameters.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Get number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_parameters(model: torch.nn.Module):
    """
    Print detailed information about trainable parameters.

    Args:
        model: PyTorch model
    """
    trainable_params = get_trainable_parameters(model)
    total_params = get_model_parameters(model)
    trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0

    print(f"Trainable Parameters:")
    print(f"  Trainable: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"  Total:     {total_params:,}")
    print(f"  Non-trainable: {total_params - trainable_params:,}")


def apply_lora_adapters(
    model: torch.nn.Module,
    task_type: str = "CAUSAL_LM",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    use_quantization: bool = False
) -> torch.nn.Module:
    """
    Apply LoRA adapters to a model.

    Args:
        model: Base model to add LoRA to
        task_type: Task type (CAUSAL_LM, SEQ_CLS, QUESTION_ANS)
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: Modules to apply LoRA to (None = auto-detect)
        use_quantization: Whether model is quantized (prepares for kbit training)

    Returns:
        Model with LoRA adapters
    """
    print(f"Applying LoRA adapters...")
    print(f"  Rank (r): {r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Dropout: {lora_dropout}")

    # Prepare model for k-bit training if quantized
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training")

    # Auto-detect target modules if not specified
    if target_modules is None:
        # Common patterns for different architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Llama, Mistral style
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif hasattr(model, "transformer"):
            # GPT-2 style
            target_modules = ["c_attn", "c_proj"]
        else:
            # Default to common attention patterns
            target_modules = ["q_proj", "v_proj"]

        print(f"  Auto-detected target modules: {target_modules}")
    else:
        print(f"  Target modules: {target_modules}")

    # Create LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=task_type
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    return model


def apply_prefix_tuning(
    model: torch.nn.Module,
    task_type: str = "CAUSAL_LM",
    num_virtual_tokens: int = 20,
    prefix_projection: bool = False
) -> torch.nn.Module:
    """
    Apply Prefix Tuning to a model.

    Args:
        model: Base model
        task_type: Task type
        num_virtual_tokens: Number of virtual tokens (prefix length)
        prefix_projection: Whether to use MLP reparameterization

    Returns:
        Model with prefix tuning
    """
    print(f"Applying Prefix Tuning...")
    print(f"  Virtual tokens: {num_virtual_tokens}")
    print(f"  Prefix projection: {prefix_projection}")

    prefix_config = PrefixTuningConfig(
        task_type=task_type,
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection=prefix_projection
    )

    model = get_peft_model(model, prefix_config)
    print_trainable_parameters(model)

    return model


def merge_and_save_adapters(
    model: torch.nn.Module,
    output_path: str,
    adapter_only: bool = False
):
    """
    Merge LoRA adapters with base model and save.

    Args:
        model: PEFT model with adapters
        output_path: Path to save merged model
        adapter_only: If True, save only adapter weights (smaller)
    """
    if not hasattr(model, "merge_and_unload"):
        print("Model doesn't have adapters to merge. Saving as is...")
        model.save_pretrained(output_path)
        return

    if adapter_only:
        # Save only adapter weights (much smaller)
        model.save_pretrained(output_path)
        print(f"Adapter weights saved to: {output_path}")
    else:
        # Merge adapters into base model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        print(f"Merged model saved to: {output_path}")


def get_model_memory_footprint(model: torch.nn.Module) -> float:
    """
    Get model memory footprint in GB.

    Args:
        model: PyTorch model

    Returns:
        Memory footprint in GB
    """
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem_total = mem_params + mem_buffers

    return mem_total / 1024**3


def load_model_for_inference(
    model_path: str,
    task_type: str = "causal_lm",
    device: str = "cuda",
    quantization: Optional[str] = None
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a finetuned model for inference.

    Args:
        model_path: Path to saved model
        task_type: Type of task
        device: Device to load on
        quantization: Optional quantization for inference

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model for inference from: {model_path}")

    model, tokenizer = load_model_with_quantization(
        model_path,
        task_type=task_type,
        quantization=quantization,
        device_map=device
    )

    model.eval()
    print("Model ready for inference!")

    return model, tokenizer

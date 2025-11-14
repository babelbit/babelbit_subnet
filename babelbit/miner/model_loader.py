"""
Model loading utilities for Babelbit miner.
Handles loading and caching of Hugging Face models.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_auth_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a Hugging Face model and tokenizer.
    
    Args:
        model_id: Hugging Face model ID
        revision: Model revision/branch
        cache_dir: Cache directory for models
        device: Device to load model on
        load_in_8bit: Whether to use 8-bit quantization
        load_in_4bit: Whether to use 4-bit quantization
        use_auth_token: Hugging Face auth token for private models
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Set up cache directory
    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    
    # Get auth token from environment if not provided
    if use_auth_token is None:
        use_auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    logger.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        token=use_auth_token,
        trust_remote_code=True,
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Determine device map
    device_map = "auto" if device == "cuda" else None
    
    logger.info(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        token=use_auth_token,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Move to device if not using device_map
    if device_map is None and device == "cuda":
        model = model.to(device)
    
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    return model, tokenizer
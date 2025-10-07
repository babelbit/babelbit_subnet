from typing import Any, Optional, List
from traceback import format_exc
import os
import torch

# Import the schemas - these should be available in the chute environment
try:
    from babelbit.chute_template.schemas import BBPredictedUtterance, BBPredictOutput
except ImportError:
    # Fallback imports if the above doesn't work in the chute environment
    from schemas import BBPredictedUtterance, BBPredictOutput

# Simple in-process cache for tokenized static prompt prefixes.
_PROMPT_CACHE: dict[str, torch.Tensor] = {}

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _pick_device() -> torch.device:
    # Prefer CUDA, then MPS, fallback CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

_DEVICE = _pick_device()
_DTYPE = torch.float16 if _DEVICE.type == "cuda" else (torch.bfloat16 if _DEVICE.type == "cpu" and torch.cuda.is_available() is False else torch.float32)

def _prepare_inputs(tokenizer, prompt: str) -> torch.Tensor:
    """Tokenize prompt with caching of static system+instruction part.

    We attempt to split prompt at the final user prefix so that the large, static
    system portion can be re-used. This heuristic can be refined if needed.
    """
    try:
        # Simple cache key (full prompt) if heuristic split isn't trivial.
        if len(prompt) < 256:  # tiny prompts: just tokenize directly
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            # Validate tensor before moving to device
            if inputs.numel() == 0:
                raise ValueError("Empty tokenization result")
            return inputs.to(_DEVICE)

        cache_key = None
        # Heuristic: find last occurrence of "Continue the utterance" which is static
        marker = "Continue the utterance"
        idx = prompt.rfind(marker)
        if idx != -1:
            static_part = prompt[:idx]
            cache_key = static_part
            dynamic_part = prompt[idx:]
            if cache_key in _PROMPT_CACHE:
                static_ids = _PROMPT_CACHE[cache_key]
            else:
                static_ids = tokenizer.encode(static_part, return_tensors="pt")
                # Validate before caching
                if static_ids.numel() == 0:
                    raise ValueError("Empty static tokenization result")
                _PROMPT_CACHE[cache_key] = static_ids
                
            dyn_ids = tokenizer.encode(dynamic_part, return_tensors="pt")
            if dyn_ids.numel() == 0:
                raise ValueError("Empty dynamic tokenization result")
                
            # Concatenate (naively) – for HF tokenizers this is safe when both are 1 x N tensors
            if dyn_ids.size(1) > 1:
                full = torch.cat([static_ids, dyn_ids[:, 1:]], dim=1)
            else:
                full = static_ids
                
            # Validate final tensor
            if full.numel() == 0:
                raise ValueError("Empty concatenated tensor")
            return full.to(_DEVICE)

        # Fallback: no split
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if inputs.numel() == 0:
            raise ValueError("Empty fallback tokenization result")
        return inputs.to(_DEVICE)
        
    except Exception as e:
        print(f"Error in _prepare_inputs: {str(e)}")
        # Emergency fallback - create a simple tensor with just the EOS token
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            fallback_tensor = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
            return fallback_tensor.to(_DEVICE)
        else:
            # Last resort - return a tensor with token ID 1 (common for many models)
            fallback_tensor = torch.tensor([[1]], dtype=torch.long)
            return fallback_tensor.to(_DEVICE)

def _predict(
    model: Any | None, data: BBPredictedUtterance, model_name: str
) -> BBPredictOutput:
    try:
        if not model:
            return BBPredictOutput(
                success=False, 
                error="Model not loaded", 
                utterance=data,
                context_used="",
                model=model_name
            )

        if not data.prefix:
            return BBPredictOutput(
                success=False, 
                error="No input provided", 
                utterance=data,
                context_used="",
                model=model_name
            )

        # Extract the LLM model and tokenizer
        llm_model = model.get("model")
        tokenizer = model.get("tokenizer")
        
        if not llm_model or not tokenizer:
            return BBPredictOutput(
                success=False, 
                error="Model or tokenizer not found", 
                utterance=data,
                context_used="",
                model=model_name
            )

        print(f"Generating prediction for prefix: '{data.prefix}'")
        print(f"Using context: '{data.context}'")

        # Create a prompt similar to the guess_full_utterance function
        system_msg = (
            "You are a helpful assistant that completes the current utterance naturally and succinctly. "
            "Return only the completed utterance text without quotes or extra commentary."
        )
        
        # Build the prompt with context and prefix
        if data.context:
            user_msg = f"Context:\n{data.context}\n\nContinue the utterance that begins with:\n{data.prefix}"
        else:
            user_msg = f"Continue the utterance that begins with:\n{data.prefix}"
        
        # For this template, we'll use the model's chat template if available
        # Otherwise fall back to simple concatenation
        try:
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback for models without chat template
                prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
        except:
            # Simple fallback if chat template fails
            prompt = f"{data.prefix}"
        
        # Move model to device lazily (only first call) and set eval mode.
        if getattr(llm_model, "_bb_moved", False) is False:
            device_to_use = _DEVICE
            try:
                # First try to move without dtype conversion
                llm_model.to(device_to_use)
                llm_model.eval()
                print(f"Model moved to {device_to_use}")
            except Exception as e:
                print(f"Error moving model to device: {str(e)}")
                # If GPU fails, fall back to CPU
                device_to_use = torch.device("cpu")
                llm_model.to(device_to_use)
                llm_model.eval()
                print(f"Fell back to CPU device")
            setattr(llm_model, "_bb_moved", True)
            # Update the device for this model instance
            setattr(llm_model, "_bb_device", device_to_use)

        # Get the device the model is actually on
        model_device = getattr(llm_model, "_bb_device", _DEVICE)
        
        # Tokenize / prepare inputs (with caching heuristic)
        try:
            inputs = _prepare_inputs(tokenizer, prompt)
            # Move inputs to the same device as the model
            inputs = inputs.to(model_device)
            # Validate input tensor dimensions
            if inputs.dim() != 2 or inputs.size(0) != 1:
                raise ValueError(f"Invalid input tensor shape: {inputs.shape}")
            # Check for invalid token IDs
            vocab_size = getattr(tokenizer, 'vocab_size', 50000)  # fallback size
            if torch.any(inputs >= vocab_size) or torch.any(inputs < 0):
                raise ValueError("Input contains invalid token IDs")
        except Exception as e:
            print(f"Error preparing inputs: {str(e)}")
            # Create a safe fallback input
            fallback_text = data.prefix[:50] if data.prefix else "Hello"  # Truncate to avoid issues
            inputs = tokenizer.encode(fallback_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(model_device)

        max_new_tokens = _get_env_int("CHUTE_MAX_NEW_TOKENS", 24)
        temperature = _get_env_float("CHUTE_TEMPERATURE", 0.7)
        top_p = _get_env_float("CHUTE_TOP_P", 0.95)
        top_k = _get_env_int("CHUTE_TOP_K", 50)
        do_sample = os.getenv("CHUTE_DO_SAMPLE", "1") not in ("0", "false", "False")

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Optional early exit if prefix already ends with EOS
        if tokenizer.eos_token and data.prefix.strip().endswith(tokenizer.eos_token):
            prediction = ""  # nothing new
            # Build output directly (skip generation)
            predicted_utterance = BBPredictedUtterance(
                index=data.index,
                step=data.step,
                prefix=data.prefix,
                prediction=prediction,
                context=data.context,
                ground_truth=data.ground_truth,
                done=data.done,
            )
            return BBPredictOutput(
                success=True,
                utterance=predicted_utterance,
                context_used="",
                model=model_name,
            )

        try:
            with torch.no_grad():
                # Skip autocast if on CPU to avoid potential issues
                if model_device.type == "cuda":
                    with torch.autocast(device_type="cuda", enabled=True):
                        outputs = llm_model.generate(inputs, **gen_kwargs)
                else:
                    outputs = llm_model.generate(inputs, **gen_kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error during generation: {str(e)}")
                # Try on CPU as fallback
                inputs_cpu = inputs.cpu()
                llm_model_cpu = llm_model.cpu()
                with torch.no_grad():
                    outputs = llm_model_cpu.generate(inputs_cpu, **gen_kwargs)
                # Update model device reference
                setattr(llm_model, "_bb_device", torch.device("cpu"))
            else:
                raise e
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part (remove the input prompt)
        if generated_text.startswith(prompt):
            prediction = generated_text[len(prompt):].strip()
        else:
            # Try to find the assistant's response after the prompt
            prediction = generated_text.strip()
            
        # Clean up the prediction - remove any system/user prefixes that might leak through
        prediction = prediction.replace("System:", "").replace("User:", "").replace("Assistant:", "").strip()
        
        # If the prediction contains the original prefix, try to extract just the completion
        if data.prefix in prediction and prediction != data.prefix:
            # Find where the prefix ends and take what comes after
            prefix_pos = prediction.find(data.prefix)
            if prefix_pos != -1:
                after_prefix = prediction[prefix_pos + len(data.prefix):].strip()
                if after_prefix:
                    prediction = after_prefix
        
        # Ensure we have some prediction (avoid trivial echo)
        if not prediction or prediction.strip() == "" or prediction.strip() == data.prefix.strip():
            prediction = os.getenv("CHUTE_FALLBACK_COMPLETION", "...")
        
        # Update the utterance with the prediction
        predicted_utterance = BBPredictedUtterance(
            index=data.index,
            step=data.step,
            prefix=data.prefix,
            prediction=prediction,
            context=data.context,  # Preserve the context
            ground_truth=data.ground_truth,
            done=data.done
        )

        return BBPredictOutput(
            success=True,
            utterance=predicted_utterance,
            context_used="",  # Extend later with actual context usage info
            model=model_name,
        )
        
    except Exception as e:
        print(f"Error in predict_utterance: {str(e)}")
        print(format_exc())
        return BBPredictOutput(
            success=False, 
            error=str(e), 
            utterance=data,
            context_used="",
            model=model_name
        )

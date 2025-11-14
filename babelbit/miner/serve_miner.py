"""
Simple FastAPI-based miner server for Babelbit subnet.
Serves predictions via HTTP endpoint that validators can call directly.

Note: Run register_axon.py first to register your miner on-chain,
then run this script to serve predictions.
"""
import asyncio
import logging
import os
from pathlib import Path
from traceback import format_exc
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from babelbit.miner.model_loader import load_model_and_tokenizer
from babelbit.utils.settings import get_settings

logger = logging.getLogger(__name__)

# Simple in-process cache for tokenized static prompt prefixes
_PROMPT_CACHE: dict[str, torch.Tensor] = {}


class BBUtteranceEvaluation(BaseModel):
    """Evaluation result for utterance prediction."""
    lexical_similarity: float = 0.0
    semantic_similarity: float = 0.0
    earliness: float = 0.0
    u_step: float = 0.0


class PredictRequest(BaseModel):
    """Request schema matching chute template."""
    index: str  # UUID session identifier
    step: int
    prefix: str
    context: str = ""
    done: bool = False
    ground_truth: str | None = None
    prediction: str = ""
    evaluation: BBUtteranceEvaluation | None = None


class PredictResponse(BaseModel):
    """Simple response schema expected by validator."""
    prediction: str


class BabelbitMiner:
    """Miner that serves predictions using a Hugging Face model."""
    
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the miner with a model.
        
        Args:
            model_id: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-hf")
            revision: Model revision/branch to use
            cache_dir: Directory for model cache
            device: Device to load model on ("cuda" or "cpu")
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Determine device and dtype
        self.device = self._pick_device() if device == "cuda" else torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Model and tokenizer loaded on demand
        self._model = None
        self._tokenizer = None
        self._model_lock = asyncio.Lock()
        self._model_moved = False
        
        logger.info(f"Initialized BabelbitMiner with model: {model_id}")
        logger.info(f"Target device: {self.device}, dtype: {self.dtype}")
    
    def _pick_device(self) -> torch.device:
        """Select best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _get_env_int(self, name: str, default: int) -> int:
        """Get integer from environment variable."""
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default
    
    def _get_env_float(self, name: str, default: float) -> float:
        """Get float from environment variable."""
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default
    
    def _prepare_inputs(self, prompt: str) -> torch.Tensor:
        """Tokenize prompt with caching of static system+instruction part."""
        try:
            # Simple cache key for tiny prompts
            if len(prompt) < 256:
                inputs = self._tokenizer.encode(prompt, return_tensors="pt")
                if inputs.numel() == 0:
                    raise ValueError("Empty tokenization result")
                return inputs.to(self.device)
            
            # Heuristic: find last occurrence of static instruction
            marker = "Continue the utterance"
            idx = prompt.rfind(marker)
            if idx != -1:
                static_part = prompt[:idx]
                cache_key = static_part
                dynamic_part = prompt[idx:]
                
                if cache_key in _PROMPT_CACHE:
                    static_ids = _PROMPT_CACHE[cache_key]
                else:
                    static_ids = self._tokenizer.encode(static_part, return_tensors="pt")
                    if static_ids.numel() == 0:
                        raise ValueError("Empty static tokenization result")
                    _PROMPT_CACHE[cache_key] = static_ids
                
                dyn_ids = self._tokenizer.encode(dynamic_part, return_tensors="pt")
                if dyn_ids.numel() == 0:
                    raise ValueError("Empty dynamic tokenization result")
                
                # Concatenate tokens
                if dyn_ids.size(1) > 1:
                    full = torch.cat([static_ids, dyn_ids[:, 1:]], dim=1)
                else:
                    full = static_ids
                
                if full.numel() == 0:
                    raise ValueError("Empty concatenated tensor")
                return full.to(self.device)
            
            # Fallback: no split
            inputs = self._tokenizer.encode(prompt, return_tensors="pt")
            if inputs.numel() == 0:
                raise ValueError("Empty fallback tokenization result")
            return inputs.to(self.device)
            
        except Exception as e:
            logger.error(f"Error in _prepare_inputs: {str(e)}")
            # Emergency fallback - create a simple tensor with EOS token
            if hasattr(self._tokenizer, 'eos_token_id') and self._tokenizer.eos_token_id is not None:
                fallback_tensor = torch.tensor([[self._tokenizer.eos_token_id]], dtype=torch.long)
            else:
                fallback_tensor = torch.tensor([[1]], dtype=torch.long)
            return fallback_tensor.to(self.device)
    
    async def load(self):
        """Load model and tokenizer (called once at startup)."""
        async with self._model_lock:
            if self._model is None:
                logger.info(f"Loading model {self.model_id}...")
                self._model, self._tokenizer = await asyncio.to_thread(
                    load_model_and_tokenizer,
                    model_id=self.model_id,
                    revision=self.revision,
                    cache_dir=self.cache_dir,
                    device=self.device,
                    load_in_8bit=self.load_in_8bit,
                    load_in_4bit=self.load_in_4bit,
                )
                logger.info(f"Model loaded successfully on {self.device}")
    
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Generate prediction for the given prefix and context.
        
        Args:
            request: Prediction request with prefix and context
            
        Returns:
            PredictResponse with generated text (just the prediction string)
        """
        try:
            # Ensure model is loaded
            if self._model is None:
                await self.load()
            
            if not request.prefix:
                logger.warning("No prefix provided, returning empty prediction")
                return PredictResponse(prediction="")
            
            logger.info(f"Generating prediction for prefix: '{request.prefix}'")
            logger.info(f"Using context: '{request.context}'")
            
            # Create prompt similar to the chute template
            system_msg = (
                "You are a helpful assistant that completes the current utterance naturally and succinctly. "
                "Return only the completed utterance text without quotes or extra commentary."
            )
            
            # Build the prompt with context and prefix
            if request.context:
                user_msg = f"Context:\n{request.context}\n\nContinue the utterance that begins with:\n{request.prefix}"
            else:
                user_msg = f"Continue the utterance that begins with:\n{request.prefix}"
            
            # Use chat template if available
            try:
                if hasattr(self._tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                    prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Fallback for models without chat template
                    prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
            except Exception:
                # Simple fallback if chat template fails
                prompt = f"{request.prefix}"
            
            # Move model to device lazily (only first call)
            if not self._model_moved:
                try:
                    self._model.to(self.device)
                    self._model.eval()
                    logger.info(f"Model moved to {self.device}")
                    self._model_moved = True
                except Exception as e:
                    logger.error(f"Error moving model to device: {str(e)}")
                    # Fallback to CPU
                    self.device = torch.device("cpu")
                    self._model.to(self.device)
                    self._model.eval()
                    logger.info("Fell back to CPU device")
                    self._model_moved = True
            
            # Tokenize with caching
            try:
                inputs = await asyncio.to_thread(self._prepare_inputs, prompt)
                
                # Validate input tensor
                if inputs.dim() != 2 or inputs.size(0) != 1:
                    raise ValueError(f"Invalid input tensor shape: {inputs.shape}")
                
                vocab_size = getattr(self._tokenizer, 'vocab_size', 50000)
                if torch.any(inputs >= vocab_size) or torch.any(inputs < 0):
                    raise ValueError("Input contains invalid token IDs")
                    
            except Exception as e:
                logger.error(f"Error preparing inputs: {str(e)}")
                # Create safe fallback input
                fallback_text = request.prefix[:50] if request.prefix else "Hello"
                inputs = self._tokenizer.encode(
                    fallback_text, return_tensors="pt", max_length=512, truncation=True
                )
                inputs = inputs.to(self.device)
            
            # Get generation parameters from environment
            max_new_tokens = self._get_env_int("CHUTE_MAX_NEW_TOKENS", 24)
            temperature = self._get_env_float("CHUTE_TEMPERATURE", 0.7)
            top_p = self._get_env_float("CHUTE_TOP_P", 0.95)
            top_k = self._get_env_int("CHUTE_TOP_K", 50)
            do_sample = os.getenv("CHUTE_DO_SAMPLE", "1") not in ("0", "false", "False")
            
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            
            # Check for early exit if prefix already ends with EOS
            if self._tokenizer.eos_token and request.prefix.strip().endswith(self._tokenizer.eos_token):
                return PredictResponse(prediction="")
            
            # Generate prediction
            try:
                def generate():
                    with torch.no_grad():
                        if self.device.type == "cuda":
                            with torch.autocast(device_type="cuda", enabled=True):
                                return self._model.generate(inputs, **gen_kwargs)
                        else:
                            return self._model.generate(inputs, **gen_kwargs)
                
                outputs = await asyncio.to_thread(generate)
                
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"CUDA error during generation: {str(e)}")
                    # Try on CPU as fallback
                    inputs_cpu = inputs.cpu()
                    self._model.cpu()
                    self.device = torch.device("cpu")
                    
                    with torch.no_grad():
                        outputs = self._model.generate(inputs_cpu, **gen_kwargs)
                else:
                    raise e
            
            # Decode the generated text
            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part (remove the input prompt)
            if generated_text.startswith(prompt):
                prediction = generated_text[len(prompt):].strip()
            else:
                prediction = generated_text.strip()
            
            # Clean up the prediction
            prediction = prediction.replace("System:", "").replace("User:", "").replace("Assistant:", "").strip()
            
            # If prediction contains the original prefix, extract just the completion
            if request.prefix in prediction and prediction != request.prefix:
                prefix_pos = prediction.find(request.prefix)
                if prefix_pos != -1:
                    after_prefix = prediction[prefix_pos + len(request.prefix):].strip()
                    if after_prefix:
                        prediction = after_prefix
            
            # Ensure we have some prediction
            if not prediction or prediction.strip() == "" or prediction.strip() == request.prefix.strip():
                prediction = os.getenv("CHUTE_FALLBACK_COMPLETION", "...")
            
            # Return full utterance (prefix + prediction)
            full_prediction = request.prefix + ' ' + prediction
            
            logger.info(f"Generated: {full_prediction[:100]}...")
            
            return PredictResponse(prediction=full_prediction)
            
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.error(format_exc())
            # Return empty prediction on error
            return PredictResponse(prediction="")


# Global miner instance
miner_instance: Optional[BabelbitMiner] = None


async def startup():
    """FastAPI startup event handler."""
    global miner_instance
    
    settings = get_settings()
    
    # Get model configuration
    model_id = settings.MINER_MODEL_ID
    revision = getattr(settings, 'MINER_MODEL_REVISION', None)
    cache_dir = settings.BABELBIT_CACHE_DIR / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Quantization settings
    load_in_8bit = getattr(settings, 'MINER_LOAD_IN_8BIT', False)
    load_in_4bit = getattr(settings, 'MINER_LOAD_IN_4BIT', False)
    device = getattr(settings, 'MINER_DEVICE', 'cuda')
    
    logger.info(f"Model: {model_id}")
    logger.info(f"Revision: {revision or 'main'}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Quantization: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Create and load miner
    miner_instance = BabelbitMiner(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    
    logger.info("Loading model...")
    try:
        await miner_instance.load()
        logger.info("✅ Model loaded successfully")
        logger.info("")
    except Exception as e:
        logger.error("❌ Failed to load model!")
        logger.error(f"   Error: {e}")
        logger.error("")
        logger.error("Common fixes:")
        logger.error("  1. For gated models (Llama, etc): Set HF_TOKEN environment variable")
        logger.error("     export HF_TOKEN=your_huggingface_token")
        logger.error("  2. Check model ID is correct and you have access")
        logger.error("  3. Ensure you have enough disk space and RAM/VRAM")
        raise


# Create FastAPI app
app = FastAPI(title="Babelbit Miner", on_startup=[startup])


@app.get("/healthz")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": miner_instance.model_id if miner_instance else "not_loaded",
        "model_loaded": miner_instance is not None and miner_instance._model is not None,
    }


@app.get("/health")
async def health_alt():
    """Alternative health check endpoint."""
    return {
        "status": "healthy",
        "model": miner_instance.model_id if miner_instance else "not_loaded",
        "model_loaded": miner_instance is not None and miner_instance._model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """Prediction endpoint matching chute template."""
    if miner_instance is None:
        raise HTTPException(status_code=503, detail="Miner not initialized")
    
    return await miner_instance.predict(request)


async def main():
    """Main entry point for the miner server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    settings = get_settings()
    axon_port = settings.MINER_AXON_PORT
    
    logger.info("=" * 60)
    logger.info("Starting Babelbit Miner Server")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  Make sure you've registered your axon first:")
    logger.info("   uv run python babelbit/miner/register_axon.py")
    logger.info("")
    
    # Start FastAPI server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=axon_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    
    logger.info(f"🚀 Miner serving predictions on port {axon_port}")
    logger.info("   Press Ctrl+C to stop.")
    logger.info("")
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down miner server...")


if __name__ == "__main__":
    asyncio.run(main())
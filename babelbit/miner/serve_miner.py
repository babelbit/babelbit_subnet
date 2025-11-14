"""
Simple FastAPI-based miner server for Babelbit subnet.
Serves predictions via HTTP endpoint that validators can call directly.

Note: Run register_axon.py first to register your miner on-chain,
then run this script to serve predictions.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from babelbit.miner.model_loader import load_model_and_tokenizer
from babelbit.utils.settings import get_settings

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    """Request schema matching chute template."""
    index: str  # UUID session identifier
    step: int
    prefix: str
    context: str


class PredictResponse(BaseModel):
    """Response schema matching chute template."""
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
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Model and tokenizer loaded on demand
        self._model = None
        self._tokenizer = None
        self._model_lock = asyncio.Lock()
        
        logger.info(f"Initialized BabelbitMiner with model: {model_id}")
    
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
            PredictResponse with generated text
        """
        # Ensure model is loaded
        if self._model is None:
            await self.load()
        
        # Construct prompt from context and prefix
        prompt = f"{request.context}\n{request.prefix}"
        
        logger.debug(f"Generating prediction for step {request.step}, index {request.index}")
        
        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Generate prediction
        try:
            output = await asyncio.to_thread(
                self._model.generate,
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            
            # Decode only the new tokens (skip the prompt)
            generated_text = self._tokenizer.decode(
                output[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            logger.debug(f"Generated: {generated_text[:100]}...")
            
            return PredictResponse(prediction=generated_text)
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


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
    return {"status": "ok"}


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
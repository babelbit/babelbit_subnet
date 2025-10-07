def _health(model: Any | None, repo_name: str) -> dict[str, Any]:
    return {
        "status": "healthy",
        "model": repo_name,
        "model_loaded": model is not None,
    }


def load_model_from_huggingface_hub(model_path: str):
    """Load LLM model from local path downloaded from HuggingFace Hub."""
    try:
        # Load tokenizer and model for text generation
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        raise ValueError(f"Failed to load LLM model from {model_path}: {e}")


def _load_model(repo_name: str, revision: str):
    try:
        model_path = snapshot_download(repo_name, revision=revision)
        print(f"Downloaded model from Hf to: {model_path}")
        model = load_model_from_huggingface_hub(model_path=model_path)
        print("✅ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

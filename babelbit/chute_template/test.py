import os
from typing import Any
from importlib.util import spec_from_file_location, module_from_spec
from logging import getLogger
from random import randint
from traceback import format_exc

from uvicorn import run
from fastapi import FastAPI
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from babelbit.chute_template.schemas import (
    BBPredictedUtterance,
    BBPredictOutput,
)
from babelbit.utils.settings import get_settings
# from babelbit.utils.video_processing import download_video
# from babelbit.utils.image_processing import image_to_base64, pil_from_array
from babelbit.utils.async_clients import get_async_client

settings = get_settings()
# import scorevision.chute_template.load
chute_template_load_spec = spec_from_file_location(
    "chute_load",
    str(settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_LOAD_UTILS),
)
chute_template_load = module_from_spec(chute_template_load_spec)
chute_template_load.os = os
chute_template_load.Any = Any
chute_template_load.snapshot_download = snapshot_download
chute_template_load.AutoTokenizer = AutoTokenizer
chute_template_load.AutoModelForCausalLM = AutoModelForCausalLM
chute_template_load_spec.loader.exec_module(chute_template_load)

# import scorevision.chute_template.predict
chute_template_predict_spec = spec_from_file_location(
    "chute_predict",
    str(settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_PREDICT_UTILS),
)
chute_template_predict = module_from_spec(chute_template_predict_spec)
chute_template_predict.Any = Any
# chute_template_predict.Image = Image
chute_template_predict.randint = randint
chute_template_predict.format_exc = format_exc
chute_template_predict.torch = torch
chute_template_predict.BBPredictedUtterance = BBPredictedUtterance
chute_template_predict.BBPredictOutput = BBPredictOutput
# chute_template_predict.SVFrameResult = SVFrameResult
# chute_template_predict.SVPredictInput = SVPredictInput
# chute_template_predict.SVPredictOutput = SVPredictOutput
# chute_template_predict.SVBox = SVBox
chute_template_predict_spec.loader.exec_module(chute_template_predict)

logger = getLogger(__name__)


def deploy_mock_chute(huggingface_repo: str, huggingface_revision: str) -> None:
    chute = FastAPI(title="mock-chute")
    global model
    model = None

    @chute.on_event("startup")
    async def load_model():
        global model
        model = chute_template_load._load_model(
            repo_name=huggingface_repo,
            revision=huggingface_revision,
        )

    @chute.post("/health")
    async def health() -> dict[str, Any]:
        return chute_template_load._health(
            model=model,
            repo_name=huggingface_repo,
        )

    @chute.post("/" + settings.CHUTES_MINER_PREDICT_ENDPOINT)
    async def predict(data: BBPredictedUtterance) -> BBPredictOutput:
        return chute_template_predict._predict(
            model=model,
            data=data,
            model_name=huggingface_repo,
        )

    @chute.get("/api/tasks/next/v2")
    async def mock_challenge():
        return {
            "task_id": "0",  # utterance prediction
            "challenge_uid": "mock-challenge-001",
            "dialogues": [
                {
                    "dialogue_uid": "mock-dialogue-001",
                    "utterances": [
                        "Hello, how are you today?",
                        "I'm doing well, thank you for asking."
                    ]
                }
            ]
        }

    run(chute)


async def test_chute_health_endpoint(base_url: str) -> None:
    logger.info("🔍 Testing `/health`...")
    session = await get_async_client()
    settings = get_settings()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
    }
    url = f"{base_url}/health"
    logger.info(url)
    try:
        async with session.post(url, headers=headers, json={}) as response:
            text = await response.text()
            logger.info(f"Response: {text} ({response.status})")
            health = await response.json()
            logger.info(health)
        assert health.get("model_loaded"), "Model not loaded"
        logger.info("✅ /health passed")
    except Exception as e:
        logger.error(f"❌ /health failed: {e}")


async def get_chute_logs(instance_id: str) -> None:
    session = await get_async_client()
    settings = get_settings()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
    }
    url = f"https://api.chutes.ai/instances/{instance_id}/logs"  # ?backfill=10000"
    logger.info(url)
    try:
        async with session.get(url, headers=headers) as response:
            text = await response.text()
            logger.info(f"Response: {text} ({response.status})")
    except Exception as e:
        logger.error(f"❌ /logs failed: {e}")


async def test_chute_predict_endpoint(
    base_url: str, test_utterances: list[BBPredictedUtterance]
) -> None:
    logger.info("🔍 Testing `/predict` with utterance data...")
    session = await get_async_client()
    settings = get_settings()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.CHUTES_API_KEY.get_secret_value()}",
    }
    url = f"{base_url}/{settings.CHUTES_MINER_PREDICT_ENDPOINT}"
    logger.info(url)
    
    try:
        successful_predictions = 0
        total_predictions = len(test_utterances)
        
        for i, utterance in enumerate(test_utterances):
            logger.info(f"Testing utterance {i+1}/{total_predictions}: '{utterance.prefix}'")
            
            async with session.post(
                url,
                headers=headers,
                json=utterance.model_dump(mode="json"),
            ) as response:
                text = await response.text()
                logger.info(f"Response status: {response.status}")
                assert response.status == 200, f"Non-200 response from predict for utterance '{utterance.prefix}'"
                output = await response.json()
                # logger.info(f"Prediction output: {output}")  # Commented out to reduce noise
            
            # Validate the response structure
            assert output["success"] is True, f"Prediction failed: {output}"
            assert "utterance" in output, "Missing utterance in response"
            assert "prediction" in output["utterance"], "Missing prediction in utterance"
            
            # Check that we got a non-empty prediction
            prediction = output["utterance"]["prediction"]
            assert isinstance(prediction, str), f"Prediction should be string, got {type(prediction)}"
            assert len(prediction.strip()) > 0, f"Empty prediction for input '{utterance.prefix}'"
            
            # Verify the utterance structure is preserved
            returned_utterance = output["utterance"]
            assert returned_utterance["index"] == utterance.index, "Utterance index mismatch"
            assert returned_utterance["step"] == utterance.step, "Utterance step mismatch"
            assert returned_utterance["prefix"] == utterance.prefix, "Utterance prefix mismatch"
            
            logger.info(f"✅ Utterance {i+1} prediction: '{utterance.prefix}' → '{prediction}'")
            successful_predictions += 1
        
        logger.info(f"✅ /predict passed: {successful_predictions}/{total_predictions} predictions successful")
        
    except Exception as e:
        logger.error(f"❌ /predict failed: {e}")
        raise


# Helper function to create test utterances
def create_test_utterances() -> list[BBPredictedUtterance]:
    """Create a set of test utterances for prediction testing"""
    test_cases = [
        ("Hello", "session-1", 1),
        ("The weather today is", "session-2", 1), 
        ("Once upon a time", "session-3", 1),
        ("I think that", "session-4", 1),
        ("The quick brown fox", "session-5", 1),
    ]
    
    return [
        BBPredictedUtterance(
            index=session_id,
            step=step,
            prefix=prefix,
            prediction="",  # Will be filled by the model
            ground_truth=None,
            done=False
        )
        for prefix, session_id, step in test_cases
    ]

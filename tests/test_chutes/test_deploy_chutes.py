import os
import pytest
import asyncio
import threading
import time
import requests
from dotenv import load_dotenv
from babelbit.chute_template.test import deploy_mock_chute
from babelbit.chute_template.schemas import BBPredictedUtterance, BBPredictOutput

# Load environment variables from .env file first
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_API_KEY', '')

@pytest.fixture(scope="session")
def mock_chute_server():
    """Start mock chute server in background thread for testing"""
    print(f"HF_TOKEN set to: {os.environ.get('HF_TOKEN', 'NOT SET')}")
    
    def run_server():
        deploy_mock_chute("distilgpt2", "main")
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start up
    max_wait = 30  # seconds
    for _ in range(max_wait * 10):
        try:
            response = requests.post("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                break
        except:
            time.sleep(0.1)
    else:
        pytest.fail("Mock chute server failed to start within 30 seconds")
    
    yield "http://localhost:8000"
    
    # Server will be cleaned up when pytest exits (daemon thread)

def test_chute_health_endpoint(mock_chute_server):
    """Test that the health endpoint returns success"""
    response = requests.post(f"{mock_chute_server}/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data

def test_chute_predict_endpoint(mock_chute_server):
    """Test that the predict endpoint works with BBPredictedUtterance"""
    payload = {
        "index": "test-session-123",  # Changed from "id" to "index"
        "step": 1,
        "prefix": "Hello world",
        "prediction": "",
        "ground_truth": "",
        "done": False
    }
    
    response = requests.post(
        f"{mock_chute_server}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert "success" in data
    assert data["success"] is True
    assert "utterance" in data
    assert "prediction" in data["utterance"]
    
    # Check that we got some prediction back
    prediction = data["utterance"]["prediction"]
    assert isinstance(prediction, str)
    assert len(prediction) > 0
    
    print(f"Generated prediction: '{prediction}' for input: 'Hello world'")  # Debug output

@pytest.mark.asyncio
async def test_chute_multiple_utterances(mock_chute_server):
    """Test the chute with multiple different utterances"""
    from babelbit.chute_template.test import create_test_utterances, test_chute_predict_endpoint
    
    # Create test utterances
    test_utterances = create_test_utterances()
    
    # Test the chute with multiple utterances
    await test_chute_predict_endpoint(mock_chute_server, test_utterances)

@pytest.mark.asyncio
async def test_chute_integration():
    """Test the chute integration without starting a server"""
    # This tests the template loading without the server
    from babelbit.chute_template.test import chute_template_load
    
    # Test that the model loading works
    model = chute_template_load._load_model("distilgpt2", "main")
    assert model is not None
    assert "model" in model
    assert "tokenizer" in model

if __name__ == "__main__":
    # For backwards compatibility, still run the server if called directly
    print(f"HF_TOKEN set to: {os.environ.get('HF_TOKEN', 'NOT SET')}")
    deploy_mock_chute("distilgpt2", "main")
#!/usr/bin/env python3
"""
Test script to verify the miner API is working correctly.
Tests both health check and prediction endpoints.
"""
import asyncio
import httpx
import sys
from typing import Optional


async def test_health(base_url: str, timeout: float = 5.0) -> bool:
    """Test the /healthz endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/healthz")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
    except httpx.ConnectError:
        print(f"❌ Connection failed - is the miner server running on {base_url}?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


async def test_predict(base_url: str, timeout: float = 30.0) -> bool:
    """Test the /predict endpoint."""
    print("\n🔍 Testing prediction endpoint...")
    
    # Sample request matching the validator's expected format
    test_request = {
        "index": "test-session-123",  # UUID string
        "step": 1,
        "prefix": "Hello, my name is",
        "context": "This is a test conversation to verify the miner API is working.",
        "done": False
    }
    
    print(f"   Sending request:")
    print(f"   - prefix: '{test_request['prefix']}'")
    print(f"   - context: '{test_request['context'][:50]}...'")
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/predict",
                json=test_request
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check simple response format: {"prediction": "..."}
                prediction = data.get("prediction", "")
                
                if not prediction:
                    print(f"❌ Prediction returned empty")
                    return False
                
                print(f"✅ Prediction successful!")
                print(f"   Generated text: '{prediction[:100]}{'...' if len(prediction) > 100 else ''}'")
                print(f"   Full prediction length: {len(prediction)} characters")
                return True
            else:
                print(f"❌ Prediction failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
    except httpx.ReadTimeout:
        print(f"❌ Prediction timed out after {timeout}s")
        print(f"   The model might be loading or inference is slow")
        return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


async def main():
    """Run all API tests."""
    # Get base URL from command line or use default
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        # Default port from settings
        base_url = "http://localhost:8091"
    
    print("=" * 60)
    print("Babelbit Miner API Test")
    print("=" * 60)
    print(f"Testing miner at: {base_url}")
    print()
    
    # Run tests
    health_ok = await test_health(base_url)
    
    if not health_ok:
        print("\n❌ Health check failed - stopping tests")
        print("\nMake sure the miner server is running:")
        print("   uv run babelbit/miner/serve_miner.py")
        sys.exit(1)
    
    predict_ok = await test_predict(base_url)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Health check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Prediction:   {'✅ PASS' if predict_ok else '❌ FAIL'}")
    print()
    
    if health_ok and predict_ok:
        print("✅ All tests passed! Miner API is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

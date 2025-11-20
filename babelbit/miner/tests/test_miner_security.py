#!/usr/bin/env python3
"""
Test script to verify the miner security modes work correctly.
Tests that production mode rejects non-Bittensor requests.
"""
import asyncio
import httpx
import sys


async def test_production_mode_rejects_plain_requests():
    """
    Test that in production mode (MINER_DEV_MODE=0), 
    the miner rejects requests without Bittensor headers.
    
    Expected: HTTP 401 Unauthorized
    """
    print("=" * 60)
    print("Testing Production Mode Security")
    print("=" * 60)
    print("\n⚠️  This test assumes the miner is running with MINER_DEV_MODE=0")
    print("   (production mode - the default)")
    print()
    
    base_url = "http://localhost:8091"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test health endpoint (should work without headers)
            print("🔍 Testing /healthz endpoint...")
            health_response = await client.get(f"{base_url}/healthz")
            if health_response.status_code == 200:
                print("✅ Health check works (expected)")
            else:
                print(f"❌ Health check failed: {health_response.status_code}")
                return False
            
            # Test prediction endpoint without Bittensor headers
            print("\n🔍 Testing /predict endpoint without Bittensor headers...")
            predict_payload = {
                "index": "test-session",
                "step": 0,
                "prefix": "Hello",
                "context": "",
                "done": False,
                "prediction": ""
            }
            
            predict_response = await client.post(
                f"{base_url}/predict",
                json=predict_payload
            )
            
            if predict_response.status_code == 401:
                print("✅ Prediction rejected with 401 (expected in production mode)")
                print(f"   Detail: {predict_response.json().get('detail', 'N/A')}")
                return True
            else:
                print(f"❌ Unexpected status code: {predict_response.status_code}")
                print(f"   Expected: 401 Unauthorized")
                print(f"   Response: {predict_response.text[:200]}")
                return False
                
    except httpx.ConnectError:
        print(f"❌ Connection failed - is the miner server running on {base_url}?")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


async def test_dev_mode_accepts_plain_requests():
    """
    Test that in dev mode (MINER_DEV_MODE=1),
    the miner accepts requests without Bittensor headers.
    
    Expected: HTTP 200 OK with prediction
    """
    print("\n" + "=" * 60)
    print("Testing Dev Mode")
    print("=" * 60)
    print("\n⚠️  This test assumes the miner is running with MINER_DEV_MODE=1")
    print()
    
    base_url = "http://localhost:8091"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test prediction endpoint without Bittensor headers
            print("🔍 Testing /predict endpoint without Bittensor headers...")
            predict_payload = {
                "index": "test-session",
                "step": 0,
                "prefix": "Hello, my name is",
                "context": "",
                "done": False,
                "prediction": ""
            }
            
            predict_response = await client.post(
                f"{base_url}/predict",
                json=predict_payload
            )
            
            if predict_response.status_code == 200:
                data = predict_response.json()
                prediction = data.get("prediction", "")
                print(f"✅ Prediction accepted (expected in dev mode)")
                print(f"   Generated: {prediction[:80]}...")
                return True
            elif predict_response.status_code == 401:
                print(f"❌ Prediction rejected with 401")
                print(f"   This means the miner is in PRODUCTION mode")
                print(f"   Please restart with: MINER_DEV_MODE=1 uv run python babelbit/miner/serve_miner.py")
                return False
            else:
                print(f"❌ Unexpected status code: {predict_response.status_code}")
                print(f"   Response: {predict_response.text[:200]}")
                return False
                
    except httpx.ConnectError:
        print(f"❌ Connection failed - is the miner server running on {base_url}?")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


async def main():
    """Run the security tests based on expected mode."""
    print("\n" + "=" * 60)
    print("Babelbit Miner Security Test")
    print("=" * 60)
    print()
    print("This test will check if the miner properly enforces")
    print("Bittensor protocol headers in production mode.")
    print()
    
    # Check what mode we're testing
    print("Which mode is your miner running in?")
    print("  1) Production mode (MINER_DEV_MODE=0 or not set) - DEFAULT")
    print("  2) Dev mode (MINER_DEV_MODE=1)")
    print()
    
    choice = input("Enter 1 or 2 [1]: ").strip() or "1"
    
    if choice == "1":
        success = await test_production_mode_rejects_plain_requests()
    elif choice == "2":
        success = await test_dev_mode_accepts_plain_requests()
    else:
        print("Invalid choice")
        return 1
    
    print()
    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    if success:
        print("✅ Security test PASSED")
        return 0
    else:
        print("❌ Security test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

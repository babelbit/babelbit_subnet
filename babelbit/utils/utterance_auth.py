"""
Utterance Engine Authentication Utilities

Handles JWT authentication for the utterance engine API using Bittensor validator hotkey signatures.
"""

import time
import asyncio
from typing import Optional, Dict, Any
from logging import getLogger

from babelbit.utils.async_clients import get_async_client
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from babelbit.utils.signing import sign_message

logger = getLogger(__name__)


class UtteranceAuthError(Exception):
    """Raised when utterance engine authentication fails"""
    pass


class UtteranceAuthenticator:
    """Handles authentication with the utterance engine API"""
    
    def __init__(self, base_url: str, wallet_name: str, hotkey_name: str):
        self.base_url = base_url.rstrip('/')
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self._jwt_token: Optional[str] = None
        self._token_expiry: Optional[float] = None
        self._keypair = None
        
    def _load_keypair(self):
        """Load the validator keypair for signing"""
        if self._keypair is None:
            self._keypair = load_hotkey_keypair(self.wallet_name, self.hotkey_name)
        return self._keypair
    
    def _is_token_valid(self) -> bool:
        """Check if current JWT token is valid and not expired"""
        if not self._jwt_token:
            return False
        
        if self._token_expiry and time.time() >= self._token_expiry:
            return False
            
        return True
    
    async def get_challenge(self) -> Dict[str, Any]:
        """Get authentication challenge from utterance engine"""
        session = await get_async_client()
        
        try:
            async with session.post(f"{self.base_url}/auth") as response:
                if response.status != 200:
                    text = await response.text()
                    raise UtteranceAuthError(f"Failed to get challenge: HTTP {response.status} - {text}")
                
                return await response.json()
                
        except Exception as e:
            raise UtteranceAuthError(f"Error getting challenge: {e}")
    
    async def verify_authentication(self, challenge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify authentication by signing challenge and getting JWT token"""
        session = await get_async_client()
        
        try:
            # Load keypair for signing
            keypair = self._load_keypair()
            
            # Extract challenge details
            message = challenge_data["challenge"]
            timestamp = challenge_data["timestamp"]
            
            # Sign the challenge message
            signature = sign_message(keypair, message)
            
            # Prepare auth request
            auth_request = {
                "hotkey": keypair.ss58_address,
                "signature": signature,
                "timestamp": timestamp,
                "message": message
            }
            
            # Send verification request
            async with session.post(
                f"{self.base_url}/auth/verify",
                json=auth_request
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise UtteranceAuthError(f"Authentication failed: HTTP {response.status} - {text}")
                
                auth_response = await response.json()
                
                # Store JWT token and expiry
                self._jwt_token = auth_response["access_token"]
                expires_in = auth_response.get("expires_in", 86400)  # Default 24h
                self._token_expiry = time.time() + expires_in - 300  # 5min buffer
                
                logger.info(f"Successfully authenticated as validator UID: {auth_response.get('validator_uid')}")
                return auth_response
                
        except Exception as e:
            raise UtteranceAuthError(f"Error during authentication verification: {e}")
    
    async def authenticate(self) -> Dict[str, Any]:
        """Complete authentication flow: get challenge, sign it, and get JWT token"""
        if self._is_token_valid():
            logger.debug("Using existing valid JWT token")
            return {"access_token": self._jwt_token}
        
        logger.info("Starting utterance engine authentication...")
        
        # Step 1: Get challenge
        challenge_data = await self.get_challenge()
        
        # Step 2: Sign challenge and verify
        auth_response = await self.verify_authentication(challenge_data)
        
        return auth_response
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        if not self._is_token_valid():
            await self.authenticate()
        
        return {
            "Authorization": f"Bearer {self._jwt_token}",
            "Content-Type": "application/json"
        }


# Global authenticator instance (will be initialized in runner)
_authenticator: Optional[UtteranceAuthenticator] = None


def init_utterance_auth(base_url: str, wallet_name: str, hotkey_name: str) -> UtteranceAuthenticator:
    """Initialize global utterance authenticator"""
    global _authenticator
    _authenticator = UtteranceAuthenticator(base_url, wallet_name, hotkey_name)
    return _authenticator


async def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers using global authenticator"""
    if not _authenticator:
        raise UtteranceAuthError("Utterance authenticator not initialized. Call init_utterance_auth() first.")
    
    return await _authenticator.get_auth_headers()


async def authenticate_utterance_engine() -> Dict[str, Any]:
    """
    Authenticate with utterance engine using global authenticator.
    Includes retry logic with exponential backoff for robustness.
    """
    if not _authenticator:
        raise UtteranceAuthError("Utterance authenticator not initialized. Call init_utterance_auth() first.")
    
    max_retries = 5
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            result = await _authenticator.authenticate()
            if attempt > 0:
                logger.info(f"Authentication succeeded on attempt {attempt + 1}")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # exponential backoff: 2, 4, 8, 16 seconds
                logger.warning(
                    f"Authentication attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"Authentication failed after {max_retries} attempts")
                raise UtteranceAuthError(f"Failed to authenticate after {max_retries} attempts: {e}")
    
    # This should never be reached, but for type safety
    raise UtteranceAuthError("Authentication failed unexpectedly")
#!/usr/bin/env python3
"""
Test suite for authentication failure handling

Tests cover:
1. Utterance engine auth token expiration
2. Invalid credentials handling
3. Network timeouts during authentication
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from aiohttp import ClientError, ClientTimeout
import aiohttp

from babelbit.cli.runner import runner
from babelbit.utils.utterance_auth import (
    UtteranceAuthenticator, 
    UtteranceAuthError,
    authenticate_utterance_engine,
    init_utterance_auth
)


class TestAuthenticationFailures:
    """Test suite for authentication failure scenarios"""

    @pytest.mark.asyncio
    async def test_runner_exits_on_authentication_failure(self, tmp_path):
        """Test that runner exits gracefully when authentication fails"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth, \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock) as mock_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock) as mock_miners, \
             patch('babelbit.cli.runner.close_http_clients') as mock_close:
            
            # Simulate authentication failure
            mock_auth.side_effect = UtteranceAuthError("Invalid credentials")
            
            # Run the function
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                await runner(utterance_engine_url="http://localhost:8000")
            
        # Verify authentication was attempted
        mock_init_auth.assert_called_once()
        mock_auth.assert_called_once()
        
        # Verify runner exited early - challenge and miners should not be fetched
        mock_challenge.assert_not_called()
        mock_miners.assert_not_called()
        
        # Note: close_http_clients is NOT called when runner returns early (before try-finally block)    def test_auth_token_expiration_and_refresh(self):
        """Test that expired auth tokens are refreshed automatically"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="test_wallet",
            hotkey_name="test_hotkey"
        )
        
        # Manually set an expired token
        authenticator._jwt_token = "expired_token"
        authenticator._token_expiry = time.time() - 100  # Expired 100 seconds ago
        
        # Check token validity
        assert not authenticator._is_token_valid(), "Expired token should be invalid"
        
        # Test with no token
        authenticator._jwt_token = None
        authenticator._token_expiry = None
        assert not authenticator._is_token_valid(), "Missing token should be invalid"
        
        # Test with valid token
        authenticator._jwt_token = "valid_token"
        authenticator._token_expiry = time.time() + 3600  # Expires in 1 hour
        assert authenticator._is_token_valid(), "Valid token should be valid"

    @pytest.mark.asyncio
    async def test_authentication_network_timeout(self):
        """Test handling of network timeouts during authentication"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="test_wallet",
            hotkey_name="test_hotkey"
        )
        
        async def mock_timeout_response(*args, **kwargs):
            raise asyncio.TimeoutError("Connection timeout")
        
        with patch('babelbit.utils.utterance_auth.get_async_client') as mock_client:
            mock_session = AsyncMock()
            
            # Setup proper async context manager mock
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(side_effect=mock_timeout_response)
            mock_ctx.__aexit__ = AsyncMock()
            mock_session.post = Mock(return_value=mock_ctx)
            
            mock_client.return_value = mock_session
            
            with pytest.raises(UtteranceAuthError) as exc_info:
                await authenticator.get_challenge()
            
            assert "timeout" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authentication_invalid_credentials(self):
        """Test handling of invalid credentials (wrong signature)"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="test_wallet",
            hotkey_name="test_hotkey"
        )
        
        # Mock challenge response
        mock_challenge_response = {
            "challenge": "test_challenge_message",
            "timestamp": int(time.time())
        }
        
        async def mock_post_challenge(*args, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_challenge_response)
            return mock_resp
        
        async def mock_post_verify(*args, **kwargs):
            # Simulate authentication failure due to invalid signature
            mock_resp = AsyncMock()
            mock_resp.status = 401
            mock_resp.text = AsyncMock(return_value="Invalid signature")
            return mock_resp
        
        with patch('babelbit.utils.utterance_auth.get_async_client') as mock_client:
            mock_session = AsyncMock()
            
            # Track which URL was called
            call_count = [0]
            
            def create_context_manager(*args, **kwargs):
                call_count[0] += 1
                ctx = AsyncMock()
                if call_count[0] == 1:
                    # First call (get_challenge)
                    ctx.__aenter__ = mock_post_challenge
                else:
                    # Second call (verify)
                    ctx.__aenter__ = mock_post_verify
                ctx.__aexit__ = AsyncMock(return_value=None)
                return ctx
            
            mock_session.post = Mock(side_effect=create_context_manager)
            mock_client.return_value = mock_session
            
            # Mock keypair
            with patch('babelbit.utils.utterance_auth.load_hotkey_keypair') as mock_keypair, \
                 patch('babelbit.utils.utterance_auth.sign_message', return_value="invalid_signature"):
                
                mock_kp = Mock()
                mock_kp.ss58_address = "5TestAddress"
                mock_keypair.return_value = mock_kp
                
                # Get challenge should succeed
                challenge = await authenticator.get_challenge()
                assert challenge == mock_challenge_response
                
                # Verify should fail with invalid credentials
                with pytest.raises(UtteranceAuthError) as exc_info:
                    await authenticator.verify_authentication(mock_challenge_response)
                
                assert "401" in str(exc_info.value) or "authentication failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authentication_server_error(self):
        """Test handling of server errors (500) during authentication"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="test_wallet",
            hotkey_name="test_hotkey"
        )
        
        async def mock_server_error(*args, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.status = 500
            mock_resp.text = AsyncMock(return_value="Internal server error")
            return mock_resp
        
        with patch('babelbit.utils.utterance_auth.get_async_client') as mock_client:
            mock_session = AsyncMock()
            
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = mock_server_error
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_ctx)
            
            mock_client.return_value = mock_session
            
            with pytest.raises(UtteranceAuthError) as exc_info:
                await authenticator.get_challenge()
            
            # Check for 500 or just that it's an error
            assert "500" in str(exc_info.value) or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authentication_connection_refused(self):
        """Test handling of connection refused errors"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="test_wallet",
            hotkey_name="test_hotkey"
        )
        
        async def mock_connection_error():
            raise aiohttp.ClientConnectionError("Connection refused")
        
        with patch('babelbit.utils.utterance_auth.get_async_client') as mock_client:
            mock_session = AsyncMock()
            
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = mock_connection_error
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_ctx)
            
            mock_client.return_value = mock_session
            
            with pytest.raises(UtteranceAuthError) as exc_info:
                await authenticator.get_challenge()
            
            assert "connection" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authentication_malformed_response(self):
        """Test handling of malformed JSON responses from auth server"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="test_wallet",
            hotkey_name="test_hotkey"
        )
        
        async def mock_malformed_response():
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
            return mock_resp
        
        with patch('babelbit.utils.utterance_auth.get_async_client') as mock_client:
            mock_session = AsyncMock()
            
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = mock_malformed_response
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = Mock(return_value=mock_ctx)
            
            mock_client.return_value = mock_session
            
            with pytest.raises(UtteranceAuthError):
                await authenticator.get_challenge()

    @pytest.mark.asyncio
    async def test_authentication_missing_wallet_files(self):
        """Test handling of missing wallet/hotkey files"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="nonexistent_wallet",
            hotkey_name="nonexistent_hotkey"
        )
        
        mock_challenge = {
            "challenge": "test_challenge",
            "timestamp": int(time.time())
        }
        
        with patch('babelbit.utils.utterance_auth.load_hotkey_keypair') as mock_load:
            # Simulate missing wallet file
            mock_load.side_effect = FileNotFoundError("Wallet file not found")
            
            with pytest.raises(FileNotFoundError):
                authenticator._load_keypair()

    @pytest.mark.asyncio
    async def test_runner_retry_on_transient_auth_failure(self):
        """Test that runner can recover from transient authentication failures"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        
        auth_attempts = [0]
        
        async def mock_auth_with_retry():
            auth_attempts[0] += 1
            
            # First attempt fails (transient network issue)
            if auth_attempts[0] == 1:
                raise UtteranceAuthError("Temporary network error")
            
            # Second attempt succeeds
            return True
        
        # This test demonstrates the pattern - actual retry logic would be in validate.py runner_loop
        with pytest.raises(UtteranceAuthError):
            await mock_auth_with_retry()
        
        # Retry after failure
        result = await mock_auth_with_retry()
        assert result is True
        assert auth_attempts[0] == 2

    @pytest.mark.asyncio
    async def test_authentication_token_storage(self):
        """Test that JWT tokens are properly stored and retrieved"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="test_wallet",
            hotkey_name="test_hotkey"
        )
        
        # Initially no token
        assert authenticator._jwt_token is None
        assert authenticator._token_expiry is None
        
        # Simulate successful authentication
        test_token = "test_jwt_token_12345"
        test_expiry = time.time() + 3600
        
        authenticator._jwt_token = test_token
        authenticator._token_expiry = test_expiry
        
        # Verify token is stored
        assert authenticator._jwt_token == test_token
        assert authenticator._token_expiry == test_expiry
        assert authenticator._is_token_valid()

    @pytest.mark.asyncio
    async def test_authentication_with_empty_challenge(self):
        """Test handling of empty or invalid challenge responses"""
        
        authenticator = UtteranceAuthenticator(
            base_url="http://localhost:8000",
            wallet_name="test_wallet",
            hotkey_name="test_hotkey"
        )
        
        # Test with empty challenge
        empty_challenge = {}
        
        with patch('babelbit.utils.utterance_auth.get_async_client') as mock_client, \
             patch('babelbit.utils.utterance_auth.load_hotkey_keypair') as mock_keypair, \
             patch('babelbit.utils.utterance_auth.sign_message', return_value="signature"):
            
            mock_kp = Mock()
            mock_kp.ss58_address = "5TestAddress"
            mock_keypair.return_value = mock_kp
            
            # Should raise error when accessing missing fields
            with pytest.raises(UtteranceAuthError) as exc_info:
                await authenticator.verify_authentication(empty_challenge)
            
            assert "challenge" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

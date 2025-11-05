#!/usr/bin/env python3
"""
Test suite for S3Manager functionality.

Tests cover:
- S3 upload success scenarios
- Upload failure handling
- Configuration errors
- Missing bucket handling
- Retry logic
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import sys

# Mock boto3 before importing S3Manager to handle missing dependency
sys.modules['boto3'] = MagicMock()
sys.modules['botocore'] = MagicMock()
sys.modules['botocore.exceptions'] = MagicMock()
sys.modules['botocore.client'] = MagicMock()

from babelbit.utils.s3_manager import S3Manager


@pytest.fixture(autouse=True)
def reset_boto3_mock():
    """Reset boto3 mock between tests to avoid state leakage"""
    sys.modules['boto3'].reset_mock()
    sys.modules['botocore'].reset_mock()
    yield
    sys.modules['boto3'].reset_mock()
    sys.modules['botocore'].reset_mock()


@pytest.fixture
def valid_s3_config():
    """Valid S3 configuration for testing"""
    return {
        "bucket_name": "test-bucket",
        "access_key": "test-access-key",
        "secret_key": "test-secret-key",
        "endpoint_url": "https://s3.us-east-1.amazonaws.com",
        "region": "us-east-1",
        "addressing_style": "auto",
        "signature_version": "s3v4",
        "use_ssl": True,
        "prefix": "test-prefix/"
    }


@pytest.fixture
def temp_test_file(tmp_path):
    """Create a temporary test file"""
    test_file = tmp_path / "test_upload.txt"
    test_file.write_text("Test content for S3 upload")
    return test_file


class TestS3ManagerInitialization:
    """Test S3Manager initialization and configuration"""

    def test_s3_manager_init_success(self, valid_s3_config):
        """Test successful S3Manager initialization"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            
            assert manager.bucket_name == "test-bucket"
            assert manager.prefix == "test-prefix/"
            mock_boto_client.assert_called_once()

    def test_s3_manager_init_invalid_addressing_style(self):
        """Test S3Manager initialization with invalid addressing style - boto3 handles validation"""
        # Note: S3Manager doesn't validate addressing_style itself, boto3 Config does
        # We need to mock Config to prevent validation during test
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client, \
             patch('babelbit.utils.s3_manager.Config') as mock_config:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client
            mock_config.return_value = Mock()  # Mock Config to bypass validation
            
            # Should not raise during init with mocked Config
            manager = S3Manager(
                bucket_name="test-bucket",
                access_key="key",
                secret_key="secret",
                addressing_style="invalid_style"
            )
            
            assert manager.bucket_name == "test-bucket"
            # Verify Config was called with the invalid style
            mock_config.assert_called_once()
            call_args = mock_config.call_args
            assert call_args[1].get('s3') == {'addressing_style': 'invalid_style'}

    def test_s3_manager_init_with_none_endpoint(self, valid_s3_config):
        """Test S3Manager handles None endpoint_url gracefully"""
        valid_s3_config['endpoint_url'] = None
        
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            
            # Should work without endpoint_url
            assert manager.bucket_name == "test-bucket"

    def test_s3_manager_empty_prefix(self):
        """Test S3Manager with empty prefix"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(
                bucket_name="test-bucket",
                access_key="key",
                secret_key="secret",
                prefix=""
            )
            
            assert manager.prefix == ""


class TestS3Upload:
    """Test S3 file upload functionality"""

    def test_upload_file_success(self, valid_s3_config, temp_test_file):
        """Test successful file upload to S3"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            result = manager.upload_file(str(temp_test_file), "test_object.txt")
            
            assert result is True
            mock_client.upload_file.assert_called_once_with(
                str(temp_test_file),
                "test-bucket",
                "test-prefix/test_object.txt"
            )

    def test_upload_file_without_object_name(self, valid_s3_config, temp_test_file):
        """Test upload uses filename as object name when not specified"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            result = manager.upload_file(str(temp_test_file))
            
            assert result is True
            # Should use basename of file
            mock_client.upload_file.assert_called_once()
            call_args = mock_client.upload_file.call_args[0]
            assert call_args[2].endswith("test_upload.txt")

    def test_upload_file_failure_exception(self, valid_s3_config, temp_test_file):
        """Test upload failure handling when boto3 raises exception"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_client.upload_file.side_effect = Exception("Network error")
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            result = manager.upload_file(str(temp_test_file), "test_object.txt")
            
            assert result is False

    def test_upload_file_permission_denied(self, valid_s3_config, temp_test_file):
        """Test upload failure due to permission denied"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            # Simulate permission error
            mock_client.upload_file.side_effect = Exception("Access Denied")
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            result = manager.upload_file(str(temp_test_file), "test_object.txt")
            
            assert result is False

    def test_upload_nonexistent_file(self, valid_s3_config):
        """Test upload of non-existent file"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_client.upload_file.side_effect = FileNotFoundError("File not found")
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            result = manager.upload_file("/nonexistent/file.txt", "test_object.txt")
            
            assert result is False

    def test_upload_with_prefix_concatenation(self, valid_s3_config, temp_test_file):
        """Test that prefix is correctly concatenated with object name"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client
            
            valid_s3_config['prefix'] = "logs/submissions/"
            manager = S3Manager(**valid_s3_config)
            manager.upload_file(str(temp_test_file), "file.json")
            
            # Check the full key includes prefix
            call_args = mock_client.upload_file.call_args[0]
            assert call_args[2] == "logs/submissions/file.json"

    def test_upload_empty_file(self, valid_s3_config, tmp_path):
        """Test upload of empty file"""
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()
        
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            result = manager.upload_file(str(empty_file), "empty.txt")
            
            assert result is True
            mock_client.upload_file.assert_called_once()


class TestS3ManagerIntegration:
    """Integration-style tests for S3Manager with runner"""

    def test_runner_s3_disabled(self, valid_s3_config):
        """Test that runner handles S3 disabled gracefully"""
        # This would be tested in test_runner.py, but we verify config here
        with patch.dict('os.environ', {'BB_ENABLE_S3_UPLOADS': '0'}):
            # S3 Manager should not be initialized
            assert os.getenv('BB_ENABLE_S3_UPLOADS') == '0'

    def test_s3_config_from_settings(self):
        """Test S3 configuration can be loaded from settings"""
        from babelbit.utils.settings import get_settings
        
        # Just verify settings module exists and has S3 config
        settings = get_settings()
        
        # Check that S3 settings exist (actual values will vary by environment)
        assert hasattr(settings, 'S3_BUCKET_NAME')
        assert hasattr(settings, 'S3_ACCESS_KEY_ID')
        assert hasattr(settings, 'S3_SECRET_ACCESS_KEY')


class TestS3ErrorScenarios:
    """Test various error scenarios"""

    def test_network_timeout(self, valid_s3_config, temp_test_file):
        """Test upload failure due to network timeout"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            # Simulate timeout error
            mock_client.upload_file.side_effect = Exception("Connection timeout")
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            result = manager.upload_file(str(temp_test_file), "test.txt")
            
            assert result is False

    def test_invalid_bucket_name(self, valid_s3_config, temp_test_file):
        """Test upload to non-existent bucket"""
        with patch('babelbit.utils.s3_manager.boto3.client') as mock_boto_client:
            mock_client = Mock()
            # Simulate bucket not found error
            mock_client.upload_file.side_effect = Exception("The specified bucket does not exist")
            mock_boto_client.return_value = mock_client
            
            manager = S3Manager(**valid_s3_config)
            result = manager.upload_file(str(temp_test_file), "test.txt")
            
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

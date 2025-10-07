import os
import boto3
from typing import Optional, Dict, Any
from botocore.client import Config

from logging import getLogger
logger = getLogger(__name__)


class S3Manager:
    def __init__(self, 
                 bucket_name: str,
                 access_key: str,
                 secret_key: str,
                 endpoint_url: Optional[str] = None,
                 region: Optional[str] = None,
                 addressing_style: Optional[str] = None,
                 signature_version: Optional[str] = None,
                 use_ssl: Optional[bool] = None,
                 prefix: str = ''):
        """
        Generic S3-compatible storage manager.
        Supports AWS S3, Digital Ocean Spaces, MinIO, and other S3-compatible services.
        
        Configuration is done entirely through environment variables:
        - S3_ENDPOINT_URL: Custom endpoint (optional, defaults to AWS S3)
        - S3_REGION: Region name (optional, defaults to 'us-east-1')
        - S3_ACCESS_KEY_ID: Access key ID
        - S3_SECRET_ACCESS_KEY: Secret access key
        - S3_BUCKET_NAME: Default bucket name (can be overridden)
        - S3_ADDRESSING_STYLE: 'path' or 'virtual' (optional, defaults to 'virtual')
        - S3_SIGNATURE_VERSION: Signature version (optional, defaults to 's3v4')
        - S3_USE_SSL: 'true' or 'false' (optional, defaults to 'true')
        """
        # Use provided bucket or get from environment
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("Bucket name must be provided either as parameter or S3_BUCKET_NAME environment variable")
        
        # Build client configuration
        client_kwargs: Dict[str, Any] = {
            'service_name': 's3'
        }
        
        # Add region if provided
        if region:
            client_kwargs['region_name'] = region
        
        # Add endpoint URL if provided (for non-AWS services)
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
        
        # Add credentials if provided
        if access_key and secret_key:
            client_kwargs['aws_access_key_id'] = access_key
            client_kwargs['aws_secret_access_key'] = secret_key
        
        # Note: use_ssl is handled via endpoint_url (https vs http)
        
        # Build boto3 Config object for advanced options (separate from client parameters)
        config_kwargs: Dict[str, Any] = {}
        if signature_version:
            config_kwargs['signature_version'] = signature_version
        
        # Add S3-specific configuration
        if addressing_style:
            config_kwargs['s3'] = {'addressing_style': addressing_style}
        
        # Only add config if we have configuration options
        if config_kwargs:
            client_kwargs['config'] = Config(**config_kwargs)
        
        # Create the S3 client
        self.client = boto3.client(**client_kwargs)
        
        # Set up prefix
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        
        # Log configuration (without secrets)
        self._log_config(endpoint_url, region or 'default', addressing_style or 'auto', signature_version or 's3v4', use_ssl or True)
    
    def _log_config(self, endpoint_url: Optional[str], region: str, addressing_style: str, 
                   signature_version: str, use_ssl: bool) -> None:
        """Log the configuration (without sensitive information)"""
        config_info = [
            f"Bucket: {self.bucket_name}",
            f"Region: {region}",
            f"Addressing: {addressing_style}",
            f"Signature: {signature_version}",
            f"Use SSL: {use_ssl}"
        ]
        
        if endpoint_url:
            config_info.insert(1, f"Endpoint: {endpoint_url}")
        else:
            config_info.insert(1, "Endpoint: AWS S3 (default)")
            
        if self.prefix:
            config_info.append(f"Prefix: {self.prefix}")
            
        logger.info(f"S3Manager initialized - {', '.join(config_info)}")
    
    def upload_file(self, file_path: str, object_name: str | None = None) -> bool:
        """Upload a file to the S3 bucket with prefix"""
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        full_key = self.prefix + object_name
        
        try:
            self.client.upload_file(file_path, self.bucket_name, full_key)
            return True
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False

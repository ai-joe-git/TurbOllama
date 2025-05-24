"""
Model Download Utilities
"""

import os
import requests
import hashlib
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from urllib.parse import urlparse
import time

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Advanced model downloading with progress tracking and resume support."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def download_from_url(
        self, 
        url: str, 
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 8192
    ) -> str:
        """Download file from URL with progress tracking and resume support."""
        
        if not filename:
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            
        if not filename.endswith('.gguf'):
            raise ValueError("Only GGUF files are supported")
            
        file_path = self.cache_dir / filename
        temp_path = file_path.with_suffix('.tmp')
        
        # Check if file already exists
        if file_path.exists():
            logger.info(f"File already exists: {filename}")
            return str(file_path)
        
        # Get file info
        headers = {}
        if temp_path.exists():
            # Resume download
            headers['Range'] = f'bytes={temp_path.stat().st_size}-'
            mode = 'ab'
            downloaded = temp_path.stat().st_size
            logger.info(f"Resuming download from byte {downloaded}")
        else:
            mode = 'wb'
            downloaded = 0
        
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get total size
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])
            elif 'content-length' in response.headers:
                total_size = int(response.headers['content-length']) + downloaded
            else:
                total_size = 0
            
            logger.info(f"Downloading {filename}: {total_size / (1024**3):.1f} GB")
            
            # Download with progress
            with open(temp_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=downloaded,
                    unit='B',
                    unit_scale=True,
                    desc=filename
                ) as pbar:
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            
                            if progress_callback:
                                progress_callback(downloaded, total_size)
            
            # Move to final location
            temp_path.rename(file_path)
            logger.info(f"Download completed: {filename}")
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if temp_path.exists() and temp_path.stat().st_size == 0:
                temp_path.unlink()  # Remove empty temp file
            raise
    
    async def download_from_huggingface(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """Download model from HuggingFace with automatic file selection."""
        
        try:
            # List available files
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                raise ValueError(f"No GGUF files found in repository {repo_id}")
            
            # Select file
            if filename:
                if filename not in gguf_files:
                    raise ValueError(f"File {filename} not found in repository")
                target_file = filename
            else:
                target_file = self._select_best_gguf_file(gguf_files)
            
            logger.info(f"Downloading {target_file} from {repo_id}")
            
            # Download using HuggingFace hub
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=target_file,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                local_files_only=False
            )
            
            logger.info(f"Downloaded to: {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            raise
    
    def _select_best_gguf_file(self, gguf_files: list) -> str:
        """Select the best GGUF file from available options."""
        # Preference order for quantizations
        preference_order = [
            'Q4_K_M', 'Q4_K_S', 'Q5_K_M', 'Q5_K_S', 'Q6_K',
            'Q4_0', 'Q5_0', 'Q8_0', 'F16'
        ]
        
        for preferred_quant in preference_order:
            for file in gguf_files:
                if preferred_quant.lower() in file.lower():
                    return file
        
        # If no preferred quantization found, return first file
        return gguf_files[0]
    
    def verify_checksum(self, file_path: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify file checksum."""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        actual_hash = hash_func.hexdigest()
        return actual_hash.lower() == expected_hash.lower()
    
    def get_download_info(self, url: str) -> Dict[str, Any]:
        """Get download information without downloading."""
        try:
            response = requests.head(url, timeout=10)
            response.raise_for_status()
            
            return {
                'url': url,
                'size': int(response.headers.get('content-length', 0)),
                'content_type': response.headers.get('content-type', ''),
                'last_modified': response.headers.get('last-modified', ''),
                'supports_resume': 'accept-ranges' in response.headers
            }
        except Exception as e:
            logger.error(f"Failed to get download info: {e}")
            return {}

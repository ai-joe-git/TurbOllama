"""
Model Management for TurboLlama
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from .config import Config
from .gpu_manager import GPUManager
from .llama_wrapper import LlamaWrapper

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information structure."""
    name: str
    path: str
    size: int  # bytes
    format: str  # GGUF, GGML, etc.
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    capabilities: List[str] = None  # text, vision, tools
    downloaded_at: Optional[float] = None
    last_used: Optional[float] = None


class ModelManager:
    """Manages model downloading, caching, and loading."""
    
    def __init__(self, config: Config, gpu_manager: GPUManager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.models_dir = Path(config.models.cache_dir).expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_model: Optional[LlamaWrapper] = None
        self.current_model_name: Optional[str] = None
        
        # Model registry
        self.model_registry = self._load_model_registry()
        
        # HuggingFace API
        self.hf_api = HfApi()
    
    def _load_model_registry(self) -> Dict[str, ModelInfo]:
        """Load model registry from disk."""
        registry_path = self.models_dir / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    return {
                        name: ModelInfo(**info) 
                        for name, info in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")
        
        return {}
    
    def _save_model_registry(self) -> None:
        """Save model registry to disk."""
        registry_path = self.models_dir / "registry.json"
        try:
            data = {
                name: {
                    'name': info.name,
                    'path': info.path,
                    'size': info.size,
                    'format': info.format,
                    'quantization': info.quantization,
                    'context_length': info.context_length,
                    'capabilities': info.capabilities or [],
                    'downloaded_at': info.downloaded_at,
                    'last_used': info.last_used,
                }
                for name, info in self.model_registry.items()
            }
            
            with open(registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def list_models(self) -> List[str]:
        """List all available models."""
        # Refresh registry by scanning directory
        self._scan_models_directory()
        return list(self.model_registry.keys())
    
    def _scan_models_directory(self) -> None:
        """Scan models directory and update registry."""
        for file_path in self.models_dir.rglob("*.gguf"):
            relative_path = file_path.relative_to(self.models_dir)
            model_name = str(relative_path).replace('/', ':').replace('.gguf', '')
            
            if model_name not in self.model_registry:
                # Add new model to registry
                stat = file_path.stat()
                self.model_registry[model_name] = ModelInfo(
                    name=model_name,
                    path=str(file_path),
                    size=stat.st_size,
                    format="GGUF",
                    quantization=self._detect_quantization(file_path.name),
                    capabilities=self._detect_capabilities(file_path),
                    downloaded_at=stat.st_mtime,
                )
        
        self._save_model_registry()
    
    def _detect_quantization(self, filename: str) -> Optional[str]:
        """Detect quantization type from filename."""
        quantizations = [
            'Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'Q4_0', 'Q4_1', 
            'Q4_K_S', 'Q4_K_M', 'Q5_0', 'Q5_1', 'Q5_K_S', 'Q5_K_M', 
            'Q6_K', 'Q8_0', 'F16', 'F32'
        ]
        
        filename_upper = filename.upper()
        for quant in quantizations:
            if quant in filename_upper:
                return quant
        
        return None
    
    def _detect_capabilities(self, model_path: Path) -> List[str]:
        """Detect model capabilities (text, vision, tools)."""
        capabilities = ["text"]  # All models support text
        
        # Check for vision capabilities
        model_name = model_path.name.lower()
        if any(vision_keyword in model_name for vision_keyword in [
            'llava', 'vision', 'multimodal', 'clip', 'blip'
        ]):
            capabilities.append("vision")
        
        # Check for tool/function calling capabilities
        if any(tool_keyword in model_name for tool_keyword in [
            'function', 'tool', 'agent', 'react'
        ]):
            capabilities.append("tools")
        
        return capabilities
    
    async def pull_model(self, model_identifier: str, specific_file: Optional[str] = None) -> str:
        """
        Download a model from various sources.
        
        Args:
            model_identifier: Can be:
                - HuggingFace repo (e.g., "microsoft/DialoGPT-medium")
                - Ollama-style name (e.g., "llama2:7b")
                - Direct URL
            specific_file: Specific file to download from HF repo
        
        Returns:
            Local model name for loading
        """
        logger.info(f"Pulling model: {model_identifier}")
        
        # Check if it's a HuggingFace repository
        if '/' in model_identifier and not model_identifier.startswith('http'):
            return await self._pull_hf_model(model_identifier, specific_file)
        
        # Check if it's a URL
        elif model_identifier.startswith('http'):
            return await self._pull_url_model(model_identifier)
        
        # Assume it's an Ollama-style model name
        else:
            return await self._pull_ollama_style_model(model_identifier)
    
    async def _pull_hf_model(self, repo_id: str, filename: Optional[str] = None) -> str:
        """Download model from HuggingFace."""
        try:
            # List files in repository
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                raise ValueError(f"No GGUF files found in repository {repo_id}")
            
            # Select file to download
            if filename:
                if filename not in gguf_files:
                    raise ValueError(f"File {filename} not found in repository {repo_id}")
                target_file = filename
            else:
                # Auto-select best file (prefer Q4_K_M quantization)
                target_file = self._select_best_gguf_file(gguf_files)
            
            logger.info(f"Downloading {target_file} from {repo_id}")
            
            # Download file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=target_file,
                cache_dir=str(self.models_dir),
                resume_download=True
            )
            
            # Create model name
            model_name = f"{repo_id.replace('/', ':')}:{target_file.replace('.gguf', '')}"
            
            # Add to registry
            self.model_registry[model_name] = ModelInfo(
                name=model_name,
                path=local_path,
                size=Path(local_path).stat().st_size,
                format="GGUF",
                quantization=self._detect_quantization(target_file),
                capabilities=self._detect_capabilities(Path(local_path)),
                downloaded_at=time.time(),
            )
            
            self._save_model_registry()
            
            logger.info(f"Successfully downloaded model: {model_name}")
            return model_name
            
        except Exception as e:
            logger.error(f"Failed to download model from HuggingFace: {e}")
            raise
    
    def _select_best_gguf_file(self, gguf_files: List[str]) -> str:
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
    
    async def _pull_url_model(self, url: str) -> str:
        """Download model from direct URL."""
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        
        if not filename.endswith('.gguf'):
            raise ValueError("URL must point to a .gguf file")
        
        model_path = self.models_dir / filename
        
        logger.info(f"Downloading {filename} from {url}")
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
        
        model_name = filename.replace('.gguf', '')
        
        # Add to registry
        self.model_registry[model_name] = ModelInfo(
            name=model_name,
            path=str(model_path),
            size=model_path.stat().st_size,
            format="GGUF",
            quantization=self._detect_quantization(filename),
            capabilities=self._detect_capabilities(model_path),
            downloaded_at=time.time(),
        )
        
        self._save_model_registry()
        
        logger.info(f"Successfully downloaded model: {model_name}")
        return model_name
    
    async def _pull_ollama_style_model(self, model_name: str) -> str:
        """Handle Ollama-style model names by mapping to HuggingFace repos."""
        # Map common Ollama model names to HuggingFace repositories
        ollama_to_hf_map = {
            'llama2:7b': 'TheBloke/Llama-2-7B-Chat-GGUF',
            'llama2:13b': 'TheBloke/Llama-2-13B-Chat-GGUF',
            'llama2:70b': 'TheBloke/Llama-2-70B-Chat-GGUF',
            'codellama:7b': 'TheBloke/CodeLlama-7B-Instruct-GGUF',
            'codellama:13b': 'TheBloke/CodeLlama-13B-Instruct-GGUF',
            'mistral:7b': 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
            'llava:7b': 'mys/ggml_llava-v1.5-7b',
            'llava:13b': 'mys/ggml_llava-v1.5-13b',
        }
        
        if model_name in ollama_to_hf_map:
            hf_repo = ollama_to_hf_map[model_name]
            return await self._pull_hf_model(hf_repo)
        else:
            raise ValueError(f"Unknown Ollama-style model: {model_name}")
    
    async def load_model(self, model_name: str) -> None:
        """Load a model for inference."""
        if model_name not in self.model_registry:
            # Try to pull the model
            logger.info(f"Model {model_name} not found locally, attempting to download...")
            model_name = await self.pull_model(model_name)
        
        model_info = self.model_registry[model_name]
        
        # Unload current model if different
        if self.loaded_model and self.current_model_name != model_name:
            logger.info(f"Unloading current model: {self.current_model_name}")
            del self.loaded_model
            self.loaded_model = None
            self.current_model_name = None
        
        # Load new model if not already loaded
        if not self.loaded_model:
            logger.info(f"Loading model: {model_name}")
            
            # Get llama.cpp arguments from GPU manager
            llama_args = self.gpu_manager.get_llama_cpp_args()
            
            # Load model
            self.loaded_model = LlamaWrapper(
                model_path=model_info.path,
                **llama_args
            )
            
            self.current_model_name = model_name
            
            # Update last used time
            model_info.last_used = time.time()
            self._save_model_registry()
            
            logger.info(f"Successfully loaded model: {model_name}")
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self.model_registry.get(model_name)
    
    def remove_model(self, model_name: str) -> None:
        """Remove a model from disk and registry."""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.model_registry[model_name]
        
        # Unload if currently loaded
        if self.current_model_name == model_name:
            del self.loaded_model
            self.loaded_model = None
            self.current_model_name = None
        
        # Remove file
        model_path = Path(model_info.path)
        if model_path.exists():
            model_path.unlink()
        
        # Remove from registry
        del self.model_registry[model_name]
        self._save_model_registry()
        
        logger.info(f"Removed model: {model_name}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the loaded model."""
        if not self.loaded_model:
            raise RuntimeError("No model loaded. Use load_model() first.")
        
        return await self.loaded_model.generate(prompt, **kwargs)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with the loaded model."""
        if not self.loaded_model:
            raise RuntimeError("No model loaded. Use load_model() first.")
        
        return await self.loaded_model.chat(messages, **kwargs)
    
    async def benchmark(self, prompt: str, iterations: int = 5) -> Dict[str, float]:
        """Run performance benchmark on the loaded model."""
        if not self.loaded_model:
            raise RuntimeError("No model loaded. Use load_model() first.")
        
        logger.info(f"Running benchmark with {iterations} iterations")
        
        times = []
        token_counts = []
        
        for i in range(iterations):
            start_time = time.time()
            response = await self.generate(prompt, max_tokens=100)
            end_time = time.time()
            
            elapsed = end_time - start_time
            token_count = len(response.split())  # Rough token count
            
            times.append(elapsed)
            token_counts.append(token_count)
            
            logger.info(f"Iteration {i+1}: {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.1f} tok/s)")
        
        avg_time = sum(times) / len(times)
        total_tokens = sum(token_counts)
        avg_tokens_per_second = total_tokens / sum(times)
        
        return {
            'avg_latency': avg_time,
            'avg_tokens_per_second': avg_tokens_per_second,
            'total_tokens': total_tokens,
            'total_time': sum(times),
            'iterations': iterations
        }

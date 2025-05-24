"""
Configuration management for TurboLlama
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass


class HardwareConfig(BaseModel):
    """Hardware configuration settings."""
    auto_detect: bool = True
    gpu_backend: str = "auto"  # auto, cuda, vulkan, rocm, xpu, cpu
    gpu_layers: int = -1  # -1 means all layers
    gpu_memory_fraction: float = 0.9
    cpu_threads: Optional[int] = None
    prefer_gpu: bool = True
    numa_enabled: bool = True


class ModelConfig(BaseModel):
    """Model configuration settings."""
    default: str = "llama2:7b"
    cache_dir: str = "~/.turbollama/models"
    auto_download: bool = True
    context_size: int = 4096
    batch_size: int = 512
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 11434
    cors_origins: list = Field(default_factory=lambda: ["*"])
    enable_streaming: bool = True
    enable_function_calling: bool = True
    max_concurrent_requests: int = 10
    request_timeout: int = 300


class InterfaceConfig(BaseModel):
    """Gradio interface configuration."""
    port: int = 7860
    theme: str = "dark"
    enable_voice: bool = True
    show_performance: bool = True
    custom_css: Optional[str] = None
    enable_file_upload: bool = True
    max_file_size: int = 100  # MB


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_size: int = 10  # MB
    backup_count: int = 5


class Config(BaseModel):
    """Main configuration class."""
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def get_config_path(cls) -> Path:
        """Get the configuration file path."""
        config_dir = Path.home() / ".turbollama"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.yaml"
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from file."""
        if config_path:
            path = Path(config_path)
        else:
            path = cls.get_config_path()
            
        if path.exists():
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                return cls(**data)
        else:
            # Create default config
            config = cls()
            config.save(path)
            return config
    
    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if config_path is None:
            config_path = self.get_config_path()
            
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_mappings = {
            'TURBOLLAMA_GPU_BACKEND': ('hardware', 'gpu_backend'),
            'TURBOLLAMA_MODEL_CACHE': ('models', 'cache_dir'),
            'TURBOLLAMA_API_PORT': ('api', 'port'),
            'TURBOLLAMA_GUI_PORT': ('interface', 'port'),
            'TURBOLLAMA_LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                section_obj = getattr(self, section)
                if hasattr(section_obj, key):
                    # Convert to appropriate type
                    field_type = section_obj.__annotations__.get(key, str)
                    if field_type == int:
                        value = int(value)
                    elif field_type == float:
                        value = float(value)
                    elif field_type == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    
                    setattr(section_obj, key, value)

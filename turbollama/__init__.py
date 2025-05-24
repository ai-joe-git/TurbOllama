"""
TurboLlama - The Ultimate llama.cpp Wrapper with Integrated GUI

A high-performance AI inference engine that combines the speed of llama.cpp
with a modern Gradio interface and universal GPU support.
"""

__version__ = "1.0.0"
__author__ = "AI Joe"
__email__ = "ai.joe.git@gmail.com"
__license__ = "MIT"

from .core.llama_wrapper import LlamaWrapper
from .core.gpu_manager import GPUManager
from .core.model_manager import ModelManager
from .api.server import TurboLlamaServer
from .ui.gradio_interface import GradioInterface

__all__ = [
    "LlamaWrapper",
    "GPUManager", 
    "ModelManager",
    "TurboLlamaServer",
    "GradioInterface",
]

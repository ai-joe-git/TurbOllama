"""
GPU Management and Hardware Detection for TurboLlama
"""

import os
import platform
import subprocess
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None

from .config import Config

logger = logging.getLogger(__name__)


class GPUBackend(Enum):
    """Supported GPU backends."""
    CUDA = "cuda"
    VULKAN = "vulkan"
    ROCM = "rocm"
    XPU = "xpu"  # Intel GPU
    METAL = "metal"  # Apple Silicon
    CPU = "cpu"


@dataclass
class GPUInfo:
    """GPU information structure."""
    id: int
    name: str
    memory_total: float  # GB
    memory_free: float   # GB
    backend: GPUBackend
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None


class GPUManager:
    """Manages GPU detection and backend selection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.available_gpus: List[GPUInfo] = []
        self.selected_backend: GPUBackend = GPUBackend.CPU
        self.selected_gpu: Optional[GPUInfo] = None
        
        self._detect_gpus()
        self._select_optimal_backend()
    
    def _detect_gpus(self) -> None:
        """Detect all available GPUs."""
        self.available_gpus = []
        
        # Detect NVIDIA GPUs
        nvidia_gpus = self._detect_nvidia_gpus()
        self.available_gpus.extend(nvidia_gpus)
        
        # Detect AMD GPUs
        amd_gpus = self._detect_amd_gpus()
        self.available_gpus.extend(amd_gpus)
        
        # Detect Intel GPUs
        intel_gpus = self._detect_intel_gpus()
        self.available_gpus.extend(intel_gpus)
        
        # Detect Apple Silicon
        if platform.system() == "Darwin":
            apple_gpu = self._detect_apple_silicon()
            if apple_gpu:
                self.available_gpus.append(apple_gpu)
        
        logger.info(f"Detected {len(self.available_gpus)} GPU(s)")
        for gpu in self.available_gpus:
            logger.info(f"  {gpu.name} ({gpu.backend.value}) - {gpu.memory_total:.1f}GB")
    
    def _detect_nvidia_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using multiple methods."""
        gpus = []
        
        # Method 1: GPUtil
        if GPUtil:
            try:
                nvidia_gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(nvidia_gpus):
                    gpus.append(GPUInfo(
                        id=i,
                        name=gpu.name,
                        memory_total=gpu.memoryTotal / 1024,  # Convert MB to GB
                        memory_free=gpu.memoryFree / 1024,
                        backend=GPUBackend.CUDA,
                        driver_version=gpu.driver
                    ))
            except Exception as e:
                logger.debug(f"GPUtil detection failed: {e}")
        
        # Method 2: PyTorch CUDA
        if torch and torch.cuda.is_available() and not gpus:
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory / (1024**3)  # Convert to GB
                    
                    gpus.append(GPUInfo(
                        id=i,
                        name=props.name,
                        memory_total=memory_total,
                        memory_free=memory_total * 0.9,  # Estimate
                        backend=GPUBackend.CUDA,
                        compute_capability=f"{props.major}.{props.minor}"
                    ))
            except Exception as e:
                logger.debug(f"PyTorch CUDA detection failed: {e}")
        
        # Method 3: nvidia-smi
        if not gpus:
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 4:
                                gpus.append(GPUInfo(
                                    id=int(parts[0]),
                                    name=parts[1],
                                    memory_total=float(parts[2]) / 1024,  # MB to GB
                                    memory_free=float(parts[3]) / 1024,
                                    backend=GPUBackend.CUDA,
                                    driver_version=parts[4] if len(parts) > 4 else None
                                ))
            except Exception as e:
                logger.debug(f"nvidia-smi detection failed: {e}")
        
        return gpus
    
    def _detect_amd_gpus(self) -> List[GPUInfo]:
        """Detect AMD GPUs."""
        gpus = []
        
        # Check for ROCm support (Linux)
        if platform.system() == "Linux":
            try:
                result = subprocess.run([
                    'rocm-smi', '--showproductname', '--showmeminfo', 'vram'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    # Parse rocm-smi output
                    lines = result.stdout.strip().split('\n')
                    gpu_id = 0
                    for line in lines:
                        if 'GPU' in line and 'vram' in line.lower():
                            # Extract GPU info from rocm-smi output
                            # This is a simplified parser
                            gpus.append(GPUInfo(
                                id=gpu_id,
                                name="AMD GPU (ROCm)",
                                memory_total=8.0,  # Default estimate
                                memory_free=7.0,
                                backend=GPUBackend.ROCM
                            ))
                            gpu_id += 1
            except Exception as e:
                logger.debug(f"ROCm detection failed: {e}")
        
        # Check for Vulkan support (all platforms)
        if self._check_vulkan_support():
            # If we haven't detected AMD GPUs via ROCm, check via Vulkan
            if not gpus:
                try:
                    # Use vulkaninfo to detect AMD GPUs
                    result = subprocess.run([
                        'vulkaninfo', '--summary'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0 and 'AMD' in result.stdout:
                        gpus.append(GPUInfo(
                            id=0,
                            name="AMD GPU (Vulkan)",
                            memory_total=8.0,  # Default estimate
                            memory_free=7.0,
                            backend=GPUBackend.VULKAN
                        ))
                except Exception as e:
                    logger.debug(f"Vulkan AMD detection failed: {e}")
        
        return gpus
    
    def _detect_intel_gpus(self) -> List[GPUInfo]:
        """Detect Intel GPUs (Arc and integrated)."""
        gpus = []
        
        # Check for Intel Extension for PyTorch
        if ipex:
            try:
                if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
                    device_count = ipex.xpu.device_count()
                    for i in range(device_count):
                        device_name = ipex.xpu.get_device_name(i)
                        # Get memory info if available
                        try:
                            memory_info = ipex.xpu.get_device_properties(i)
                            memory_total = getattr(memory_info, 'total_memory', 8 * 1024**3) / 1024**3
                        except:
                            memory_total = 8.0  # Default estimate
                        
                        gpus.append(GPUInfo(
                            id=i,
                            name=device_name,
                            memory_total=memory_total,
                            memory_free=memory_total * 0.9,
                            backend=GPUBackend.XPU
                        ))
            except Exception as e:
                logger.debug(f"Intel GPU detection failed: {e}")
        
        # Fallback: Check for Intel graphics via system info
        if not gpus and platform.system() == "Windows":
            try:
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController', 'get', 'name'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Intel' in line and ('Arc' in line or 'Iris' in line or 'UHD' in line):
                            gpus.append(GPUInfo(
                                id=0,
                                name=line.strip(),
                                memory_total=4.0,  # Conservative estimate
                                memory_free=3.5,
                                backend=GPUBackend.XPU
                            ))
                            break
            except Exception as e:
                logger.debug(f"Windows Intel GPU detection failed: {e}")
        
        return gpus
    
    def _detect_apple_silicon(self) -> Optional[GPUInfo]:
        """Detect Apple Silicon GPU."""
        try:
            result = subprocess.run([
                'system_profiler', 'SPHardwareDataType'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                output = result.stdout
                if any(chip in output for chip in ['Apple M1', 'Apple M2', 'Apple M3']):
                    # Estimate memory based on system RAM
                    memory_total = 8.0  # Conservative estimate
                    return GPUInfo(
                        id=0,
                        name="Apple Silicon GPU",
                        memory_total=memory_total,
                        memory_free=memory_total * 0.8,
                        backend=GPUBackend.METAL
                    )
        except Exception as e:
            logger.debug(f"Apple Silicon detection failed: {e}")
        
        return None
    
    def _check_vulkan_support(self) -> bool:
        """Check if Vulkan is available."""
        try:
            result = subprocess.run([
                'vulkaninfo', '--summary'
            ], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _select_optimal_backend(self) -> None:
        """Select the optimal GPU backend based on available hardware."""
        if self.config.hardware.gpu_backend != "auto":
            # Use user-specified backend
            backend_map = {
                "cuda": GPUBackend.CUDA,
                "vulkan": GPUBackend.VULKAN,
                "rocm": GPUBackend.ROCM,
                "xpu": GPUBackend.XPU,
                "metal": GPUBackend.METAL,
                "cpu": GPUBackend.CPU
            }
            self.selected_backend = backend_map.get(
                self.config.hardware.gpu_backend, 
                GPUBackend.CPU
            )
        else:
            # Auto-select based on available GPUs
            if not self.config.hardware.prefer_gpu:
                self.selected_backend = GPUBackend.CPU
            elif any(gpu.backend == GPUBackend.CUDA for gpu in self.available_gpus):
                self.selected_backend = GPUBackend.CUDA
            elif any(gpu.backend == GPUBackend.METAL for gpu in self.available_gpus):
                self.selected_backend = GPUBackend.METAL
            elif any(gpu.backend == GPUBackend.XPU for gpu in self.available_gpus):
                self.selected_backend = GPUBackend.XPU
            elif any(gpu.backend == GPUBackend.ROCM for gpu in self.available_gpus):
                self.selected_backend = GPUBackend.ROCM
            elif any(gpu.backend == GPUBackend.VULKAN for gpu in self.available_gpus):
                self.selected_backend = GPUBackend.VULKAN
            else:
                self.selected_backend = GPUBackend.CPU
        
        # Select the best GPU for the chosen backend
        compatible_gpus = [
            gpu for gpu in self.available_gpus 
            if gpu.backend == self.selected_backend
        ]
        
        if compatible_gpus:
            # Select GPU with most free memory
            self.selected_gpu = max(compatible_gpus, key=lambda g: g.memory_free)
        
        logger.info(f"Selected backend: {self.selected_backend.value}")
        if self.selected_gpu:
            logger.info(f"Selected GPU: {self.selected_gpu.name}")
    
    def get_llama_cpp_args(self) -> Dict[str, any]:
        """Get llama.cpp arguments for the selected backend."""
        args = {
            'n_ctx': self.config.models.context_size,
            'n_batch': self.config.models.batch_size,
            'verbose': False,
        }
        
        if self.selected_backend == GPUBackend.CPU:
            args.update({
                'n_gpu_layers': 0,
                'n_threads': self.config.hardware.cpu_threads or os.cpu_count(),
            })
        else:
            args.update({
                'n_gpu_layers': self.config.hardware.gpu_layers,
            })
            
            if self.selected_backend == GPUBackend.CUDA:
                args['n_gpu_layers'] = self.config.hardware.gpu_layers
                if self.selected_gpu:
                    args['main_gpu'] = self.selected_gpu.id
            
            elif self.selected_backend == GPUBackend.VULKAN:
                args['n_gpu_layers'] = self.config.hardware.gpu_layers
                # Vulkan-specific settings would go here
            
            elif self.selected_backend == GPUBackend.ROCM:
                args['n_gpu_layers'] = self.config.hardware.gpu_layers
                # ROCm-specific settings would go here
            
            elif self.selected_backend == GPUBackend.XPU:
                args['n_gpu_layers'] = self.config.hardware.gpu_layers
                # Intel GPU-specific settings would go here
            
            elif self.selected_backend == GPUBackend.METAL:
                args['n_gpu_layers'] = self.config.hardware.gpu_layers
                # Metal-specific settings would go here
        
        return args
    
    def get_hardware_info(self) -> Dict[str, any]:
        """Get comprehensive hardware information."""
        return {
            'selected_backend': self.selected_backend.value,
            'selected_gpu': {
                'name': self.selected_gpu.name,
                'memory_total': self.selected_gpu.memory_total,
                'memory_free': self.selected_gpu.memory_free,
            } if self.selected_gpu else None,
            'available_gpus': [
                {
                    'id': gpu.id,
                    'name': gpu.name,
                    'backend': gpu.backend.value,
                    'memory_total': gpu.memory_total,
                    'memory_free': gpu.memory_free,
                }
                for gpu in self.available_gpus
            ],
            'cpu_threads': self.config.hardware.cpu_threads or os.cpu_count(),
            'platform': platform.system(),
        }

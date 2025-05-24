"""
Hardware Detection and System Information
"""

import os
import platform
import subprocess
import psutil
import logging
from typing import Dict, List, Optional, Any
import json

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """Comprehensive hardware detection."""
    return {
        'cpu': get_cpu_info(),
        'memory': get_memory_info(),
        'gpus': get_gpu_info(),
        'platform': get_platform_info(),
        'capabilities': get_system_capabilities()
    }


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    info = {
        'cores': os.cpu_count(),
        'threads': os.cpu_count(),
        'architecture': platform.machine(),
        'name': 'Unknown CPU'
    }
    
    # Try to get detailed CPU info
    if cpuinfo:
        try:
            cpu_data = cpuinfo.get_cpu_info()
            info.update({
                'name': cpu_data.get('brand_raw', 'Unknown CPU'),
                'architecture': cpu_data.get('arch', platform.machine()),
                'bits': cpu_data.get('bits', 64),
                'frequency': cpu_data.get('hz_advertised_friendly', 'Unknown'),
                'vendor': cpu_data.get('vendor_id_raw', 'Unknown'),
                'flags': cpu_data.get('flags', [])
            })
        except Exception as e:
            logger.debug(f"Failed to get detailed CPU info: {e}")
    
    # Detect SIMD capabilities
    info['simd_support'] = detect_simd_support()
    
    # Detect performance/efficiency cores (Intel)
    info['core_types'] = detect_core_types()
    
    return info


def get_memory_info() -> Dict[str, Any]:
    """Get memory information."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total': memory.total / (1024**3),  # GB
        'available': memory.available / (1024**3),
        'used': memory.used / (1024**3),
        'percentage': memory.percent,
        'swap_total': swap.total / (1024**3),
        'swap_used': swap.used / (1024**3),
        'swap_percentage': swap.percent
    }


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get GPU information."""
    gpus = []
    
    # NVIDIA GPUs
    nvidia_gpus = detect_nvidia_gpus()
    gpus.extend(nvidia_gpus)
    
    # AMD GPUs
    amd_gpus = detect_amd_gpus()
    gpus.extend(amd_gpus)
    
    # Intel GPUs
    intel_gpus = detect_intel_gpus()
    gpus.extend(intel_gpus)
    
    # Apple Silicon
    if platform.system() == "Darwin":
        apple_gpu = detect_apple_silicon()
        if apple_gpu:
            gpus.append(apple_gpu)
    
    return gpus


def detect_nvidia_gpus() -> List[Dict[str, Any]]:
    """Detect NVIDIA GPUs."""
    gpus = []
    
    # Method 1: GPUtil
    if GPUtil:
        try:
            nvidia_gpus = GPUtil.getGPUs()
            for gpu in nvidia_gpus:
                gpus.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory': gpu.memoryTotal,  # MB
                    'memory_free': gpu.memoryFree,
                    'memory_used': gpu.memoryUsed,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100,  # Percentage
                    'backend': 'cuda',
                    'driver_version': gpu.driver
                })
        except Exception as e:
            logger.debug(f"GPUtil detection failed: {e}")
    
    # Method 2: nvidia-ml-py
    if not gpus:
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpus.append({
                    'id': i,
                    'name': name,
                    'memory': memory_info.total // (1024**2),  # MB
                    'memory_free': memory_info.free // (1024**2),
                    'memory_used': memory_info.used // (1024**2),
                    'backend': 'cuda'
                })
        except Exception as e:
            logger.debug(f"pynvml detection failed: {e}")
    
    # Method 3: nvidia-smi
    if not gpus:
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,driver_version',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            gpus.append({
                                'id': int(parts[0]),
                                'name': parts[1],
                                'memory': int(parts[2]),
                                'memory_free': int(parts[3]),
                                'memory_used': int(parts[4]),
                                'temperature': int(parts[5]) if parts[5] != '[Not Supported]' else None,
                                'load': int(parts[6]) if parts[6] != '[Not Supported]' else None,
                                'backend': 'cuda',
                                'driver_version': parts[7] if len(parts) > 7 else None
                            })
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")
    
    return gpus


def detect_amd_gpus() -> List[Dict[str, Any]]:
    """Detect AMD GPUs."""
    gpus = []
    
    # ROCm detection (Linux)
    if platform.system() == "Linux":
        try:
            result = subprocess.run([
                'rocm-smi', '--showid', '--showproductname', '--showmeminfo', 'vram', '--showuse', '--showtemp'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse rocm-smi output (simplified)
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if 'GPU' in line and i < len(lines) - 1:
                        gpus.append({
                            'id': i,
                            'name': 'AMD GPU (ROCm)',
                            'memory': 8192,  # Default estimate
                            'backend': 'rocm'
                        })
        except Exception as e:
            logger.debug(f"ROCm detection failed: {e}")
    
    # Vulkan detection (all platforms)
    if not gpus:
        try:
            result = subprocess.run([
                'vulkaninfo', '--summary'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and 'AMD' in result.stdout:
                gpus.append({
                    'id': 0,
                    'name': 'AMD GPU (Vulkan)',
                    'memory': 8192,  # Default estimate
                    'backend': 'vulkan'
                })
        except Exception as e:
            logger.debug(f"Vulkan detection failed: {e}")
    
    return gpus


def detect_intel_gpus() -> List[Dict[str, Any]]:
    """Detect Intel GPUs."""
    gpus = []
    
    # Intel GPU detection via system info
    if platform.system() == "Windows":
        try:
            result = subprocess.run([
                'wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterRAM'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip() and 'Intel' in line:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            memory_mb = int(parts[0]) // (1024**2) if parts[0].isdigit() else 4096
                            name = ' '.join(parts[1:])
                            
                            gpus.append({
                                'id': len(gpus),
                                'name': name,
                                'memory': memory_mb,
                                'backend': 'xpu'
                            })
        except Exception as e:
            logger.debug(f"Intel GPU detection failed: {e}")
    
    elif platform.system() == "Linux":
        try:
            # Check for Intel GPU via lspci
            result = subprocess.run([
                'lspci', '-nn'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA' in line and 'Intel' in line:
                        gpus.append({
                            'id': len(gpus),
                            'name': 'Intel Integrated Graphics',
                            'memory': 4096,  # Default estimate
                            'backend': 'xpu'
                        })
                        break
        except Exception as e:
            logger.debug(f"lspci detection failed: {e}")
    
    return gpus


def detect_apple_silicon() -> Optional[Dict[str, Any]]:
    """Detect Apple Silicon GPU."""
    try:
        result = subprocess.run([
            'system_profiler', 'SPHardwareDataType'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            for chip in ['Apple M1', 'Apple M2', 'Apple M3', 'Apple M4']:
                if chip in output:
                    # Estimate memory based on chip type
                    if 'Pro' in output or 'Max' in output:
                        memory = 16384  # 16GB estimate
                    else:
                        memory = 8192   # 8GB estimate
                    
                    return {
                        'id': 0,
                        'name': f"{chip} GPU",
                        'memory': memory,
                        'backend': 'metal'
                    }
    except Exception as e:
        logger.debug(f"Apple Silicon detection failed: {e}")
    
    return None


def get_platform_info() -> Dict[str, Any]:
    """Get platform information."""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation()
    }


def detect_simd_support() -> Dict[str, bool]:
    """Detect SIMD instruction set support."""
    support = {
        'sse': False,
        'sse2': False,
        'sse3': False,
        'ssse3': False,
        'sse4_1': False,
        'sse4_2': False,
        'avx': False,
        'avx2': False,
        'avx512': False,
        'neon': False  # ARM
    }
    
    if cpuinfo:
        try:
            cpu_data = cpuinfo.get_cpu_info()
            flags = cpu_data.get('flags', [])
            
            for flag in flags:
                flag_lower = flag.lower()
                if flag_lower in support:
                    support[flag_lower] = True
                elif 'avx512' in flag_lower:
                    support['avx512'] = True
                elif flag_lower == 'neon':
                    support['neon'] = True
        except Exception as e:
            logger.debug(f"SIMD detection failed: {e}")
    
    return support


def detect_core_types() -> Dict[str, int]:
    """Detect performance and efficiency cores (Intel hybrid architecture)."""
    core_types = {
        'performance': 0,
        'efficiency': 0,
        'total': os.cpu_count()
    }
    
    # This is a simplified detection - real implementation would need
    # more sophisticated CPU topology detection
    if platform.system() == "Windows":
        try:
            result = subprocess.run([
                'wmic', 'cpu', 'get', 'NumberOfCores,NumberOfLogicalProcessors'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse output to detect hybrid architecture
                # This is a placeholder - actual implementation would be more complex
                core_types['performance'] = os.cpu_count()
        except Exception as e:
            logger.debug(f"Core type detection failed: {e}")
    
    return core_types


def get_system_capabilities() -> Dict[str, bool]:
    """Get system capabilities for AI inference."""
    capabilities = {
        'cuda_available': False,
        'rocm_available': False,
        'vulkan_available': False,
        'metal_available': False,
        'openvino_available': False,
        'tensorrt_available': False
    }
    
    # CUDA
    try:
        import torch
        capabilities['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    # ROCm
    if platform.system() == "Linux":
        try:
            result = subprocess.run(['which', 'rocm-smi'], capture_output=True)
            capabilities['rocm_available'] = result.returncode == 0
        except:
            pass
    
    # Vulkan
    try:
        result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, timeout=5)
        capabilities['vulkan_available'] = result.returncode == 0
    except:
        pass
    
    # Metal (macOS)
    if platform.system() == "Darwin":
        capabilities['metal_available'] = True
    
    return capabilities


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    return {
        'hardware': detect_hardware(),
        'environment': {
            'python_path': os.sys.executable,
            'python_version': platform.python_version(),
            'working_directory': os.getcwd(),
            'environment_variables': {
                key: value for key, value in os.environ.items()
                if any(keyword in key.upper() for keyword in [
                    'CUDA', 'HIP', 'VULKAN', 'INTEL', 'PYTORCH', 'TENSORFLOW'
                ])
            }
        },
        'disk_usage': {
            path: {
                'total': psutil.disk_usage(path).total / (1024**3),
                'free': psutil.disk_usage(path).free / (1024**3),
                'used': psutil.disk_usage(path).used / (1024**3)
            }
            for path in ['/'] if platform.system() != "Windows" else ['C:\\']
        }
    }

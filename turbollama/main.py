#!/usr/bin/env python3
"""
TurboLlama Main Entry Point
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
import logging
from typing import Optional

from .core.config import Config
from .api.server import TurboLlamaServer
from .ui.gradio_interface import GradioInterface
from .core.gpu_manager import GPUManager
from .core.model_manager import ModelManager
from .utils.hardware import detect_hardware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="TurboLlama - The Ultimate llama.cpp Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  turbollama serve --model llama2:7b --gui
  turbollama serve --hf-model microsoft/DialoGPT-medium
  turbollama pull llama2:7b
  turbollama list
  turbollama benchmark --model llama2:7b
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start TurboLlama server')
    serve_parser.add_argument('--model', '-m', type=str, help='Model name to serve')
    serve_parser.add_argument('--hf-model', type=str, help='HuggingFace model repository')
    serve_parser.add_argument('--hf-file', type=str, help='Specific file from HF repo')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=11434, help='Port for API server')
    serve_parser.add_argument('--gui', action='store_true', help='Launch Gradio interface')
    serve_parser.add_argument('--gui-port', type=int, default=7860, help='Port for Gradio interface')
    serve_parser.add_argument('--backend', choices=['auto', 'cuda', 'vulkan', 'rocm', 'xpu', 'cpu'], 
                             default='auto', help='GPU backend to use')
    serve_parser.add_argument('--gpu-layers', type=int, default=-1, help='Number of layers to offload to GPU')
    serve_parser.add_argument('--context-size', type=int, default=4096, help='Context window size')
    serve_parser.add_argument('--batch-size', type=int, default=512, help='Batch size for processing')
    serve_parser.add_argument('--threads', type=int, help='Number of CPU threads')
    serve_parser.add_argument('--config', type=str, help='Path to configuration file')
    
    # Pull command
    pull_parser = subparsers.add_parser('pull', help='Download a model')
    pull_parser.add_argument('model', help='Model name or HuggingFace repository')
    pull_parser.add_argument('--file', help='Specific file to download')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a model')
    remove_parser.add_argument('model', help='Model name to remove')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('--model', '-m', required=True, help='Model to benchmark')
    benchmark_parser.add_argument('--prompt', default='Hello, how are you?', help='Prompt for benchmark')
    benchmark_parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
    
    return parser


async def serve_command(args):
    """Handle serve command."""
    try:
        # Load configuration
        config = Config.load(args.config)
        
        # Override config with command line arguments
        if args.host:
            config.api.host = args.host
        if args.port:
            config.api.port = args.port
        if args.backend != 'auto':
            config.hardware.gpu_backend = args.backend
        if args.gpu_layers != -1:
            config.hardware.gpu_layers = args.gpu_layers
        if args.context_size:
            config.models.context_size = args.context_size
        if args.batch_size:
            config.models.batch_size = args.batch_size
        if args.threads:
            config.hardware.cpu_threads = args.threads
            
        # Initialize GPU manager
        gpu_manager = GPUManager(config)
        
        # Initialize model manager
        model_manager = ModelManager(config, gpu_manager)
        
        # Determine model to load
        model_name = None
        if args.model:
            model_name = args.model
        elif args.hf_model:
            model_name = await model_manager.pull_hf_model(args.hf_model, args.hf_file)
        elif config.models.default:
            model_name = config.models.default
        else:
            logger.error("No model specified. Use --model or --hf-model")
            return 1
            
        # Load the model
        logger.info(f"Loading model: {model_name}")
        await model_manager.load_model(model_name)
        
        # Create server
        server = TurboLlamaServer(config, model_manager)
        
        # Start Gradio interface if requested
        if args.gui:
            gradio_interface = GradioInterface(config, model_manager)
            gradio_task = asyncio.create_task(
                gradio_interface.launch(port=args.gui_port)
            )
            logger.info(f"Gradio interface starting on http://localhost:{args.gui_port}")
        
        # Start API server
        logger.info(f"TurboLlama API server starting on http://{args.host}:{args.port}")
        logger.info("API endpoints:")
        logger.info("  - OpenAI compatible: /v1/chat/completions")
        logger.info("  - Ollama compatible: /api/chat, /api/generate")
        logger.info("  - Health check: /health")
        
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Shutting down TurboLlama...")
        return 0
    except Exception as e:
        logger.error(f"Error starting TurboLlama: {e}")
        return 1


async def pull_command(args):
    """Handle pull command."""
    try:
        config = Config.load()
        gpu_manager = GPUManager(config)
        model_manager = ModelManager(config, gpu_manager)
        
        logger.info(f"Downloading model: {args.model}")
        await model_manager.pull_model(args.model, args.file)
        logger.info("Model downloaded successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return 1


def list_command(args):
    """Handle list command."""
    try:
        config = Config.load()
        gpu_manager = GPUManager(config)
        model_manager = ModelManager(config, gpu_manager)
        
        models = model_manager.list_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No models found. Use 'turbollama pull <model>' to download models.")
        return 0
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return 1


def remove_command(args):
    """Handle remove command."""
    try:
        config = Config.load()
        gpu_manager = GPUManager(config)
        model_manager = ModelManager(config, gpu_manager)
        
        model_manager.remove_model(args.model)
        logger.info(f"Model {args.model} removed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error removing model: {e}")
        return 1


def info_command(args):
    """Handle info command."""
    try:
        hardware_info = detect_hardware()
        
        print("TurboLlama System Information")
        print("=" * 40)
        print(f"Version: {__version__}")
        print(f"Python: {sys.version}")
        print()
        
        print("Hardware Information:")
        print(f"  CPU: {hardware_info['cpu']['name']}")
        print(f"  CPU Cores: {hardware_info['cpu']['cores']}")
        print(f"  RAM: {hardware_info['memory']['total']:.1f} GB")
        print()
        
        if hardware_info['gpus']:
            print("GPU Information:")
            for i, gpu in enumerate(hardware_info['gpus']):
                print(f"  GPU {i}: {gpu['name']}")
                print(f"    Memory: {gpu['memory']:.1f} GB")
                print(f"    Backend: {gpu['backend']}")
        else:
            print("No GPUs detected")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return 1


async def benchmark_command(args):
    """Handle benchmark command."""
    try:
        config = Config.load()
        gpu_manager = GPUManager(config)
        model_manager = ModelManager(config, gpu_manager)
        
        # Load model for benchmarking
        await model_manager.load_model(args.model)
        
        logger.info(f"Running benchmark with model: {args.model}")
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Iterations: {args.iterations}")
        
        # Run benchmark
        results = await model_manager.benchmark(
            args.prompt, 
            iterations=args.iterations
        )
        
        print("\nBenchmark Results:")
        print("=" * 40)
        print(f"Average tokens/second: {results['avg_tokens_per_second']:.1f}")
        print(f"Average latency: {results['avg_latency']:.2f}s")
        print(f"Total tokens generated: {results['total_tokens']}")
        print(f"Total time: {results['total_time']:.2f}s")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return 1


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle commands
    if args.command == 'serve':
        return await serve_command(args)
    elif args.command == 'pull':
        return await pull_command(args)
    elif args.command == 'list':
        return list_command(args)
    elif args.command == 'remove':
        return remove_command(args)
    elif args.command == 'info':
        return info_command(args)
    elif args.command == 'benchmark':
        return await benchmark_command(args)
    else:
        parser.print_help()
        return 1


def cli_main():
    """CLI entry point for setuptools."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli_main()

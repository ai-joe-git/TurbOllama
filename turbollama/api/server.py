"""
Main FastAPI Server for TurboLlama
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..core.config import Config
from ..core.model_manager import ModelManager
from .openai_routes import OpenAIRoutes
from .ollama_routes import OllamaRoutes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting TurboLlama API server...")
    yield
    logger.info("Shutting down TurboLlama API server...")


class TurboLlamaServer:
    """Main TurboLlama API server."""
    
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        
        # Create FastAPI app
        self.app = FastAPI(
            title="TurboLlama API",
            description="High-performance llama.cpp wrapper with OpenAI and Ollama compatibility",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=lifespan
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Request tracking
        self.active_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request tracking middleware
        @self.app.middleware("http")
        async def track_requests(request: Request, call_next):
            self.active_requests += 1
            self.total_requests += 1
            start_time = time.time()
            
            try:
                response = await call_next(request)
                return response
            finally:
                self.active_requests -= 1
                process_time = time.time() - start_time
                logger.debug(f"{request.method} {request.url.path} - {process_time:.3f}s")
    
    def _setup_routes(self):
        """Setup API routes."""
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time,
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "model_loaded": self.model_manager.loaded_model is not None,
                "current_model": self.model_manager.current_model_name
            }
        
        # System information
        @self.app.get("/api/system")
        async def system_info():
            """Get system information."""
            return {
                "hardware": self.model_manager.gpu_manager.get_hardware_info(),
                "models": {
                    "available": self.model_manager.list_models(),
                    "current": self.model_manager.current_model_name,
                    "loaded": self.model_manager.loaded_model is not None
                },
                "performance": self.model_manager.loaded_model.get_performance_stats() 
                             if self.model_manager.loaded_model else None,
                "config": {
                    "max_concurrent_requests": self.config.api.max_concurrent_requests,
                    "request_timeout": self.config.api.request_timeout,
                    "streaming_enabled": self.config.api.enable_streaming
                }
            }
        
        # Performance metrics
        @self.app.get("/api/performance")
        async def performance_metrics():
            """Get performance metrics."""
            if not self.model_manager.loaded_model:
                raise HTTPException(status_code=404, detail="No model loaded")
            
            stats = self.model_manager.loaded_model.get_performance_stats()
            return {
                "model": self.model_manager.current_model_name,
                "performance": stats,
                "server": {
                    "uptime": time.time() - self.start_time,
                    "active_requests": self.active_requests,
                    "total_requests": self.total_requests
                }
            }
        
        # Model management endpoints
        @self.app.get("/api/models")
        async def list_models():
            """List available models."""
            models = []
            for model_name in self.model_manager.list_models():
                model_info = self.model_manager.get_model_info(model_name)
                if model_info:
                    models.append({
                        "name": model_info.name,
                        "size": model_info.size,
                        "format": model_info.format,
                        "quantization": model_info.quantization,
                        "capabilities": model_info.capabilities or [],
                        "last_used": model_info.last_used
                    })
            
            return {"models": models}
        
        @self.app.post("/api/models/load")
        async def load_model(request: Dict[str, str]):
            """Load a specific model."""
            model_name = request.get("model")
            if not model_name:
                raise HTTPException(status_code=400, detail="Model name required")
            
            try:
                await self.model_manager.load_model(model_name)
                return {"status": "success", "model": model_name}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/models/pull")
        async def pull_model(request: Dict[str, str], background_tasks: BackgroundTasks):
            """Pull/download a model."""
            model_identifier = request.get("model")
            if not model_identifier:
                raise HTTPException(status_code=400, detail="Model identifier required")
            
            # Start download in background
            background_tasks.add_task(
                self._download_model_background, 
                model_identifier, 
                request.get("file")
            )
            
            return {"status": "downloading", "model": model_identifier}
        
        @self.app.delete("/api/models/{model_name}")
        async def remove_model(model_name: str):
            """Remove a model."""
            try:
                self.model_manager.remove_model(model_name)
                return {"status": "success", "message": f"Model {model_name} removed"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Benchmark endpoint
        @self.app.post("/api/benchmark")
        async def benchmark_model(request: Dict[str, Any]):
            """Run performance benchmark."""
            if not self.model_manager.loaded_model:
                raise HTTPException(status_code=404, detail="No model loaded")
            
            prompt = request.get("prompt", "Hello, how are you?")
            iterations = request.get("iterations", 5)
            
            try:
                results = await self.model_manager.benchmark(prompt, iterations)
                return {
                    "model": self.model_manager.current_model_name,
                    "benchmark": results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Setup OpenAI compatible routes
        openai_routes = OpenAIRoutes(self.model_manager)
        self.app.include_router(openai_routes.router, prefix="/v1")
        
        # Setup Ollama compatible routes
        ollama_routes = OllamaRoutes(self.model_manager)
        self.app.include_router(ollama_routes.router, prefix="/api")
    
    async def _download_model_background(self, model_identifier: str, file: Optional[str] = None):
        """Download model in background."""
        try:
            logger.info(f"Starting background download: {model_identifier}")
            await self.model_manager.pull_model(model_identifier, file)
            logger.info(f"Background download completed: {model_identifier}")
        except Exception as e:
            logger.error(f"Background download failed: {e}")
    
    async def start(self):
        """Start the server."""
        config = uvicorn.Config(
            app=self.app,
            host=self.config.api.host,
            port=self.config.api.port,
            log_level="info",
            access_log=True,
            loop="asyncio"
        )
        
        server = uvicorn.Server(config)
        await server.serve()

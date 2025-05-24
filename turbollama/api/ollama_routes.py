"""
Ollama Compatible API Routes
"""

import time
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class OllamaMessage(BaseModel):
    """Ollama message format."""
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    """Ollama chat request."""
    model: str
    messages: List[OllamaMessage]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class OllamaGenerateRequest(BaseModel):
    """Ollama generate request."""
    model: str
    prompt: str
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    context: Optional[List[int]] = None
    raw: Optional[bool] = False


class OllamaPullRequest(BaseModel):
    """Ollama pull request."""
    name: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = True


class OllamaRoutes:
    """Ollama compatible API routes."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Ollama compatible routes."""
        
        @self.router.get("/tags")
        async def list_models():
            """List models in Ollama format."""
            models = []
            for model_name in self.model_manager.list_models():
                model_info = self.model_manager.get_model_info(model_name)
                if model_info:
                    models.append({
                        "name": model_name,
                        "modified_at": datetime.fromtimestamp(
                            model_info.last_used or model_info.downloaded_at or time.time()
                        ).isoformat() + "Z",
                        "size": model_info.size,
                        "digest": f"sha256:{'0' * 64}",  # Placeholder digest
                        "details": {
                            "format": model_info.format,
                            "family": "llama",
                            "families": ["llama"],
                            "parameter_size": self._estimate_parameters(model_info.size),
                            "quantization_level": model_info.quantization or "unknown"
                        }
                    })
            
            return {"models": models}
        
        @self.router.post("/chat")
        async def chat(request: OllamaChatRequest):
            """Ollama chat endpoint."""
            # Ensure model is loaded
            if self.model_manager.current_model_name != request.model:
                try:
                    await self.model_manager.load_model(request.model)
                except Exception as e:
                    raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
            
            if not self.model_manager.loaded_model:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Convert messages
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            if request.stream:
                return StreamingResponse(
                    self._stream_chat(request, messages),
                    media_type="application/x-ndjson"
                )
            else:
                return await self._generate_chat(request, messages)
        
        @self.router.post("/generate")
        async def generate(request: OllamaGenerateRequest):
            """Ollama generate endpoint."""
            # Ensure model is loaded
            if self.model_manager.current_model_name != request.model:
                try:
                    await self.model_manager.load_model(request.model)
                except Exception as e:
                    raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
            
            if not self.model_manager.loaded_model:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            if request.stream:
                return StreamingResponse(
                    self._stream_generate(request),
                    media_type="application/x-ndjson"
                )
            else:
                return await self._generate_response(request)
        
        @self.router.post("/pull")
        async def pull_model(request: OllamaPullRequest):
            """Pull/download a model."""
            if request.stream:
                return StreamingResponse(
                    self._stream_pull(request.name),
                    media_type="application/x-ndjson"
                )
            else:
                try:
                    await self.model_manager.pull_model(request.name)
                    return {"status": "success"}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.delete("/delete")
        async def delete_model(request: Dict[str, str]):
            """Delete a model."""
            model_name = request.get("name")
            if not model_name:
                raise HTTPException(status_code=400, detail="Model name required")
            
            try:
                self.model_manager.remove_model(model_name)
                return {"status": "success"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/show")
        async def show_model(request: Dict[str, str]):
            """Show model information."""
            model_name = request.get("name")
            if not model_name:
                raise HTTPException(status_code=400, detail="Model name required")
            
            model_info = self.model_manager.get_model_info(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail="Model not found")
            
            return {
                "license": "Unknown",
                "modelfile": f"FROM {model_info.path}",
                "parameters": self._get_model_parameters(model_info),
                "template": "{{ .Prompt }}",
                "details": {
                    "format": model_info.format,
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": self._estimate_parameters(model_info.size),
                    "quantization_level": model_info.quantization or "unknown"
                }
            }
        
        @self.router.post("/copy")
        async def copy_model(request: Dict[str, str]):
            """Copy a model (not implemented)."""
            raise HTTPException(status_code=501, detail="Model copying not implemented")
        
        @self.router.get("/ps")
        async def list_running_models():
            """List running models."""
            if self.model_manager.loaded_model and self.model_manager.current_model_name:
                model_info = self.model_manager.get_model_info(self.model_manager.current_model_name)
                return {
                    "models": [{
                        "name": self.model_manager.current_model_name,
                        "size": model_info.size if model_info else 0,
                        "size_vram": 0,  # Would need GPU memory tracking
                        "digest": f"sha256:{'0' * 64}",
                        "details": {
                            "format": model_info.format if model_info else "unknown",
                            "family": "llama",
                            "families": ["llama"],
                            "parameter_size": self._estimate_parameters(model_info.size) if model_info else "unknown",
                            "quantization_level": model_info.quantization if model_info else "unknown"
                        },
                        "expires_at": datetime.fromtimestamp(time.time() + 3600).isoformat() + "Z"
                    }]
                }
            else:
                return {"models": []}
    
    async def _generate_chat(self, request: OllamaChatRequest, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate non-streaming chat response."""
        try:
            start_time = time.time()
            
            # Extract options
            options = request.options or {}
            temperature = options.get("temperature", 0.7)
            max_tokens = options.get("num_predict", 512)
            top_p = options.get("top_p", 0.9)
            
            response_text = await self.model_manager.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            return {
                "model": request.model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "done": True,
                "total_duration": int((time.time() - start_time) * 1e9),  # nanoseconds
                "load_duration": 0,
                "prompt_eval_count": sum(len(msg["content"].split()) for msg in messages),
                "prompt_eval_duration": 0,
                "eval_count": len(response_text.split()),
                "eval_duration": int((time.time() - start_time) * 1e9)
            }
            
        except Exception as e:
            logger.error(f"Error in Ollama chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _stream_chat(self, request: OllamaChatRequest, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Generate streaming chat response."""
        try:
            start_time = time.time()
            
            # Extract options
            options = request.options or {}
            temperature = options.get("temperature", 0.7)
            max_tokens = options.get("num_predict", 512)
            top_p = options.get("top_p", 0.9)
            
            response_text = ""
            async for chunk in self.model_manager.loaded_model.chat_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            ):
                response_text += chunk
                
                chunk_data = {
                    "model": request.model,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "message": {
                        "role": "assistant",
                        "content": chunk
                    },
                    "done": False
                }
                yield json.dumps(chunk_data) + "\n"
            
            # Final chunk
            final_chunk = {
                "model": request.model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True,
                "total_duration": int((time.time() - start_time) * 1e9),
                "load_duration": 0,
                "prompt_eval_count": sum(len(msg["content"].split()) for msg in messages),
                "prompt_eval_duration": 0,
                "eval_count": len(response_text.split()),
                "eval_duration": int((time.time() - start_time) * 1e9)
            }
            yield json.dumps(final_chunk) + "\n"
            
        except Exception as e:
            logger.error(f"Error in streaming Ollama chat: {e}")
            error_chunk = {
                "error": str(e)
            }
            yield json.dumps(error_chunk) + "\n"
    
    async def _generate_response(self, request: OllamaGenerateRequest) -> Dict[str, Any]:
        """Generate non-streaming response."""
        try:
            start_time = time.time()
            
            # Extract options
            options = request.options or {}
            temperature = options.get("temperature", 0.7)
            max_tokens = options.get("num_predict", 512)
            top_p = options.get("top_p", 0.9)
            
            response_text = await self.model_manager.generate(
                request.prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            return {
                "model": request.model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "response": response_text,
                "done": True,
                "context": [],  # Would need context tracking
                "total_duration": int((time.time() - start_time) * 1e9),
                "load_duration": 0,
                "prompt_eval_count": len(request.prompt.split()),
                "prompt_eval_duration": 0,
                "eval_count": len(response_text.split()),
                "eval_duration": int((time.time() - start_time) * 1e9)
            }
            
        except Exception as e:
            logger.error(f"Error in Ollama generate: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _stream_generate(self, request: OllamaGenerateRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        try:
            start_time = time.time()
            
            # Extract options
            options = request.options or {}
            temperature = options.get("temperature", 0.7)
            max_tokens = options.get("num_predict", 512)
            top_p = options.get("top_p", 0.9)
            
            response_text = ""
            async for chunk in self.model_manager.loaded_model.generate_stream(
                request.prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            ):
                response_text += chunk
                
                chunk_data = {
                    "model": request.model,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "response": chunk,
                    "done": False
                }
                yield json.dumps(chunk_data) + "\n"
            
            # Final chunk
            final_chunk = {
                "model": request.model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "response": "",
                "done": True,
                "context": [],
                "total_duration": int((time.time() - start_time) * 1e9),
                "load_duration": 0,
                "prompt_eval_count": len(request.prompt.split()),
                "prompt_eval_duration": 0,
                "eval_count": len(response_text.split()),
                "eval_duration": int((time.time() - start_time) * 1e9)
            }
            yield json.dumps(final_chunk) + "\n"
            
        except Exception as e:
            logger.error(f"Error in streaming Ollama generate: {e}")
            error_chunk = {
                "error": str(e)
            }
            yield json.dumps(error_chunk) + "\n"
    
    async def _stream_pull(self, model_name: str) -> AsyncGenerator[str, None]:
        """Stream model download progress."""
        try:
            # Send initial status
            yield json.dumps({
                "status": "pulling manifest"
            }) + "\n"
            
            # Start download
            await self.model_manager.pull_model(model_name)
            
            # Send completion status
            yield json.dumps({
                "status": "success"
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            yield json.dumps({
                "error": str(e)
            }) + "\n"
    
    def _estimate_parameters(self, size_bytes: int) -> str:
        """Estimate parameter count from model size."""
        # Rough estimation: 1B parameters ≈ 2GB for FP16
        params_billion = size_bytes / (2 * 1024**3)
        if params_billion < 1:
            return f"{params_billion * 1000:.0f}M"
        else:
            return f"{params_billion:.1f}B"
    
    def _get_model_parameters(self, model_info) -> str:
        """Get model parameters string."""
        # This would be extracted from model metadata in a real implementation
        return f"temperature 0.7\ntop_p 0.9\ntop_k 40\nrepeat_penalty 1.1"

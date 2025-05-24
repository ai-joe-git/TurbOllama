"""
OpenAI Compatible API Routes
"""

import time
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the message")


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(512, ge=1, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None, description="User identifier")


class CompletionRequest(BaseModel):
    """OpenAI completion request."""
    model: str = Field(..., description="Model to use")
    prompt: str = Field(..., description="Prompt to complete")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(512, ge=1)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10)
    stream: Optional[bool] = Field(False)
    stop: Optional[List[str]] = Field(None)


class EmbeddingRequest(BaseModel):
    """OpenAI embedding request."""
    model: str = Field(..., description="Model to use for embeddings")
    input: List[str] = Field(..., description="Input texts to embed")
    user: Optional[str] = Field(None)


class OpenAIRoutes:
    """OpenAI compatible API routes."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup OpenAI compatible routes."""
        
        @self.router.get("/models")
        async def list_models():
            """List available models in OpenAI format."""
            models = []
            for model_name in self.model_manager.list_models():
                model_info = self.model_manager.get_model_info(model_name)
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(model_info.downloaded_at) if model_info and model_info.downloaded_at else int(time.time()),
                    "owned_by": "turbollama",
                    "permission": [],
                    "root": model_name,
                    "parent": None
                })
            
            return {
                "object": "list",
                "data": models
            }
        
        @self.router.post("/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI chat completions endpoint."""
            # Ensure model is loaded
            if self.model_manager.current_model_name != request.model:
                try:
                    await self.model_manager.load_model(request.model)
                except Exception as e:
                    raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
            
            if not self.model_manager.loaded_model:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Convert messages to internal format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Generate response
            if request.stream:
                return StreamingResponse(
                    self._stream_chat_completion(request, messages),
                    media_type="text/plain"
                )
            else:
                return await self._generate_chat_completion(request, messages)
        
        @self.router.post("/completions")
        async def completions(request: CompletionRequest):
            """OpenAI completions endpoint."""
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
                    self._stream_completion(request),
                    media_type="text/plain"
                )
            else:
                return await self._generate_completion(request)
        
        @self.router.post("/embeddings")
        async def embeddings(request: EmbeddingRequest):
            """OpenAI embeddings endpoint (placeholder)."""
            # Note: This would require embedding model support
            raise HTTPException(
                status_code=501, 
                detail="Embeddings not yet supported. Use a dedicated embedding model."
            )
    
    async def _generate_chat_completion(self, request: ChatCompletionRequest, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate non-streaming chat completion."""
        try:
            start_time = time.time()
            
            # Generate response
            response_text = await self.model_manager.chat(
                messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop or []
            )
            
            # Calculate tokens (rough estimate)
            prompt_tokens = sum(len(msg["content"].split()) for msg in messages)
            completion_tokens = len(response_text.split())
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                "id": f"chatcmpl-{uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(start_time),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _stream_chat_completion(self, request: ChatCompletionRequest, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion."""
        try:
            completion_id = f"chatcmpl-{uuid4().hex[:8]}"
            created = int(time.time())
            
            # Send initial chunk
            initial_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"
            
            # Stream response
            async for chunk in self.model_manager.loaded_model.chat_stream(
                messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p
            ):
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send final chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def _generate_completion(self, request: CompletionRequest) -> Dict[str, Any]:
        """Generate non-streaming completion."""
        try:
            start_time = time.time()
            
            response_text = await self.model_manager.generate(
                request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop or []
            )
            
            # Calculate tokens
            prompt_tokens = len(request.prompt.split())
            completion_tokens = len(response_text.split())
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                "id": f"cmpl-{uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(start_time),
                "model": request.model,
                "choices": [{
                    "text": response_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error in completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Generate streaming completion."""
        try:
            completion_id = f"cmpl-{uuid4().hex[:8]}"
            created = int(time.time())
            
            async for chunk in self.model_manager.loaded_model.generate_stream(
                request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p
            ):
                chunk_data = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "text": chunk,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Final chunk
            final_chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming completion: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

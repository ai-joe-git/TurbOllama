"""
LlamaWrapper - Direct interface to llama.cpp
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
import json
import time

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None
    LlamaGrammar = None

logger = logging.getLogger(__name__)


class LlamaWrapper:
    """Wrapper around llama-cpp-python for async operations."""
    
    def __init__(self, model_path: str, **kwargs):
        if Llama is None:
            raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        
        self.model_path = model_path
        self.llama_args = kwargs
        
        # Initialize llama.cpp model
        logger.info(f"Initializing llama.cpp with model: {model_path}")
        self.llama = Llama(model_path=model_path, **kwargs)
        
        # Performance tracking
        self.total_tokens_generated = 0
        self.total_time_spent = 0.0
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        start_time = time.time()
        
        # Set default parameters
        generation_kwargs = {
            'max_tokens': kwargs.get('max_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 40),
            'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
            'stop': kwargs.get('stop', []),
            'stream': False
        }
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: self.llama(prompt, **generation_kwargs)
        )
        
        # Extract text from response
        if isinstance(response, dict) and 'choices' in response:
            text = response['choices'][0]['text']
        else:
            text = str(response)
        
        # Update performance metrics
        end_time = time.time()
        elapsed = end_time - start_time
        token_count = len(text.split())  # Rough estimate
        
        self.total_tokens_generated += token_count
        self.total_time_spent += elapsed
        
        return text
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        generation_kwargs = {
            'max_tokens': kwargs.get('max_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 40),
            'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
            'stop': kwargs.get('stop', []),
            'stream': True
        }
        
        # Run streaming generation in thread
        loop = asyncio.get_event_loop()
        
        def _stream_generator():
            for chunk in self.llama(prompt, **generation_kwargs):
                if isinstance(chunk, dict) and 'choices' in chunk:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        return delta['content']
                    elif 'text' in chunk['choices'][0]:
                        return chunk['choices'][0]['text']
                return str(chunk)
        
        # Stream tokens
        stream = self.llama(prompt, **generation_kwargs)
        for chunk in stream:
            if isinstance(chunk, dict) and 'choices' in chunk:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    yield delta['content']
                elif 'text' in chunk['choices'][0]:
                    yield chunk['choices'][0]['text']
            else:
                yield str(chunk)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with conversation history."""
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        return await self.generate(prompt, **kwargs)
    
    async def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Chat with streaming response."""
        prompt = self._messages_to_prompt(messages)
        async for chunk in self.generate_stream(prompt, **kwargs):
            yield chunk
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt format."""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.total_time_spent > 0:
            avg_tokens_per_second = self.total_tokens_generated / self.total_time_spent
        else:
            avg_tokens_per_second = 0.0
        
        return {
            'total_tokens_generated': self.total_tokens_generated,
            'total_time_spent': self.total_time_spent,
            'avg_tokens_per_second': avg_tokens_per_second
        }

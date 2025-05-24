"""
Modern Gradio Interface for TurboLlama - ChatGPT/OpenWebUI Style
"""

import gradio as gr
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..core.config import Config
from ..core.model_manager import ModelManager
from ..utils.hardware import get_system_info

logger = logging.getLogger(__name__)


class GradioInterface:
    """Modern Gradio interface with ChatGPT-like experience."""
    
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.demo = None
        
        # Chat state
        self.current_model = None
        self.system_message = "You are a helpful AI assistant."
        self.conversation_history = []
        
        # Performance tracking
        self.performance_data = []
        
        self._setup_interface()
    
    def _setup_interface(self):
        """Setup the complete Gradio interface."""
        # Custom CSS for modern look
        custom_css = self._get_custom_css()
        
        # Create the interface
        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate"
            ),
            css=custom_css,
            title="TurboLlama - AI Chat Interface",
            head="<link rel='icon' href='data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><text y=\".9em\" font-size=\"90\">🚀</text></svg>'>"
        ) as demo:
            
            # Header
            with gr.Row(elem_id="header"):
                with gr.Column(scale=1):
                    gr.HTML("""
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 2em;">🚀</span>
                            <div>
                                <h1 style="margin: 0; color: #2563eb;">TurboLlama</h1>
                                <p style="margin: 0; color: #64748b; font-size: 0.9em;">Powered by llama.cpp</p>
                            </div>
                        </div>
                    """)
                
                with gr.Column(scale=1):
                    # Model selector and settings button
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=self._get_available_models(),
                            value=self.config.models.default,
                            label="Model",
                            elem_id="model-selector",
                            scale=3
                        )
                        settings_btn = gr.Button("⚙️", scale=1, elem_id="settings-btn")
                        performance_btn = gr.Button("📊", scale=1, elem_id="performance-btn")
            
            # Main chat interface
            with gr.Row(elem_id="main-content"):
                with gr.Column(scale=4):
                    # Chat messages
                    chatbot = gr.Chatbot(
                        value=[],
                        elem_id="chatbot",
                        height=600,
                        show_copy_button=True,
                        bubble_full_width=False,
                        avatar_images=("👤", "🤖"),
                        type="messages",
                        placeholder="""
                        <div style="text-align: center; padding: 40px;">
                            <h3>🚀 Welcome to TurboLlama!</h3>
                            <p>Your high-performance AI assistant powered by the latest llama.cpp</p>
                            <p style="color: #64748b;">Start a conversation by typing a message below</p>
                        </div>
                        """
                    )
                    
                    # Input area with multimodal support
                    with gr.Row(elem_id="input-row"):
                        chat_input = gr.MultimodalTextbox(
                            placeholder="Type your message here... (Supports text, images, and files)",
                            show_label=False,
                            elem_id="chat-input",
                            scale=6,
                            file_count="multiple",
                            sources=["upload", "microphone"] if self.config.interface.enable_voice else ["upload"]
                        )
                        
                        send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")
                        stop_btn = gr.Button("Stop", variant="stop", scale=1, visible=False, elem_id="stop-btn")
                
                # Sidebar for settings and info
                with gr.Column(scale=1, elem_id="sidebar", visible=False) as sidebar:
                    with gr.Accordion("Model Settings", open=True):
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                            label="Temperature", info="Controls randomness"
                        )
                        max_tokens = gr.Slider(
                            minimum=1, maximum=4096, value=512, step=1,
                            label="Max Tokens", info="Maximum response length"
                        )
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                            label="Top P", info="Nucleus sampling"
                        )
                        top_k = gr.Slider(
                            minimum=1, maximum=100, value=40, step=1,
                            label="Top K", info="Top-k sampling"
                        )
                    
                    with gr.Accordion("System Message", open=False):
                        system_msg = gr.Textbox(
                            value=self.system_message,
                            label="System Message",
                            placeholder="You are a helpful AI assistant...",
                            lines=3
                        )
                    
                    with gr.Accordion("Model Info", open=False):
                        model_info = gr.JSON(
                            value=self._get_model_info(),
                            label="Current Model"
                        )
                    
                    with gr.Accordion("Hardware Info", open=False):
                        hardware_info = gr.JSON(
                            value=self._get_hardware_info(),
                            label="System Information"
                        )
                    
                    # Action buttons
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                        export_btn = gr.Button("💾 Export", variant="secondary")
            
            # Performance modal
            with gr.Row(elem_id="performance-modal", visible=False) as performance_modal:
                with gr.Column():
                    gr.Markdown("## 📊 Performance Dashboard")
                    
                    with gr.Row():
                        with gr.Column():
                            tokens_per_second = gr.Number(
                                value=0, label="Tokens/Second", interactive=False
                            )
                            total_tokens = gr.Number(
                                value=0, label="Total Tokens", interactive=False
                            )
                        
                        with gr.Column():
                            avg_latency = gr.Number(
                                value=0, label="Avg Latency (s)", interactive=False
                            )
                            memory_usage = gr.Number(
                                value=0, label="Memory Usage (GB)", interactive=False
                            )
                    
                    performance_plot = gr.Plot(label="Performance Over Time")
                    close_performance_btn = gr.Button("Close", variant="secondary")
            
            # Example prompts
            with gr.Row(elem_id="examples"):
                gr.Examples(
                    examples=[
                        ["Hello! Can you help me write a Python function?"],
                        ["Explain quantum computing in simple terms"],
                        ["Write a creative story about a robot learning to paint"],
                        ["Help me debug this code: print('Hello World')"],
                        ["What are the latest developments in AI?"],
                    ],
                    inputs=chat_input,
                    label="Try these examples:"
                )
            
            # Event handlers
            self._setup_event_handlers(
                chatbot, chat_input, send_btn, stop_btn, clear_btn,
                model_dropdown, settings_btn, performance_btn, sidebar,
                performance_modal, close_performance_btn,
                temperature, max_tokens, top_p, top_k, system_msg,
                model_info, hardware_info, export_btn,
                tokens_per_second, total_tokens, avg_latency, memory_usage, performance_plot
            )
        
        self.demo = demo
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for modern styling."""
        return """
        /* Modern ChatGPT-like styling */
        #header {
            background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 20px;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }
        
        #chatbot {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background: #ffffff;
        }
        
        #chat-input {
            border-radius: 24px;
            border: 2px solid #e2e8f0;
            padding: 12px 20px;
        }
        
        #chat-input:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        #send-btn {
            border-radius: 20px;
            background: linear-gradient(45deg, #2563eb, #3b82f6);
            border: none;
            color: white;
            font-weight: 600;
        }
        
        #stop-btn {
            border-radius: 20px;
            background: linear-gradient(45deg, #dc2626, #ef4444);
            border: none;
            color: white;
            font-weight: 600;
        }
        
        #sidebar {
            background: #f8fafc;
            border-radius: 12px;
            padding: 20px;
            margin-left: 20px;
        }
        
        #model-selector {
            border-radius: 8px;
        }
        
        #settings-btn, #performance-btn {
            border-radius: 8px;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            color: #475569;
        }
        
        #settings-btn:hover, #performance-btn:hover {
            background: #e2e8f0;
        }
        
        #performance-modal {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
        }
        
        #examples {
            margin-top: 20px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 12px;
        }
        
        /* Message bubbles */
        .message.user {
            background: linear-gradient(45deg, #2563eb, #3b82f6);
            color: white;
            border-radius: 18px 18px 4px 18px;
        }
        
        .message.bot {
            background: #f1f5f9;
            color: #1e293b;
            border-radius: 18px 18px 18px 4px;
        }
        
        /* Animations */
        .fade-in {
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            #sidebar {
                display: none;
            }
            #main-content {
                flex-direction: column;
            }
        }
        """
    
    def _setup_event_handlers(self, *components):
        """Setup all event handlers for the interface."""
        (chatbot, chat_input, send_btn, stop_btn, clear_btn,
         model_dropdown, settings_btn, performance_btn, sidebar,
         performance_modal, close_performance_btn,
         temperature, max_tokens, top_p, top_k, system_msg,
         model_info, hardware_info, export_btn,
         tokens_per_second, total_tokens, avg_latency, memory_usage, performance_plot) = components
        
        # Chat functionality
        chat_msg = chat_input.submit(
            self._handle_user_message,
            [chatbot, chat_input, temperature, max_tokens, top_p, top_k, system_msg],
            [chatbot, chat_input, send_btn, stop_btn],
            queue=True
        )
        
        send_msg = send_btn.click(
            self._handle_user_message,
            [chatbot, chat_input, temperature, max_tokens, top_p, top_k, system_msg],
            [chatbot, chat_input, send_btn, stop_btn],
            queue=True
        )
        
        # Bot response with streaming
        bot_msg = chat_msg.then(
            self._generate_bot_response,
            [chatbot, temperature, max_tokens, top_p, top_k, system_msg],
            [chatbot, tokens_per_second, total_tokens, avg_latency],
            queue=True
        )
        
        # Re-enable input after response
        bot_msg.then(
            lambda: (gr.update(interactive=True), gr.update(visible=True), gr.update(visible=False)),
            None,
            [chat_input, send_btn, stop_btn]
        )
        
        # Model selection
        model_dropdown.change(
            self._change_model,
            [model_dropdown],
            [model_info]
        )
        
        # Settings toggle
        settings_btn.click(
            lambda visible: gr.update(visible=not visible),
            [sidebar],
            [sidebar]
        )
        
        # Performance dashboard
        performance_btn.click(
            self._show_performance_dashboard,
            None,
            [performance_modal, performance_plot]
        )
        
        close_performance_btn.click(
            lambda: gr.update(visible=False),
            None,
            [performance_modal]
        )
        
        # Clear chat
        clear_btn.click(
            self._clear_chat,
            None,
            [chatbot]
        )
        
        # Export conversation
        export_btn.click(
            self._export_conversation,
            [chatbot],
            None
        )
        
        # System message update
        system_msg.change(
            self._update_system_message,
            [system_msg],
            None
        )
        
        # Like/dislike functionality
        chatbot.like(
            self._handle_feedback,
            None,
            None
        )
    
    async def _handle_user_message(self, history, message, temperature, max_tokens, top_p, top_k, system_msg):
        """Handle user message input."""
        if not message or (isinstance(message, dict) and not message.get("text")):
            return history, gr.update(value=""), gr.update(), gr.update()
        
        # Extract text and files from message
        if isinstance(message, dict):
            user_text = message.get("text", "")
            files = message.get("files", [])
        else:
            user_text = str(message)
            files = []
        
        # Add user message to history
        if files:
            # Handle multimodal input
            file_info = []
            for file_path in files:
                file_info.append(f"📎 {Path(file_path).name}")
            
            user_content = f"{user_text}\n\n" + "\n".join(file_info)
        else:
            user_content = user_text
        
        history.append({"role": "user", "content": user_content})
        
        return (
            history,
            gr.update(value="", interactive=False),
            gr.update(visible=False),
            gr.update(visible=True)
        )
    
    async def _generate_bot_response(self, history, temperature, max_tokens, top_p, top_k, system_msg):
        """Generate bot response with streaming."""
        if not history:
            return history, 0, 0, 0
        
        start_time = time.time()
        
        # Prepare messages for the model
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(history)
        
        try:
            # Generate response with streaming
            response_text = ""
            token_count = 0
            
            # Add empty assistant message for streaming
            history.append({"role": "assistant", "content": ""})
            
            # Stream the response
            async for chunk in self.model_manager.loaded_model.chat_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            ):
                response_text += chunk
                history[-1]["content"] = response_text
                token_count += 1
                
                # Yield intermediate results for real-time updates
                yield history, 0, 0, 0
            
            # Calculate final performance metrics
            end_time = time.time()
            elapsed = end_time - start_time
            tokens_per_second = token_count / elapsed if elapsed > 0 else 0
            
            # Update performance tracking
            self.performance_data.append({
                'timestamp': time.time(),
                'tokens_per_second': tokens_per_second,
                'latency': elapsed,
                'token_count': token_count
            })
            
            # Keep only last 100 data points
            if len(self.performance_data) > 100:
                self.performance_data = self.performance_data[-100:]
            
            yield history, tokens_per_second, token_count, elapsed
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            history[-1]["content"] = f"❌ Error: {str(e)}"
            yield history, 0, 0, 0
    
    def _change_model(self, model_name):
        """Change the current model."""
        try:
            asyncio.create_task(self.model_manager.load_model(model_name))
            self.current_model = model_name
            return self._get_model_info()
        except Exception as e:
            logger.error(f"Error changing model: {e}")
            return {"error": str(e)}
    
    def _clear_chat(self):
        """Clear the chat history."""
        self.conversation_history = []
        return []
    
    def _update_system_message(self, system_msg):
        """Update the system message."""
        self.system_message = system_msg
    
    def _handle_feedback(self, data):
        """Handle like/dislike feedback."""
        logger.info(f"Feedback received: {data}")
        # Here you could implement feedback storage/analytics
    
    def _export_conversation(self, history):
        """Export conversation to file."""
        if not history:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"turbollama_conversation_{timestamp}.json"
        
        export_data = {
            'timestamp': timestamp,
            'model': self.current_model,
            'system_message': self.system_message,
            'conversation': history
        }
        
        # In a real implementation, this would trigger a download
        logger.info(f"Exporting conversation to {filename}")
    
    def _show_performance_dashboard(self):
        """Show performance dashboard with real-time metrics."""
        if not self.performance_data:
            plot_data = None
        else:
            # Create performance plot
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in self.performance_data]
            tokens_per_sec = [d['tokens_per_second'] for d in self.performance_data]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(timestamps, tokens_per_sec, 'b-', linewidth=2, label='Tokens/Second')
            ax.set_xlabel('Time')
            ax.set_ylabel('Tokens per Second')
            ax.set_title('TurboLlama Performance Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_data = fig
        
        return gr.update(visible=True), plot_data
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            return self.model_manager.list_models()
        except:
            return ["llama2:7b"]  # Default fallback
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        if self.current_model and self.current_model in self.model_manager.model_registry:
            model_info = self.model_manager.model_registry[self.current_model]
            return {
                "name": model_info.name,
                "size": f"{model_info.size / (1024**3):.1f} GB",
                "format": model_info.format,
                "quantization": model_info.quantization,
                "capabilities": model_info.capabilities or [],
                "context_length": model_info.context_length or "Unknown"
            }
        return {"status": "No model loaded"}
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        try:
            return self.model_manager.gpu_manager.get_hardware_info()
        except:
            return {"status": "Hardware info unavailable"}
    
    async def launch(self, port: int = 7860, share: bool = False):
        """Launch the Gradio interface."""
        if self.demo is None:
            raise RuntimeError("Interface not initialized")
        
        logger.info(f"Launching Gradio interface on port {port}")
        
        # Launch with custom configuration
        self.demo.launch(
            server_port=port,
            share=share,
            server_name="0.0.0.0",
            show_error=True,
            quiet=False,
            favicon_path=None,
            ssl_verify=False,
            app_kwargs={
                "docs_url": "/docs",
                "redoc_url": "/redoc"
            }
        )

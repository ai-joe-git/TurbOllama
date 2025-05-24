#!/bin/bash

# TurboLlama Start Script for Linux/macOS
# Usage: ./start.sh [options]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_MODEL="llama2:7b"
DEFAULT_API_PORT=11434
DEFAULT_GUI_PORT=7860
DEFAULT_HOST="0.0.0.0"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "🚀 TurboLlama Start Script"
    echo "========================="
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -m, --model MODEL        Model to load (default: $DEFAULT_MODEL)"
    echo "  -p, --port PORT          API port (default: $DEFAULT_API_PORT)"
    echo "  -g, --gui-port PORT      GUI port (default: $DEFAULT_GUI_PORT)"
    echo "  -h, --host HOST          Host to bind to (default: $DEFAULT_HOST)"
    echo "  --no-gui                 Start without GUI"
    echo "  --gpu-backend BACKEND    GPU backend (cuda, vulkan, rocm, xpu, cpu)"
    echo "  --help                   Show this help message"
    echo
    echo "Examples:"
    echo "  $0                                    # Start with default settings"
    echo "  $0 -m mistral:7b --gpu-backend cuda  # Start with Mistral and CUDA"
    echo "  $0 --no-gui -p 8080                  # Start API only on port 8080"
}

# Parse command line arguments
MODEL="$DEFAULT_MODEL"
API_PORT="$DEFAULT_API_PORT"
GUI_PORT="$DEFAULT_GUI_PORT"
HOST="$DEFAULT_HOST"
ENABLE_GUI=true
GPU_BACKEND=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--port)
            API_PORT="$2"
            shift 2
            ;;
        -g|--gui-port)
            GUI_PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        --no-gui)
            ENABLE_GUI=false
            shift
            ;;
        --gpu-backend)
            GPU_BACKEND="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Check if virtual environment exists
check_venv() {
    if [ -d "venv" ]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
    else
        print_warning "Virtual environment not found. Using system Python."
    fi
}

# Check if TurboLlama is installed
check_installation() {
    if ! command -v turbollama &> /dev/null; then
        print_warning "TurboLlama not found. Please run install.sh first."
        exit 1
    fi
}

# Start TurboLlama
start_turbollama() {
    print_status "Starting TurboLlama..."
    echo
    print_status "Configuration:"
    echo "  Model: $MODEL"
    echo "  API Port: $API_PORT"
    if [ "$ENABLE_GUI" = true ]; then
        echo "  GUI Port: $GUI_PORT"
    fi
    echo "  Host: $HOST"
    if [ -n "$GPU_BACKEND" ]; then
        echo "  GPU Backend: $GPU_BACKEND"
    fi
    echo
    
    # Build command
    CMD="turbollama serve --model $MODEL --host $HOST --port $API_PORT"
    
    if [ "$ENABLE_GUI" = true ]; then
        CMD="$CMD --gui --gui-port $GUI_PORT"
    fi
    
    if [ -n "$GPU_BACKEND" ]; then
        CMD="$CMD --backend $GPU_BACKEND"
    fi
    
    CMD="$CMD $EXTRA_ARGS"
    
    print_status "Executing: $CMD"
    echo
    
    # Start TurboLlama
    eval $CMD
}

# Main function
main() {
    echo "🚀 TurboLlama Startup"
    echo "===================="
    echo
    
    check_venv
    check_installation
    start_turbollama
}

# Handle Ctrl+C gracefully
trap 'echo; print_status "Shutting down TurboLlama..."; exit 0' INT

# Run main function
main "$@"

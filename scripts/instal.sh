#!/bin/bash

# TurboLlama Installation Script for Linux/macOS
# Usage: ./install.sh

set -e

echo "🚀 TurboLlama Installation Script"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION found"
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
    else
        print_error "pip is not installed. Please install pip."
        exit 1
    fi
    
    print_success "pip found"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y build-essential cmake pkg-config libopenblas-dev
        elif command -v yum &> /dev/null; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y cmake openblas-devel
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm base-devel cmake openblas
        else
            print_warning "Unknown Linux distribution. Please install build tools manually."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install cmake
        else
            print_warning "Homebrew not found. Please install Xcode Command Line Tools manually."
        fi
    fi
    
    print_success "System dependencies installed"
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Install TurboLlama
install_turbollama() {
    print_status "Installing TurboLlama..."
    
    # Install from PyPI (when available)
    if pip install turbollama 2>/dev/null; then
        print_success "TurboLlama installed from PyPI"
    else
        print_warning "PyPI installation failed. Installing from source..."
        
        # Install from current directory
        pip install -e .
        print_success "TurboLlama installed from source"
    fi
}

# Install optional dependencies
install_optional_deps() {
    print_status "Installing optional dependencies..."
    
    # GPU support
    read -p "Do you want to install GPU support? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # CUDA support
        read -p "Install NVIDIA CUDA support? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        fi
        
        # Intel GPU support
        read -p "Install Intel GPU support? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install intel-extension-for-pytorch
        fi
    fi
    
    # Development dependencies
    read -p "Do you want to install development dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -e ".[dev]"
        print_success "Development dependencies installed"
    fi
}

# Create desktop shortcut
create_shortcut() {
    print_status "Creating desktop shortcut..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux desktop shortcut
        cat > ~/Desktop/TurboLlama.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=TurboLlama
Comment=AI Chat Interface
Exec=$(pwd)/venv/bin/turbollama serve --gui
Icon=applications-internet
Terminal=false
Categories=Development;
EOF
        chmod +x ~/Desktop/TurboLlama.desktop
        print_success "Desktop shortcut created"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS app bundle (simplified)
        print_warning "macOS shortcut creation not implemented. Use: turbollama serve --gui"
    fi
}

# Main installation process
main() {
    echo
    print_status "Starting TurboLlama installation..."
    echo
    
    check_python
    check_pip
    install_system_deps
    create_venv
    install_turbollama
    install_optional_deps
    
    # Test installation
    print_status "Testing installation..."
    if turbollama --help > /dev/null 2>&1; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
    
    create_shortcut
    
    echo
    print_success "🎉 TurboLlama installation completed successfully!"
    echo
    echo "To get started:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Start TurboLlama: turbollama serve --model llama2:7b --gui"
    echo "3. Open your browser at: http://localhost:7860"
    echo
    echo "For more information, visit: https://github.com/ai-joe-git/TurboLlama"
}

# Run main function
main "$@"

@echo off
REM TurboLlama Installation Script for Windows
REM Usage: install.bat

setlocal enabledelayedexpansion

echo 🚀 TurboLlama Installation Script for Windows
echo =============================================

REM Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found

REM Check if pip is installed
echo [INFO] Checking pip installation...
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not installed.
    echo Please install pip or reinstall Python with pip included.
    pause
    exit /b 1
)
echo [SUCCESS] pip found

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist venv (
    echo [WARNING] Virtual environment already exists. Removing...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install TurboLlama
echo [INFO] Installing TurboLlama...
pip install turbollama
if errorlevel 1 (
    echo [WARNING] PyPI installation failed. Installing from source...
    pip install -e .
    if errorlevel 1 (
        echo [ERROR] Installation failed.
        pause
        exit /b 1
    )
)
echo [SUCCESS] TurboLlama installed

REM Install optional dependencies
set /p GPU_SUPPORT="Do you want to install GPU support? (y/N): "
if /i "%GPU_SUPPORT%"=="y" (
    set /p CUDA_SUPPORT="Install NVIDIA CUDA support? (y/N): "
    if /i "!CUDA_SUPPORT!"=="y" (
        echo [INFO] Installing CUDA support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
    
    set /p INTEL_SUPPORT="Install Intel GPU support? (y/N): "
    if /i "!INTEL_SUPPORT!"=="y" (
        echo [INFO] Installing Intel GPU support...
        pip install intel-extension-for-pytorch
    )
)

set /p DEV_DEPS="Do you want to install development dependencies? (y/N): "
if /i "%DEV_DEPS%"=="y" (
    echo [INFO] Installing development dependencies...
    pip install -e ".[dev]"
    echo [SUCCESS] Development dependencies installed
)

REM Test installation
echo [INFO] Testing installation...
turbollama --help >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Installation test failed.
    pause
    exit /b 1
)
echo [SUCCESS] Installation test passed

REM Create desktop shortcut
echo [INFO] Creating desktop shortcut...
set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT=%DESKTOP%\TurboLlama.lnk
set TARGET=%CD%\venv\Scripts\turbollama.exe
set ARGS=serve --gui

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT%'); $Shortcut.TargetPath = '%TARGET%'; $Shortcut.Arguments = '%ARGS%'; $Shortcut.WorkingDirectory = '%CD%'; $Shortcut.IconLocation = '%TARGET%'; $Shortcut.Save()"

if exist "%SHORTCUT%" (
    echo [SUCCESS] Desktop shortcut created
) else (
    echo [WARNING] Failed to create desktop shortcut
)

echo.
echo [SUCCESS] 🎉 TurboLlama installation completed successfully!
echo.
echo To get started:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Start Tur

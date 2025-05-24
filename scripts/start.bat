@echo off
REM TurboLlama Start Script for Windows
REM Usage: start.bat [options]

setlocal enabledelayedexpansion

REM Default configuration
set DEFAULT_MODEL=llama2:7b
set DEFAULT_API_PORT=11434
set DEFAULT_GUI_PORT=7860
set DEFAULT_HOST=0.0.0.0

set MODEL=%DEFAULT_MODEL%
set API_PORT=%DEFAULT_API_PORT%
set GUI_PORT=%DEFAULT_GUI_PORT%
set HOST=%DEFAULT_HOST%
set ENABLE_GUI=true
set GPU_BACKEND=
set EXTRA_ARGS=

echo 🚀 TurboLlama Start Script for Windows
echo =====================================

REM Parse command line arguments (simplified)
:parse_args
if "%~1"=="" goto :start_setup
if "%~1"=="-m" (
    set MODEL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-p" (
    set API_PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set API_PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-g" (
    set GUI_PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--gui-port" (
    set GUI_PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-gui" (
    set ENABLE_GUI=false
    shift
    goto :parse_args
)
if "%~1"=="--gpu-backend" (
    set GPU_BACKEND=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    goto :show_usage
)
set EXTRA_ARGS=%EXTRA_ARGS% %~1
shift
goto :parse_args

:show_usage
echo Usage: %0 [OPTIONS]
echo.
echo Options:
echo   -m, --model MODEL        Model to load (default: %DEFAULT_MODEL%)
echo   -p, --port PORT          API port (default: %DEFAULT_API_PORT%)
echo   -g, --gui-port PORT      GUI port (default: %DEFAULT_GUI_PORT%)
echo   --no-gui                 Start without GUI
echo   --gpu-backend BACKEND    GPU backend (cuda, vulkan, rocm, xpu, cpu)
echo   --help                   Show this help message
echo.
echo Examples:
echo   %0                                    # Start with default settings
echo   %0 -m mistral:7b --gpu-backend cuda  # Start with Mistral and CUDA
echo   %0 --no-gui -p 8080                  # Start API only on port 8080
pause
exit /b 0

:start_setup
REM Check if virtual environment exists
if exist venv (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found. Using system Python.
)

REM Check if TurboLlama is installed
turbollama --help >nul 2>&1
if errorlevel 1 (
    echo [ERROR] TurboLlama not found. Please run install.bat first.
    pause
    exit /b 1
)

REM Start TurboLlama
echo [INFO] Starting TurboLlama...
echo.
echo [INFO] Configuration:
echo   Model: %MODEL%
echo   API Port: %API_PORT%
if "%ENABLE_GUI%"=="true" (
    echo   GUI Port: %GUI_PORT%
)
echo   Host: %HOST%
if not "%GPU_BACKEND%"=="" (
    echo   GPU Backend: %GPU_BACKEND%
)
echo.

REM Build command
set CMD=turbollama serve --model %MODEL% --host %HOST% --port %API_PORT%

if "%ENABLE_GUI%"=="true" (
    set CMD=!CMD! --gui --gui-port %GUI_PORT%
)

if not "%GPU_BACKEND%"=="" (
    set CMD=!CMD! --backend %GPU_BACKEND%
)

set CMD=!CMD! %EXTRA_ARGS%

echo [INFO] Executing: !CMD!
echo.

REM Start TurboLlama
!CMD!

pause

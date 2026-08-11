@echo off
echo ============================================
echo   YOLO Training Platform - Startup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if dependencies are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo [INFO] Starting YOLO Training Platform...
echo.

python main.py

pause

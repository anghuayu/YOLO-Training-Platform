#!/bin/bash
echo "============================================"
echo "  YOLO Training Platform - Startup"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "[INFO] Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
fi

echo ""
echo "[INFO] Starting YOLO Training Platform..."
echo ""

python3 main.py

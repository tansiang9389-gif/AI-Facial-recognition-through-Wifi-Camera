@echo off
title Coconut Maturity Classification System
echo ============================================
echo  Coconut Maturity Classification System
echo  Classes: Premature | Mature | Potential
echo  Powered by Roboflow + YOLOv8
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

:: Create virtual environment if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

:: Create directories
if not exist "models" mkdir models
if not exist "detections" mkdir detections
if not exist "reports" mkdir reports

:: Start the application
echo.
echo Starting Coconut Maturity Classification System...
echo Inference: Roboflow Hosted API (serverless)
echo Dashboard: http://localhost:5001
echo.
python app.py

pause

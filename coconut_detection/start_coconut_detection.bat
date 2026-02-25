@echo off
echo ============================================================
echo   Coconut Detection System — Startup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b
)

:: Create virtual environment if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

:: Check if model exists
if not exist "models\coconut_best.pt" (
    echo.
    echo [INFO] No trained model found. Running model download...
    python download_model.py
)

:: Launch
echo.
echo Starting Coconut Detection Dashboard...
echo Open http://localhost:5001 in your browser
echo.
python app.py

pause

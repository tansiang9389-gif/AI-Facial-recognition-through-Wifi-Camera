@echo off
echo ============================================
echo   CCTV Face Tracking - Web Dashboard
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b 1
)

REM Activate virtual environment
if not exist "venv" (
    echo Setting up virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    pip install flask
    echo.
    echo Setup complete!
    echo.
) else (
    call venv\Scripts\activate.bat
)

echo Starting web dashboard...
echo Open http://localhost:5000 in your browser
echo.
python app.py

pause

@echo off
chcp 65001 > nul
echo ========================================
echo    CL EVA02 ONNX Tagger Setup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not detected. Please install Python 3.8 or higher.
    echo Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Display Python version
echo Detected Python version:
python --version
echo.

REM Check if virtual environment exists, if not create it
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip using the correct method
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies with retry mechanism
echo Installing dependencies...
:retry_install
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo Installation failed, retrying in 5 seconds...
    timeout /t 5 /nobreak >nul
    goto retry_install
)
echo.

REM Check GPU support
echo Checking GPU support...
python -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"
echo.

REM Launch the application
echo Starting application...
echo The application will run at http://localhost:7870
echo Press Ctrl+C to stop the server
echo.
python app.py

pause

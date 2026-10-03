@echo off
REM One-time setup for Windows. Double-click this file.
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not on PATH.
    echo Install Python 3.11+ from https://www.python.org/downloads/ and tick "Add Python to PATH".
    pause
    exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo Installing dependencies (this takes a few minutes)...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Installation failed. Check your internet connection and retry.
    pause
    exit /b 1
)

if not exist .env (
    copy .env.example .env >nul
    echo.
    echo A .env file was created. Paste your OPENAI_API_KEY in it, save, and close Notepad.
    notepad .env
)

echo.
echo Setup complete. Double-click run.bat to start the chatbot.
pause

@echo off
REM Start the chatbot. Double-click this file.
cd /d "%~dp0"
if not exist venv (
    echo Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
streamlit run app.py
pause

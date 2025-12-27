@echo off
title Smart Infant Diagnosis Launcher
color 0a

echo =====================================================
echo     SMART INFANT DIAGNOSIS PROJECT - AUTO LAUNCHER
echo =====================================================
echo.

:: Navigate to project directory
cd /d "C:\Users\kisho\smart_infant_diagnosis"

:: Check for Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

:: Install dependencies (safe to run multiple times)
echo Installing required Python packages...
pip install tensorflow librosa scikit-learn matplotlib seaborn pandas numpy soundfile tqdm streamlit joblib >nul

:: Launch Streamlit app
echo.
echo Launching the Smart Infant Diagnosis web app...
cd app
start "" streamlit run streamlit_app.py

echo.
echo Web app is starting... It will open in your browser shortly.
echo Press CTRL + C in the terminal to stop the app when done.
pause

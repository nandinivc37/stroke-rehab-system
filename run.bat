@echo off
cd /d "%~dp0"
echo Starting Stroke Rehab System...
"%~dp0.venv\Scripts\python.exe" -m src.ui.app
echo Application exited with code %ERRORLEVEL%
pause

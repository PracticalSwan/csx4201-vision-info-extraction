@echo off
setlocal
cd /d "%~dp0"
set "APP_PYTHON=%~dp0.runtime\app\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
  echo OCR Model is not set up yet.
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
"%APP_PYTHON%" "%~dp0app.py"

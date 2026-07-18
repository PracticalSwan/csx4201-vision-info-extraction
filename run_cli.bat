@echo off
setlocal
cd /d "%~dp0"
set "APP_PYTHON=%~dp0.runtime\app\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
  echo OCR Model is not set up yet. Run setup_windows.bat first.
  exit /b 1
)
if "%~1"=="" (
  echo Usage: run_cli.bat "C:\path\to\document.pdf"
  exit /b 2
)
"%APP_PYTHON%" "%~dp0run_ocr.py" %*

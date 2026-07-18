@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" -Device cpu
if errorlevel 1 (
  echo.
  echo Setup failed. Review the message above.
  pause
  exit /b 1
)
echo.
echo Setup complete. Run launch_windows.bat.
pause

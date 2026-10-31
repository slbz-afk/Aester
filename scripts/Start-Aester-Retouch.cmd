@echo off
setlocal
set "AESTER_DEV_ROOT=%~dp0.."
if not exist "%AESTER_DEV_ROOT%\.venv\Scripts\python.exe" (
  echo This development checkout requires its existing .venv Python environment.
  pause
  exit /b 1
)
"%AESTER_DEV_ROOT%\.venv\Scripts\python.exe" -X utf8 "%~dp0launch_retouch.py" gui %*
if errorlevel 1 pause

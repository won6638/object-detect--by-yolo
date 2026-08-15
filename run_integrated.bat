@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_integrated.py %*
) else (
  python run_integrated.py %*
)

endlocal

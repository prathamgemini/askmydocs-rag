@echo off
setlocal
echo ===================================================
echo Starting AskMyDocs Backend...
echo Host: http://localhost:8000
echo ===================================================

rem Prefer a project virtual environment if one exists, else use python on PATH
set "PYTHON_EXE=python"
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
echo Python: %PYTHON_EXE%

cd /d "%~dp0app"
"%PYTHON_EXE%" -m uvicorn main:app --reload --port 8000
pause

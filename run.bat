@echo off
REM ---------------------------------------------------------------
REM  AI Companion launcher
REM  Double-click this file to start the app.
REM  Close the app window (or press Ctrl+C here) to stop it.
REM ---------------------------------------------------------------

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: No virtual environment found at .venv
    echo Expected: %~dp0.venv\Scripts\python.exe
    echo.
    echo Create it with:
    echo     py -3.12 -m venv .venv
    echo.
    pause
    exit /b 1
)

echo Starting AI Companion...
echo Close the app window to quit. This terminal will close with it.
echo.

".venv\Scripts\python.exe" -m ai_companion.main

echo.
echo AI Companion has exited.
timeout /t 3 >nul

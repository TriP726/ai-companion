@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================
echo   AI COMPANION - APPLY PATCH
echo ==============================================================
echo.
echo Running from: %CD%
echo.

if not exist "patches\ui\theme.py" (
    echo ERROR: cannot find "patches\ui\theme.py".
    echo.
    echo This .bat must sit in the SAME folder as the "patches" folder,
    echo and that folder must be your project root. Expected layout:
    echo.
    echo     C:\dev\ai_companion\apply_patch.bat      ^<- this file
    echo     C:\dev\ai_companion\patches\
    echo     C:\dev\ai_companion\ai_companion\
    echo.
    echo Contents of the current folder:
    dir /b
    echo.
    pause
    exit /b 1
)

if not exist "ai_companion\ui" (
    echo ERROR: no "ai_companion\ui" folder here.
    echo This does not look like the project root.
    echo Move apply_patch.bat AND the patches folder into C:\dev\ai_companion
    echo.
    pause
    exit /b 1
)

echo Backing up current files to patches_backup\ ...
if not exist "patches_backup" mkdir "patches_backup"
xcopy /Y /Q /I /E "ai_companion" "patches_backup\ai_companion" >nul 2>&1
xcopy /Y /Q /I /E "tests" "patches_backup\tests" >nul 2>&1
echo   done.
echo.

echo Copying UI files...
copy /Y "patches\ui\*.py" "ai_companion\ui\" >nul || goto :failed
echo Copying models...
copy /Y "patches\models\*.py" "ai_companion\models\" >nul || goto :failed
echo Copying services...
copy /Y "patches\services\*.py" "ai_companion\services\" >nul || goto :failed
echo Copying infrastructure...
copy /Y "patches\infrastructure\*.py" "ai_companion\infrastructure\" >nul || goto :failed
echo Copying entry point...
copy /Y "patches\main.py" "ai_companion\" >nul || goto :failed
echo Copying tests...
copy /Y "patches\tests\*.py" "tests\" >nul || goto :failed
echo Copying diagnostic...
copy /Y "patches\diagnose_model.py" "." >nul || goto :failed

echo.
echo All files copied. Running the test suite...
echo --------------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m pytest tests/ -q
) else (
    echo WARNING: .venv not found, skipping tests.
)
echo --------------------------------------------------------------
echo.
echo Expected: 137 passed
echo.
echo If that looks right, start the app with:  run.bat
echo A backup of your previous files is in:   patches_backup\
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: a copy step failed. Nothing further was applied.
echo Your original files are backed up in patches_backup\
pause
exit /b 1

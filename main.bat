@echo off
TITLE Art Center Client

echo Activating Python virtual environment...
CALL .\venv\Scripts\activate.bat

IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Failed to activate Python virtual environment.
    echo Please make sure the 'venv' folder exists in the current directory.
    pause
    exit /b
)

echo Starting the Python indexing script...
echo.
python main.py D:\Resources\

echo.
echo -----------------------------------------------------
echo Script finished. You can close this window now.
echo -----------------------------------------------------

pause

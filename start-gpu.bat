@echo off
setlocal
cd /d "%~dp0"

echo [BackRemove] GPU-Umgebung wird eingerichtet und geprueft...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-gpu.ps1"
if errorlevel 1 (
    echo.
    echo [BackRemove] GPU-Setup fehlgeschlagen.
    pause
    exit /b 1
)

set "INFERENCE_DEVICE=cuda"
echo.
echo [BackRemove] Starte API mit CUDA auf http://localhost:8080
echo [BackRemove] Beenden mit Strg+C.
echo.

"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8080
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [BackRemove] Server wurde mit Fehlercode %EXIT_CODE% beendet.
    pause
)

exit /b %EXIT_CODE%

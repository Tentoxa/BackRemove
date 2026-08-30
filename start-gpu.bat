@echo off
setlocal
cd /d "%~dp0"
set "API_KEY="
if not exist "%~dp0.env" (
    echo [BackRemove] Fehler: .env fehlt. API wird nicht ungeschuetzt gestartet.
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
    if /i "%%A"=="API_KEY" set "API_KEY=%%B"
)

if not defined API_KEY (
    echo [BackRemove] Fehler: API_KEY fehlt oder ist leer. API wird nicht ungeschuetzt gestartet.
    exit /b 1
)

echo [BackRemove] GPU-Umgebung wird eingerichtet und geprueft...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-gpu.ps1"
if errorlevel 1 (
    echo.
    echo [BackRemove] GPU-Setup fehlgeschlagen.
    exit /b 1
)

set "INFERENCE_DEVICE=cuda"
set "QUALITY_MODEL_ENABLED=1"
echo.
echo [BackRemove] Starte API mit withoutBG ^(fast^) und BiRefNet ^(quality^) auf http://localhost:8080
echo [BackRemove] Beenden mit Strg+C.
echo.

"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --no-proxy-headers --env-file "%~dp0.env"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [BackRemove] Server wurde mit Fehlercode %EXIT_CODE% beendet.
)

exit /b %EXIT_CODE%

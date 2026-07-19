$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python 3.12 virtual environment."
    }
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }

& $python -m pip install -r requirements-gpu.txt
if ($LASTEXITCODE -ne 0) { throw "Could not install GPU requirements." }

# withoutbg depends on the CPU distribution, which shares its Python package
# with onnxruntime-gpu. On a fresh install, activate the GPU wheel last.
& $python -c "import onnxruntime as ort; providers = ort.get_available_providers(); raise SystemExit(0 if ort.__version__ == '1.18.0' and 'CUDAExecutionProvider' in providers else 1)"
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.18.0
    if ($LASTEXITCODE -ne 0) { throw "Could not activate ONNX Runtime GPU." }
}

& $python -c "import onnxruntime as ort; providers = ort.get_available_providers(); print('ONNX Runtime providers:', providers); raise SystemExit(0 if ort.__version__ == '1.18.0' and 'CUDAExecutionProvider' in providers else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDAExecutionProvider is unavailable. Check the NVIDIA driver."
}

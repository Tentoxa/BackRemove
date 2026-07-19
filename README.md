# BackRemove

Background removal API using the withoutBG Open Weights model.

## Native setup (Windows + NVIDIA GPU)

Requires Python 3.12 and a current NVIDIA driver. Double-click
`start-gpu.bat`; it installs/checks the Pascal-compatible CUDA 11 and cuDNN 8
runtime in `.venv` and then starts the API. A system-wide CUDA toolkit is not
required.

```bat
start-gpu.bat
```

The first server start downloads the model (~495 MB) and creates a CUDA-compatible
copy in the model cache. Confirm that inference is really using the GPU:

```powershell
Invoke-RestMethod http://localhost:8080/health
# status: ok
# inference_provider: CUDAExecutionProvider
```

`INFERENCE_DEVICE` accepts `cuda`, `cpu`, or `auto` (the default). Explicit
`cuda` fails at startup instead of silently running on the CPU.

## Native CPU setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:INFERENCE_DEVICE = "cpu"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Docker

```bash
docker compose up --build
```

## Usage

```bash
curl -X POST http://localhost:8080/remove-bg \
  -F "file=@photo.jpg" \
  --output no-bg.png
```

Supported formats: JPEG, PNG, WebP, GIF, AVIF, SVG.

## Auth

Optional. Set the `API_KEY` environment variable to require an `X-API-Key` header on requests. Unset = open access.

```bash
API_KEY=my-secret uvicorn app.main:app --host 0.0.0.0 --port 8080
curl -H "X-API-Key: my-secret" -F "file=@photo.jpg" http://localhost:8080/remove-bg --output no-bg.png
```

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/remove-bg` | Optional | Remove background, returns PNG |

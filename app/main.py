import io
import os
import time
import logging
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from PIL import Image

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

from app.model import load_model, get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",") if os.environ.get("CORS_ORIGINS") else []
INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "120"))

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

MAX_FAILED_ATTEMPTS = 5
BLOCK_DURATION = 900
ATTEMPT_WINDOW = 60

_failed_attempts: dict[str, list[float]] = {}
_blocked_ips: dict[str, float] = {}

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/avif",
    "image/svg+xml",
}
MAX_FILE_SIZE = 20 * 1024 * 1024


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_blocked(ip: str) -> bool:
    if ip in _blocked_ips:
        if time.monotonic() - _blocked_ips[ip] < BLOCK_DURATION:
            return True
        del _blocked_ips[ip]
        _failed_attempts.pop(ip, None)
    return False


def _record_failure(ip: str):
    now = time.monotonic()
    attempts = _failed_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < ATTEMPT_WINDOW]
    attempts.append(now)
    _failed_attempts[ip] = attempts

    if len(attempts) >= MAX_FAILED_ATTEMPTS:
        _blocked_ips[ip] = now
        logger.warning(f"Blocked {ip} for {BLOCK_DURATION}s after {len(attempts)} failed auth attempts")


async def verify_api_key(request: Request, key: str = Security(api_key_header)):
    if API_KEY is None:
        return

    ip = _get_client_ip(request)

    if _is_blocked(ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")

    if key != API_KEY:
        _record_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="BackRemove", version="1.0.0", lifespan=lifespan)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


def _process_image(image_bytes: bytes, content_type: str) -> bytes:
    model = get_model()

    if content_type == "image/svg+xml":
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=image_bytes)
        input_image = Image.open(io.BytesIO(png_data)).convert("RGB")
    else:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = model.remove_background(input_image)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/remove-bg", dependencies=[Security(verify_api_key)])
async def remove_bg(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 256)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Max 20 MB.")
        chunks.append(chunk)
    image_bytes = b"".join(chunks)

    try:
        with anyio.fail_after(INFERENCE_TIMEOUT):
            result_bytes = await anyio.to_thread.run_sync(
                lambda: _process_image(image_bytes, file.content_type or "")
            )
    except TimeoutError:
        logger.error(f"Inference timed out after {INFERENCE_TIMEOUT}s")
        raise HTTPException(status_code=504, detail="Processing timed out.")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Background removal failed.")

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=no-bg.png"},
    )

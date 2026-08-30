import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from ipaddress import ip_address
from typing import Annotated

import anyio
from anyio.abc import TaskStatus
from fastapi import FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import APIKeyHeader

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

from app.inference import (
    GpuInferenceService,
    InferenceExecutionTimeoutError,
    InferenceQueueFullError,
    InferenceQueueTimeoutError,
    InvalidImageError,
)
from app.model import (
    ModelName,
    ModelUnavailableError,
    get_inference_provider,
    get_model_status,
)

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY must be set and non-empty.")
try:
    API_KEY_BYTES = API_KEY.encode("ascii")
except UnicodeEncodeError as exc:
    raise RuntimeError("API_KEY must contain ASCII characters only.") from exc
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

MAX_FAILED_ATTEMPTS = 5
BLOCK_DURATION = 900
ATTEMPT_WINDOW = 60

_failed_attempts: dict[str, list[float]] = {}
_blocked_ips: dict[str, float] = {}

ALLOWED_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/avif",
        "image/svg+xml",
    }
)
MAX_FILE_SIZE = 20 * 1024 * 1024


def _get_client_ip(request: Request) -> str:
    peer_ip = request.client.host if request.client else "unknown"
    if peer_ip not in {"127.0.0.1", "::1"}:
        return peer_ip

    cloudflare_ip = request.headers.get("CF-Connecting-IP")
    if cloudflare_ip:
        try:
            return str(ip_address(cloudflare_ip.strip()))
        except ValueError:
            pass
    return peer_ip


def _cleanup_expired_entries() -> None:
    """Remove expired rate-limit entries."""
    now = time.monotonic()
    expired_blocked = [
        ip for ip, ts in _blocked_ips.items() if now - ts >= BLOCK_DURATION
    ]
    for ip in expired_blocked:
        del _blocked_ips[ip]
        _failed_attempts.pop(ip, None)

    expired_attempts = [
        ip
        for ip, attempts in _failed_attempts.items()
        if not any(now - t < ATTEMPT_WINDOW for t in attempts)
    ]
    for ip in expired_attempts:
        del _failed_attempts[ip]


def _is_blocked(ip: str) -> bool:
    if ip in _blocked_ips:
        if time.monotonic() - _blocked_ips[ip] < BLOCK_DURATION:
            return True
        del _blocked_ips[ip]
        _failed_attempts.pop(ip, None)
    return False


def _record_failure(ip: str) -> None:
    now = time.monotonic()
    attempts = _failed_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < ATTEMPT_WINDOW]
    attempts.append(now)
    _failed_attempts[ip] = attempts

    if len(attempts) >= MAX_FAILED_ATTEMPTS:
        _blocked_ips[ip] = now
        logger.warning(
            "Blocked %s for %ss after %s failed auth attempts",
            ip,
            BLOCK_DURATION,
            len(attempts),
        )


async def verify_api_key(
    request: Request,
    key: str | None = Security(api_key_header),
) -> None:
    ip = _get_client_ip(request)

    if _is_blocked(ip):
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Try again later.",
            headers={"Retry-After": str(BLOCK_DURATION)},
        )

    try:
        key_bytes = key.encode("ascii") if key is not None else b""
    except UnicodeEncodeError:
        key_bytes = b""
    if not secrets.compare_digest(key_bytes, API_KEY_BYTES):
        _record_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


CLEANUP_INTERVAL = 300  # seconds between rate-limit cleanup sweeps
inference_service = GpuInferenceService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async def _periodic_cleanup(
        *,
        task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        task_status.started()
        while True:
            await anyio.sleep(CLEANUP_INTERVAL)
            _cleanup_expired_entries()

    async with anyio.create_task_group() as tg:
        await tg.start(inference_service.run)
        await tg.start(_periodic_cleanup)
        yield
        tg.cancel_scope.cancel()


app = FastAPI(title="BackRemove", version="1.1.0", lifespan=lifespan)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["X-API-Key", "Content-Type"],
        expose_headers=[
            "X-Model-Used",
            "X-Queue-Time-Ms",
            "X-Inference-Time-Ms",
        ],
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "inference_provider": get_inference_provider(),
        "models": get_model_status(),
        "gpu_queue": inference_service.status(),
    }


@app.post("/remove-bg", dependencies=[Security(verify_api_key)])
async def remove_bg(
    file: Annotated[UploadFile, File()],
    model: ModelName = ModelName.FAST,
) -> Response:
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {content_type or 'missing'}. "
                f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}"
            ),
        )

    try:
        image_bytes = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 20 MB.")

    try:
        inference = await inference_service.infer(
            image_bytes,
            content_type,
            model,
        )
    except (InferenceQueueFullError, InferenceQueueTimeoutError) as exc:
        logger.warning("GPU queue rejected request: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "2"},
        ) from exc
    except InferenceExecutionTimeoutError as exc:
        logger.error("GPU inference response deadline exceeded: %s", exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        logger.error("Requested model is unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Background removal failed")
        raise HTTPException(
            status_code=500,
            detail="Background removal failed.",
        ) from exc
    finally:
        del image_bytes

    return Response(
        content=inference.image_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": "attachment; filename=no-bg.png",
            "X-Model-Used": inference.model.value,
            "X-Queue-Time-Ms": f"{inference.queue_ms:.1f}",
            "X-Inference-Time-Ms": f"{inference.inference_ms:.1f}",
        },
    )

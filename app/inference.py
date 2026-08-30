import io
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import anyio
from anyio.abc import TaskStatus
from PIL import Image

from app.model import ModelName, load_enabled_models, remove_background

DEFAULT_QUEUE_CAPACITY = 8
DEFAULT_QUEUE_TIMEOUT = 10.0
DEFAULT_INFERENCE_TIMEOUT = 75.0
DEFAULT_PROXY_BUDGET = 90.0
MAX_IMAGE_PIXELS = 40_000_000


class InferenceQueueFullError(RuntimeError):
    pass


class InferenceQueueTimeoutError(RuntimeError):
    pass


class InferenceExecutionTimeoutError(RuntimeError):
    pass


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceResult:
    image_bytes: bytes
    model: ModelName
    queue_ms: float
    inference_ms: float


@dataclass(slots=True)
class _InferenceJob:
    image_bytes: bytes
    content_type: str
    model: ModelName
    queued_at: float = field(default_factory=time.monotonic)
    started: anyio.Event = field(default_factory=anyio.Event)
    done: anyio.Event = field(default_factory=anyio.Event)
    started_at: float | None = None
    completed_at: float | None = None
    result: bytes | None = None
    error: Exception | None = None
    cancelled: bool = False


def _positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def process_image(
    image_bytes: bytes,
    content_type: str,
    model: ModelName,
) -> bytes:
    input_image = None
    result = None
    output = None

    try:
        if content_type == "image/svg+xml":
            import cairosvg

            try:
                raster_bytes = cairosvg.svg2png(bytestring=image_bytes)
            except Exception as exc:
                raise InvalidImageError("Invalid SVG data.") from exc
        else:
            raster_bytes = image_bytes

        try:
            with (
                io.BytesIO(raster_bytes) as source,
                Image.open(source) as decoded,
            ):
                if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
                    raise InvalidImageError(
                        f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit."
                    )
                input_image = decoded.convert("RGB")
        except InvalidImageError:
            raise
        except (Image.DecompressionBombError, OSError, ValueError) as exc:
            raise InvalidImageError("Invalid image data.") from exc

        result = remove_background(input_image, model)
        output = io.BytesIO()
        result.save(output, format="PNG")
        return output.getvalue()
    finally:
        if input_image is not None:
            input_image.close()
        if result is not None:
            result.close()
        if output is not None:
            output.close()


class GpuInferenceService:
    def __init__(
        self,
        *,
        processor: Callable[[bytes, str, ModelName], bytes] = process_image,
        preloader: Callable[[], object] = load_enabled_models,
        queue_capacity: int | None = None,
        queue_timeout: float | None = None,
        inference_timeout: float | None = None,
        proxy_budget: float | None = None,
    ) -> None:
        self.queue_capacity = (
            _positive_int("GPU_QUEUE_CAPACITY", DEFAULT_QUEUE_CAPACITY)
            if queue_capacity is None
            else queue_capacity
        )
        self.queue_timeout = (
            _positive_float("GPU_QUEUE_TIMEOUT", DEFAULT_QUEUE_TIMEOUT)
            if queue_timeout is None
            else queue_timeout
        )
        self.inference_timeout = (
            _positive_float("INFERENCE_TIMEOUT", DEFAULT_INFERENCE_TIMEOUT)
            if inference_timeout is None
            else inference_timeout
        )
        self.proxy_budget = (
            _positive_float("PROXY_REQUEST_BUDGET", DEFAULT_PROXY_BUDGET)
            if proxy_budget is None
            else proxy_budget
        )
        for name, value in (
            ("GPU_QUEUE_CAPACITY", self.queue_capacity),
            ("GPU_QUEUE_TIMEOUT", self.queue_timeout),
            ("INFERENCE_TIMEOUT", self.inference_timeout),
            ("PROXY_REQUEST_BUDGET", self.proxy_budget),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.queue_timeout + self.inference_timeout >= self.proxy_budget:
            raise ValueError(
                "GPU_QUEUE_TIMEOUT + INFERENCE_TIMEOUT must stay below "
                "PROXY_REQUEST_BUDGET."
            )

        self._processor = processor
        self._preloader = preloader
        self._send = None
        self._busy_model: ModelName | None = None
        self._started = False

    async def run(
        self,
        *,
        task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        send, receive = anyio.create_memory_object_stream[_InferenceJob](
            self.queue_capacity
        )
        self._send = send
        try:
            async with send, receive:
                await anyio.to_thread.run_sync(self._preloader)
                self._started = True
                task_status.started()

                async for job in receive:
                    if job.cancelled:
                        job.completed_at = time.monotonic()
                        job.done.set()
                        continue

                    job.started_at = time.monotonic()
                    self._busy_model = job.model
                    job.started.set()
                    try:
                        job.result = await anyio.to_thread.run_sync(
                            self._processor,
                            job.image_bytes,
                            job.content_type,
                            job.model,
                        )
                    except Exception as exc:  # noqa: BLE001
                        job.error = exc
                    finally:
                        job.completed_at = time.monotonic()
                        self._busy_model = None
                        job.done.set()
        finally:
            self._started = False
            self._busy_model = None
            self._send = None

    async def infer(
        self,
        image_bytes: bytes,
        content_type: str,
        model: ModelName,
    ) -> InferenceResult:
        send = self._send
        if not self._started or send is None:
            raise RuntimeError("GPU inference service is not running.")

        job = _InferenceJob(
            image_bytes=image_bytes,
            content_type=content_type,
            model=model,
        )
        try:
            send.send_nowait(job)
        except anyio.WouldBlock as exc:
            raise InferenceQueueFullError("GPU inference queue is full.") from exc

        try:
            with anyio.fail_after(self.queue_timeout):
                await job.started.wait()
        except TimeoutError as exc:
            if not job.started.is_set():
                job.cancelled = True
                raise InferenceQueueTimeoutError(
                    "Timed out waiting for the GPU queue."
                ) from exc
        except anyio.get_cancelled_exc_class():
            if not job.started.is_set():
                job.cancelled = True
            raise

        try:
            with anyio.fail_after(self.inference_timeout):
                await job.done.wait()
        except TimeoutError as exc:
            raise InferenceExecutionTimeoutError(
                "GPU inference exceeded its response deadline."
            ) from exc

        if job.error is not None:
            raise job.error
        if job.result is None or job.started_at is None or job.completed_at is None:
            raise RuntimeError("GPU inference completed without a result.")

        return InferenceResult(
            image_bytes=job.result,
            model=job.model,
            queue_ms=(job.started_at - job.queued_at) * 1000.0,
            inference_ms=(job.completed_at - job.started_at) * 1000.0,
        )

    def status(self) -> dict[str, object]:
        send = self._send
        queued = send.statistics().current_buffer_used if send is not None else 0
        return {
            "started": self._started,
            "busy": self._busy_model is not None,
            "active_model": self._busy_model.value if self._busy_model else None,
            "queued": queued,
            "capacity": self.queue_capacity,
            "queue_timeout_seconds": self.queue_timeout,
            "inference_timeout_seconds": self.inference_timeout,
        }

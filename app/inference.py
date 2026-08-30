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
PNG_COMPRESSION_LEVEL = 3
ENCODE_CONCURRENCY = 2


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
    decode_ms: float
    inference_ms: float
    encode_ms: float


@dataclass(slots=True)
class _InferenceJob:
    image_bytes: bytes
    content_type: str
    model: ModelName
    queued_at: float = field(default_factory=time.monotonic)
    started: anyio.Event = field(default_factory=anyio.Event)
    done: anyio.Event = field(default_factory=anyio.Event)
    decoded_image: Image.Image | None = None
    result: Image.Image | None = None
    started_at: float | None = None
    decoded_at: float | None = None
    completed_at: float | None = None
    decode_ms: float = 0.0
    error: Exception | None = None
    cancelled: bool = False
    abandoned: bool = False
    admission_token: object = field(default_factory=object)
    encode_token: object = field(default_factory=object)
    encode_slot_acquired: bool = False


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


def decode_image(
    image_bytes: bytes,
    content_type: str,
) -> Image.Image:
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
            return decoded.convert("RGB")
    except InvalidImageError:
        raise
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise InvalidImageError("Invalid image data.") from exc


def encode_png(result: Image.Image) -> bytes:
    with io.BytesIO() as output:
        result.save(
            output,
            format="PNG",
            compress_level=PNG_COMPRESSION_LEVEL,
        )
        return output.getvalue()


class GpuInferenceService:
    def __init__(
        self,
        *,
        decoder: Callable[[bytes, str], Image.Image] = decode_image,
        processor: Callable[[Image.Image, ModelName], Image.Image] = (
            remove_background
        ),
        encoder: Callable[[Image.Image], bytes] = encode_png,
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

        self._decoder = decoder
        self._processor = processor
        self._encoder = encoder
        self._preloader = preloader
        self._send = None
        self._prepared_send = None
        self._admission = None
        self._encode_limiter = None
        self._waiting_for_result_slot = 0
        self._preparing = 0
        self._busy_model: ModelName | None = None
        self._started = False

    async def _decode_jobs(self, receive, prepared_send) -> None:
        async with receive, prepared_send:
            async for job in receive:
                if job.cancelled:
                    job.completed_at = time.monotonic()
                    job.done.set()
                    continue

                self._preparing += 1
                decode_started_at = time.monotonic()
                decoded_image = None
                try:
                    decoded_image = await anyio.to_thread.run_sync(
                        self._decoder,
                        job.image_bytes,
                        job.content_type,
                    )
                    job.image_bytes = b""
                    job.decoded_at = time.monotonic()
                    job.decode_ms = (
                        job.decoded_at - decode_started_at
                    ) * 1000.0
                    if job.cancelled:
                        decoded_image.close()
                        decoded_image = None
                        job.completed_at = time.monotonic()
                        job.done.set()
                        continue

                    job.decoded_image = decoded_image
                    decoded_image = None
                    try:
                        await prepared_send.send(job)
                    except BaseException:
                        if job.decoded_image is not None:
                            job.decoded_image.close()
                            job.decoded_image = None
                        raise
                except Exception as exc:  # noqa: BLE001
                    job.image_bytes = b""
                    now = time.monotonic()
                    job.decode_ms = (now - decode_started_at) * 1000.0
                    job.error = exc
                    job.started_at = now
                    job.completed_at = now
                    job.started.set()
                    job.done.set()
                finally:
                    if decoded_image is not None:
                        decoded_image.close()
                    self._preparing -= 1

    async def _run_gpu_jobs(self, receive, encode_limiter) -> None:
        async with receive:
            async for job in receive:
                input_image = job.decoded_image
                job.decoded_image = None
                if input_image is None:
                    if job.error is None:
                        job.error = RuntimeError(
                            "GPU job started without a decoded image."
                        )
                    job.completed_at = time.monotonic()
                    job.done.set()
                    continue
                if job.cancelled:
                    input_image.close()
                    job.completed_at = time.monotonic()
                    job.done.set()
                    continue

                result = None
                try:
                    self._waiting_for_result_slot += 1
                    try:
                        await encode_limiter.acquire_on_behalf_of(
                            job.encode_token
                        )
                        job.encode_slot_acquired = True
                    finally:
                        self._waiting_for_result_slot -= 1

                    if job.cancelled:
                        continue

                    job.started_at = time.monotonic()
                    self._busy_model = job.model
                    job.started.set()
                    result = await anyio.to_thread.run_sync(
                        self._processor,
                        input_image,
                        job.model,
                    )
                    if job.abandoned:
                        result.close()
                    else:
                        job.result = result
                    result = None
                except Exception as exc:  # noqa: BLE001
                    job.error = exc
                finally:
                    if result is not None:
                        result.close()
                    input_image.close()
                    if job.result is None and job.encode_slot_acquired:
                        encode_limiter.release_on_behalf_of(job.encode_token)
                        job.encode_slot_acquired = False
                    job.completed_at = time.monotonic()
                    self._busy_model = None
                    job.done.set()

    async def run(
        self,
        *,
        task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        send, receive = anyio.create_memory_object_stream[_InferenceJob](
            self.queue_capacity
        )
        prepared_send, prepared_receive = (
            anyio.create_memory_object_stream[_InferenceJob](1)
        )
        try:
            await anyio.to_thread.run_sync(self._preloader)
            self._send = send
            self._prepared_send = prepared_send
            self._admission = anyio.CapacityLimiter(self.queue_capacity + 1)
            self._encode_limiter = anyio.CapacityLimiter(ENCODE_CONCURRENCY)
            self._started = True

            async with send:
                async with anyio.create_task_group() as workers:
                    workers.start_soon(
                        self._decode_jobs,
                        receive,
                        prepared_send,
                    )
                    workers.start_soon(
                        self._run_gpu_jobs,
                        prepared_receive,
                        self._encode_limiter,
                    )
                    task_status.started()
                    await anyio.sleep_forever()
        finally:
            self._started = False
            self._busy_model = None
            self._preparing = 0
            self._waiting_for_result_slot = 0
            self._encode_limiter = None
            self._admission = None
            self._prepared_send = None
            self._send = None

    async def infer(
        self,
        image_bytes: bytes,
        content_type: str,
        model: ModelName,
    ) -> InferenceResult:
        send = self._send
        admission = self._admission
        encode_limiter = self._encode_limiter
        if (
            not self._started
            or send is None
            or admission is None
            or encode_limiter is None
        ):
            raise RuntimeError("GPU inference service is not running.")

        job = _InferenceJob(
            image_bytes=image_bytes,
            content_type=content_type,
            model=model,
        )
        try:
            admission.acquire_on_behalf_of_nowait(job.admission_token)
        except anyio.WouldBlock as exc:
            raise InferenceQueueFullError("GPU inference queue is full.") from exc

        try:
            try:
                send.send_nowait(job)
            except anyio.WouldBlock as exc:
                raise InferenceQueueFullError(
                    "GPU inference queue is full."
                ) from exc

            try:
                with anyio.fail_after(self.queue_timeout):
                    await job.started.wait()
            except TimeoutError as exc:
                if not job.started.is_set():
                    job.cancelled = True
                    job.abandoned = True
                    raise InferenceQueueTimeoutError(
                        "Timed out waiting for the GPU queue."
                    ) from exc
            except anyio.get_cancelled_exc_class():
                if not job.started.is_set():
                    job.cancelled = True
                job.abandoned = True
                raise

            try:
                with anyio.fail_after(self.inference_timeout):
                    await job.done.wait()
            except TimeoutError as exc:
                if not job.done.is_set():
                    job.abandoned = True
                    raise InferenceExecutionTimeoutError(
                        "GPU inference exceeded its response deadline."
                    ) from exc
            except anyio.get_cancelled_exc_class():
                job.abandoned = True
                raise

            if job.error is not None:
                raise job.error
            if (
                job.result is None
                or job.decoded_at is None
                or job.started_at is None
                or job.completed_at is None
            ):
                raise RuntimeError("GPU inference completed without a result.")

            result = job.result
            job.result = None
            encode_started_at = time.monotonic()
            try:
                encoded = await anyio.to_thread.run_sync(
                    self._encoder,
                    result,
                )
            finally:
                result.close()
                encode_limiter.release_on_behalf_of(job.encode_token)
                job.encode_slot_acquired = False
            encode_ms = (time.monotonic() - encode_started_at) * 1000.0

            return InferenceResult(
                image_bytes=encoded,
                model=job.model,
                queue_ms=(job.started_at - job.decoded_at) * 1000.0,
                decode_ms=job.decode_ms,
                inference_ms=(job.completed_at - job.started_at) * 1000.0,
                encode_ms=encode_ms,
            )
        finally:
            if job.result is not None:
                job.result.close()
                job.result = None
                if job.encode_slot_acquired:
                    encode_limiter.release_on_behalf_of(job.encode_token)
                    job.encode_slot_acquired = False
            admission.release_on_behalf_of(job.admission_token)

    def status(self) -> dict[str, object]:
        send = self._send
        prepared_send = self._prepared_send
        queued = self._preparing + self._waiting_for_result_slot
        if send is not None:
            queued += send.statistics().current_buffer_used
        if prepared_send is not None:
            queued += prepared_send.statistics().current_buffer_used
        return {
            "started": self._started,
            "busy": self._busy_model is not None,
            "active_model": self._busy_model.value if self._busy_model else None,
            "queued": queued,
            "capacity": self.queue_capacity,
            "queue_timeout_seconds": self.queue_timeout,
            "inference_timeout_seconds": self.inference_timeout,
        }

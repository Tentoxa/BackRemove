import io
import threading
import time
import unittest
from unittest.mock import patch

import anyio
from PIL import Image

from app.inference import (
    GpuInferenceService,
    InferenceExecutionTimeoutError,
    InferenceQueueFullError,
    InferenceQueueTimeoutError,
    InvalidImageError,
    decode_image,
)
from app.model import ModelName


class _FakeImage:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def close(self):
        self.closed = True


def _decode_fake(image_bytes, content_type):
    return _FakeImage(image_bytes)


def _encode_fake(result):
    return result.payload


class ImageProcessingTests(unittest.TestCase):
    def test_invalid_image_data_is_rejected(self):
        with self.assertRaises(InvalidImageError):
            decode_image(b"not an image", "image/png")

    def test_decoded_pixel_limit_is_enforced_before_inference(self):
        source = Image.new("RGB", (2, 2))
        encoded = io.BytesIO()
        try:
            source.save(encoded, format="PNG")
        finally:
            source.close()
        image_bytes = encoded.getvalue()
        encoded.close()

        with (
            patch("app.inference.MAX_IMAGE_PIXELS", 1),
            self.assertRaises(InvalidImageError),
        ):
            decode_image(image_bytes, "image/png")


class GpuInferenceServiceTests(unittest.TestCase):
    def test_gpu_jobs_are_serialized_and_routed(self):
        active = 0
        max_active = 0
        observed_models = []
        state_lock = threading.Lock()

        def processor(input_image, model):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                observed_models.append(model)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return _FakeImage(
                input_image.payload + model.value.encode("ascii")
            )

        async def scenario():
            service = GpuInferenceService(
                decoder=_decode_fake,
                processor=processor,
                encoder=_encode_fake,
                preloader=lambda: None,
                queue_capacity=4,
                queue_timeout=1.0,
                inference_timeout=1.0,
                proxy_budget=3.0,
            )
            results = [None, None, None]
            models = [ModelName.FAST, ModelName.QUALITY, ModelName.FAST]

            async def submit(index):
                results[index] = await service.infer(
                    b"image-", "image/png", models[index]
                )

            async with anyio.create_task_group() as service_tasks:
                await service_tasks.start(service.run)
                async with anyio.create_task_group() as jobs:
                    for index in range(len(models)):
                        jobs.start_soon(submit, index)
                service_tasks.cancel_scope.cancel()

            self.assertEqual(max_active, 1)
            self.assertEqual(observed_models, models)
            self.assertEqual(
                [result.image_bytes for result in results],
                [b"image-fast", b"image-quality", b"image-fast"],
            )

        anyio.run(scenario)

    def test_full_queue_is_rejected_without_starting_another_gpu_job(self):
        processor_started = threading.Event()
        release_processor = threading.Event()

        def processor(input_image, model):
            processor_started.set()
            release_processor.wait(timeout=2.0)
            return _FakeImage(input_image.payload)

        async def scenario():
            service = GpuInferenceService(
                decoder=_decode_fake,
                processor=processor,
                encoder=_encode_fake,
                preloader=lambda: None,
                queue_capacity=1,
                queue_timeout=1.0,
                inference_timeout=2.0,
                proxy_budget=4.0,
            )

            async with anyio.create_task_group() as service_tasks:
                await service_tasks.start(service.run)
                async with anyio.create_task_group() as jobs:
                    jobs.start_soon(
                        service.infer,
                        b"first",
                        "image/png",
                        ModelName.FAST,
                    )
                    while not processor_started.is_set():
                        await anyio.sleep(0.005)

                    jobs.start_soon(
                        service.infer,
                        b"second",
                        "image/png",
                        ModelName.FAST,
                    )
                    while service.status()["queued"] != 1:
                        await anyio.sleep(0.005)

                    with self.assertRaises(InferenceQueueFullError):
                        await service.infer(b"third", "image/png", ModelName.FAST)
                    release_processor.set()
                service_tasks.cancel_scope.cancel()

        anyio.run(scenario)

    def test_queue_timeout_cancels_job_before_gpu_execution(self):
        first_started = threading.Event()
        release_first = threading.Event()
        executions = []

        def processor(input_image, model):
            executions.append(input_image.payload)
            first_started.set()
            release_first.wait(timeout=2.0)
            return _FakeImage(input_image.payload)

        async def scenario():
            service = GpuInferenceService(
                decoder=_decode_fake,
                processor=processor,
                encoder=_encode_fake,
                preloader=lambda: None,
                queue_capacity=2,
                queue_timeout=0.03,
                inference_timeout=1.0,
                proxy_budget=2.0,
            )

            async with anyio.create_task_group() as service_tasks:
                await service_tasks.start(service.run)
                async with anyio.create_task_group() as jobs:
                    jobs.start_soon(
                        service.infer,
                        b"first",
                        "image/png",
                        ModelName.FAST,
                    )
                    while not first_started.is_set():
                        await anyio.sleep(0.005)

                    with self.assertRaises(InferenceQueueTimeoutError):
                        await service.infer(b"expired", "image/png", ModelName.QUALITY)
                    release_first.set()
                while service.status()["queued"]:
                    await anyio.sleep(0.005)
                service_tasks.cancel_scope.cancel()

            self.assertEqual(executions, [b"first"])

        anyio.run(scenario)

    def test_cancelled_queued_request_never_reaches_gpu(self):
        first_started = threading.Event()
        release_first = threading.Event()
        executions = []

        def processor(input_image, model):
            executions.append(input_image.payload)
            first_started.set()
            release_first.wait(timeout=2.0)
            return _FakeImage(input_image.payload)

        async def scenario():
            service = GpuInferenceService(
                decoder=_decode_fake,
                processor=processor,
                encoder=_encode_fake,
                preloader=lambda: None,
                queue_capacity=2,
                queue_timeout=1.0,
                inference_timeout=2.0,
                proxy_budget=4.0,
            )
            cancel_scope = anyio.CancelScope()
            cancelled = anyio.Event()

            async def submit_cancelled():
                with cancel_scope:
                    try:
                        await service.infer(
                            b"cancelled",
                            "image/png",
                            ModelName.QUALITY,
                        )
                    finally:
                        cancelled.set()

            async with anyio.create_task_group() as service_tasks:
                await service_tasks.start(service.run)
                async with anyio.create_task_group() as jobs:
                    jobs.start_soon(
                        service.infer,
                        b"first",
                        "image/png",
                        ModelName.FAST,
                    )
                    while not first_started.is_set():
                        await anyio.sleep(0.005)
                    jobs.start_soon(submit_cancelled)
                    while service.status()["queued"] != 1:
                        await anyio.sleep(0.005)
                    cancel_scope.cancel()
                    await cancelled.wait()
                    release_first.set()
                while service.status()["queued"]:
                    await anyio.sleep(0.005)
                service_tasks.cancel_scope.cancel()

            self.assertEqual(executions, [b"first"])

        anyio.run(scenario)

    def test_png_encoding_does_not_hold_gpu_slot(self):
        first_encode_started = threading.Event()
        release_first_encode = threading.Event()
        second_gpu_started = threading.Event()
        results = [None, None]

        def processor(input_image, model):
            if input_image.payload == b"second":
                second_gpu_started.set()
            return _FakeImage(input_image.payload)

        def encoder(result):
            if result.payload == b"first":
                first_encode_started.set()
                release_first_encode.wait(timeout=2.0)
            return result.payload

        async def scenario():
            service = GpuInferenceService(
                decoder=_decode_fake,
                processor=processor,
                encoder=encoder,
                preloader=lambda: None,
                queue_capacity=2,
                queue_timeout=1.0,
                inference_timeout=1.0,
                proxy_budget=3.0,
            )

            async def submit(index, payload):
                results[index] = await service.infer(
                    payload,
                    "image/png",
                    ModelName.FAST,
                )

            async with anyio.create_task_group() as service_tasks:
                await service_tasks.start(service.run)
                async with anyio.create_task_group() as jobs:
                    jobs.start_soon(submit, 0, b"first")
                    while not first_encode_started.is_set():
                        await anyio.sleep(0.005)
                    jobs.start_soon(submit, 1, b"second")
                    with anyio.fail_after(0.5):
                        while not second_gpu_started.is_set():
                            await anyio.sleep(0.005)
                    release_first_encode.set()
                service_tasks.cancel_scope.cancel()

            self.assertEqual(
                [result.image_bytes for result in results],
                [b"first", b"second"],
            )

        anyio.run(scenario)

    def test_encode_capacity_blocks_a_third_gpu_result(self):
        first_encode_started = threading.Event()
        second_encode_started = threading.Event()
        third_gpu_started = threading.Event()
        release_encoders = threading.Event()
        results = [None, None, None]

        def processor(input_image, model):
            if input_image.payload == b"third":
                third_gpu_started.set()
            return _FakeImage(input_image.payload)

        def encoder(result):
            if result.payload == b"first":
                first_encode_started.set()
                release_encoders.wait(timeout=2.0)
            elif result.payload == b"second":
                second_encode_started.set()
                release_encoders.wait(timeout=2.0)
            return result.payload

        async def scenario():
            service = GpuInferenceService(
                decoder=_decode_fake,
                processor=processor,
                encoder=encoder,
                preloader=lambda: None,
                queue_capacity=3,
                queue_timeout=1.0,
                inference_timeout=1.0,
                proxy_budget=3.0,
            )

            async def submit(index, payload):
                results[index] = await service.infer(
                    payload,
                    "image/png",
                    ModelName.FAST,
                )

            async with anyio.create_task_group() as service_tasks:
                await service_tasks.start(service.run)
                async with anyio.create_task_group() as jobs:
                    jobs.start_soon(submit, 0, b"first")
                    while not first_encode_started.is_set():
                        await anyio.sleep(0.005)
                    jobs.start_soon(submit, 1, b"second")
                    while not second_encode_started.is_set():
                        await anyio.sleep(0.005)
                    jobs.start_soon(submit, 2, b"third")
                    try:
                        with anyio.move_on_after(0.05) as started_early:
                            while not third_gpu_started.is_set():
                                await anyio.sleep(0.005)
                    finally:
                        release_encoders.set()
                    self.assertTrue(started_early.cancel_called)
                    with anyio.fail_after(0.5):
                        while not third_gpu_started.is_set():
                            await anyio.sleep(0.005)
                service_tasks.cancel_scope.cancel()

            self.assertEqual(
                [result.image_bytes for result in results],
                [b"first", b"second", b"third"],
            )

        anyio.run(scenario)

    def test_response_timeout_does_not_release_gpu_slot_early(self):
        executions = []
        produced = []

        def processor(input_image, model):
            executions.append(("start", input_image.payload))
            time.sleep(0.08)
            executions.append(("end", input_image.payload))
            result = _FakeImage(input_image.payload)
            produced.append(result)
            return result

        async def scenario():
            service = GpuInferenceService(
                decoder=_decode_fake,
                processor=processor,
                encoder=_encode_fake,
                preloader=lambda: None,
                queue_capacity=2,
                queue_timeout=0.5,
                inference_timeout=0.02,
                proxy_budget=1.0,
            )

            async with anyio.create_task_group() as service_tasks:
                await service_tasks.start(service.run)
                with self.assertRaises(InferenceExecutionTimeoutError):
                    await service.infer(b"slow", "image/png", ModelName.QUALITY)
                self.assertTrue(service.status()["busy"])
                while service.status()["busy"]:
                    await anyio.sleep(0.01)
                self.assertEqual(executions, [("start", b"slow"), ("end", b"slow")])
                self.assertTrue(produced[0].closed)
                service_tasks.cancel_scope.cancel()

        anyio.run(scenario)


if __name__ == "__main__":
    unittest.main()

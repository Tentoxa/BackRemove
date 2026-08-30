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
    process_image,
)
from app.model import ModelName


class ImageProcessingTests(unittest.TestCase):
    def test_invalid_image_data_is_rejected(self):
        with self.assertRaises(InvalidImageError):
            process_image(b"not an image", "image/png", ModelName.FAST)

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
            process_image(image_bytes, "image/png", ModelName.FAST)


class GpuInferenceServiceTests(unittest.TestCase):
    def test_gpu_jobs_are_serialized_and_routed(self):
        active = 0
        max_active = 0
        observed_models = []
        state_lock = threading.Lock()

        def processor(image_bytes, content_type, model):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                observed_models.append(model)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return image_bytes + model.value.encode("ascii")

        async def scenario():
            service = GpuInferenceService(
                processor=processor,
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

        def processor(image_bytes, content_type, model):
            processor_started.set()
            release_processor.wait(timeout=2.0)
            return image_bytes

        async def scenario():
            service = GpuInferenceService(
                processor=processor,
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

        def processor(image_bytes, content_type, model):
            executions.append(image_bytes)
            first_started.set()
            release_first.wait(timeout=2.0)
            return image_bytes

        async def scenario():
            service = GpuInferenceService(
                processor=processor,
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

        def processor(image_bytes, content_type, model):
            executions.append(image_bytes)
            first_started.set()
            release_first.wait(timeout=2.0)
            return image_bytes

        async def scenario():
            service = GpuInferenceService(
                processor=processor,
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

    def test_response_timeout_does_not_release_gpu_slot_early(self):
        executions = []

        def processor(image_bytes, content_type, model):
            executions.append(("start", image_bytes))
            time.sleep(0.08)
            executions.append(("end", image_bytes))
            return image_bytes

        async def scenario():
            service = GpuInferenceService(
                processor=processor,
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
                service_tasks.cancel_scope.cancel()

        anyio.run(scenario)


if __name__ == "__main__":
    unittest.main()

import ctypes
import logging
import os
import site
from pathlib import Path

import numpy as np
import onnxruntime as ort
from withoutbg import WithoutBG
from withoutbg.models import OpenWeightsModel

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
VALID_DEVICES = {"auto", "cpu", "cuda"}

_model = None
_inference_provider = "unloaded"
_dll_directory_handles = []
_cuda_dll_handles = []


def _prepare_cuda_runtime() -> None:
    if os.name == "nt":
        dll_dirs = []
        for site_packages in site.getsitepackages():
            nvidia_dir = Path(site_packages) / "nvidia"
            if nvidia_dir.is_dir():
                dll_dirs.extend(sorted(nvidia_dir.glob("*/bin")))

        if dll_dirs:
            os.environ["PATH"] = os.pathsep.join(
                [*(str(path) for path in dll_dirs), os.environ.get("PATH", "")]
            )
            for dll_dir in dll_dirs:
                _dll_directory_handles.append(
                    os.add_dll_directory(str(dll_dir))
                )

            for dll_dir in dll_dirs:
                if dll_dir.parent.name != "cudnn":
                    continue
                for dll_path in sorted(dll_dir.glob("cudnn*.dll")):
                    _cuda_dll_handles.append(ctypes.WinDLL(str(dll_path)))

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")


def _prepare_model_for_cuda(model_path: Path) -> Path:
    """Defuse optimized operators that CUDAExecutionProvider cannot execute."""
    import onnx
    from onnx import helper

    output_path = model_path.with_name(f"{model_path.stem}.cuda{model_path.suffix}")
    if (
        output_path.is_file()
        and output_path.stat().st_mtime >= model_path.stat().st_mtime
    ):
        return output_path

    model = onnx.load(str(model_path))
    new_nodes = []
    defused = 0

    for node in model.graph.node:
        if node.op_type != "FusedConv":
            new_nodes.append(node)
            continue

        activation = next(
            (
                attr.s.decode() if isinstance(attr.s, bytes) else attr.s
                for attr in node.attribute
                if attr.name == "activation"
            ),
            None,
        )
        if activation != "Sigmoid":
            new_nodes.append(node)
            continue

        conv_attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in node.attribute
            if attr.name != "activation"
        }
        intermediate = f"{node.output[0]}_pre_sigmoid"
        new_nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    node.input,
                    [intermediate],
                    f"{node.name}_conv",
                    **conv_attrs,
                ),
                helper.make_node(
                    "Sigmoid",
                    [intermediate],
                    node.output,
                    f"{node.name}_sigmoid",
                ),
            ]
        )
        defused += 1

    if defused == 0:
        return model_path

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    logger.info("Prepared CUDA-compatible model at %s", output_path)
    return output_path


class _CudaOpenWeightsModel(OpenWeightsModel):
    def _load_model(self) -> None:
        if self.model_path is None:
            raise RuntimeError("Model path was not resolved.")

        _prepare_cuda_runtime()

        available = ort.get_available_providers()
        if CUDA_PROVIDER not in available:
            raise RuntimeError(
                f"{CUDA_PROVIDER} is unavailable. Available providers: {available}"
            )

        cuda_model_path = _prepare_model_for_cuda(self.model_path)
        self.session = ort.InferenceSession(
            str(cuda_model_path),
            providers=[CUDA_PROVIDER],
        )
        self.session.disable_fallback()
        canvas_size = int(self.sidecar.get("canvas_size", 448))
        input_name = self.sidecar.get("input_name", "rgb")
        self.session.run(
            None,
            {
                input_name: np.zeros(
                    (1, 3, canvas_size, canvas_size),
                    dtype=np.float32,
                )
            },
        )
        active_provider = self.session.get_providers()[0]
        if active_provider != CUDA_PROVIDER:
            raise RuntimeError(
                f"Expected {CUDA_PROVIDER}, got {active_provider}."
            )


def _load_cpu_model():
    model = WithoutBG.open_weights()
    model.preload()
    return model


def load_model():
    global _model, _inference_provider
    if _model is not None:
        return _model

    requested_device = os.environ.get("INFERENCE_DEVICE", "auto").lower()
    if requested_device not in VALID_DEVICES:
        raise ValueError(
            f"Invalid INFERENCE_DEVICE={requested_device!r}; "
            f"expected one of {sorted(VALID_DEVICES)}."
        )

    logger.info("Loading model (requested device: %s)...", requested_device)
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto"
        and CUDA_PROVIDER in ort.get_available_providers()
    )

    if use_cuda:
        try:
            model = _CudaOpenWeightsModel()
            model.preload()
            _model = model
            _inference_provider = CUDA_PROVIDER
        except Exception:
            if requested_device == "cuda":
                raise
            logger.warning(
                "CUDA initialization failed; falling back to CPU.",
                exc_info=True,
            )
            _model = _load_cpu_model()
            _inference_provider = CPU_PROVIDER
    else:
        _model = _load_cpu_model()
        _inference_provider = CPU_PROVIDER

    logger.info("Model ready (provider: %s).", _inference_provider)
    return _model


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded.")
    return _model


def get_inference_provider() -> str:
    return _inference_provider

"""The single selection point for inference backends."""


def create_predictor(backend, device, provider):
    if backend == "onnxruntime":
        from .onnxruntime_predictor import ONNXRuntimePredictor
        expected = {"cpu": "CPUExecutionProvider", "gpu": "CUDAExecutionProvider"}
        cls = ONNXRuntimePredictor
    elif backend == "pytorch":
        from .pytorch_predictor import PyTorchPredictor
        expected = {"cpu": "cpu", "gpu": "cuda"}
        cls = PyTorchPredictor
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    if device not in expected or provider != expected[device]:
        raise ValueError(f"Invalid device/provider for {backend}: {device}/{provider}")
    return cls(provider=provider)

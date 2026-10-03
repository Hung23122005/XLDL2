"""Best-effort hardware metadata and runtime provider discovery."""

import os
import platform
import subprocess



def detect_hardware(device, backend="onnxruntime"):
    if backend == "onnxruntime":
        import onnxruntime as ort
        providers = ort.get_available_providers()
        version = ort.__version__
        cuda = "CUDAExecutionProvider" in providers
    elif backend == "pytorch":
        import torch
        cuda = torch.cuda.is_available()
        providers = ["cpu"] + (["cuda"] if cuda else [])
        version = torch.__version__
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    info = {
        "device_type": device,
        "processor": platform.processor() or None,
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "available_providers": providers,
        "runtime_version": version,
        "backend": backend,
    }
    if device == "gpu":
        info.update(gpu_name=None, cuda_available=cuda)
        if backend == "pytorch" and cuda:
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        if backend == "onnxruntime" and cuda:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                if result.returncode == 0:
                    info["gpu_name"] = result.stdout.strip() or None
            except (OSError, subprocess.SubprocessError):
                pass
    return info

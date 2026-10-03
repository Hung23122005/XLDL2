"""Prepare ONNX CUDA libraries in this process without changing system PATH."""
import os
from importlib.metadata import distribution, PackageNotFoundError

_DLL_HANDLES = []
_PREPARED = False


def prepare_cuda_runtime():
    global _PREPARED
    if _PREPARED:
        return
    import onnxruntime as ort
    if os.name == "nt":
        directories = set()
        for package in (
            "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
            "nvidia-cufft-cu12", "nvidia-curand-cu12", "nvidia-cuda-nvrtc-cu12",
            "nvidia-nvjitlink-cu12",
        ):
            try:
                dist = distribution(package)
            except PackageNotFoundError:
                continue
            for file in dist.files or ():
                if str(file).lower().endswith(".dll"):
                    path = dist.locate_file(file).resolve().parent
                    if path.is_dir():
                        directories.add(str(path))
        for directory in sorted(directories):
            # Retain handles: closing them removes the DLL search directories.
            _DLL_HANDLES.append(os.add_dll_directory(directory))
        if directories:
            # cuDNN dynamically loads sublibraries through the Windows loader.
            os.environ["PATH"] = os.pathsep.join(sorted(directories)) + os.pathsep + os.environ.get("PATH", "")
    ort.preload_dlls(directory="")
    _PREPARED = True

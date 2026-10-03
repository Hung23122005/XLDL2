"""Reusable ONNX Runtime predictor with an explicitly selected provider."""

import onnxruntime as ort

from .base import BasePredictor


class ONNXRuntimePredictor(BasePredictor):
    artifact_format = "onnx"
    frameworks = ("onnx", "onnxruntime")
    def __init__(self, provider):
        self.provider = provider
        self._session = None
        self._input_name = None

    def load_model(self, manifest, artifact_path=None):
        """Load once per lifecycle; unload before loading another model."""
        if self._session is not None:
            raise RuntimeError("Model already loaded. Call unload_model() first.")

        if self.provider not in ort.get_available_providers():
            raise RuntimeError(f"{self.provider} is not available.")
        options = ort.SessionOptions()
        if self.provider != "CPUExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if artifact_path is None and getattr(manifest, "is_logical", False):
            from artifact import ArtifactManager
            artifact_path = ArtifactManager().get_artifact(manifest, "onnx")
        model_path = artifact_path if artifact_path is not None else getattr(manifest, "model_path", manifest)
        try:
            if self.provider == "CUDAExecutionProvider":
                # Use NVIDIA site-packages DLLs, independent of PyTorch's CUDA version.
                from ort_runtime import prepare_cuda_runtime
                prepare_cuda_runtime()
            session = ort.InferenceSession(str(model_path), options, providers=[self.provider])
        except Exception as error:
            raise RuntimeError(
                f"Cannot load model with {self.provider}; no fallback is allowed. "
                f"Check runtime libraries and model compatibility. Original error: {error}"
            ) from error
        session.disable_fallback()
        if self.provider not in session.get_providers():
            raise RuntimeError(f"Requested provider {self.provider} was not activated; refusing fallback.")
        input_name = session.get_inputs()[0].name
        self._session = session
        self._input_name = input_name
        print("ModelLoad")

    def predict(self, input_tensor):
        """Run inference with the existing session and return all outputs."""
        if self._session is None:
            raise RuntimeError("No model loaded. Call load_model() before predict().")
        return self._session.run(None, {self._input_name: input_tensor})

    def unload_model(self):
        """Release the session owned by this predictor."""
        if self._session is not None:
            self._session = None
            self._input_name = None
            print("ModelUnload")

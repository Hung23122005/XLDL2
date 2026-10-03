"""Compatibility alias for the original ONNX Predictor import."""
if __package__:
    from .predictors.onnxruntime_predictor import ONNXRuntimePredictor, ort
else:
    from predictors.onnxruntime_predictor import ONNXRuntimePredictor, ort

Predictor = ONNXRuntimePredictor

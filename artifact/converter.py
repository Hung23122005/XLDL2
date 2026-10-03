"""Create source architectures and convert persisted weights, never fresh weights."""
import numpy as np


def create_source_model(manifest, initialize_weights=False):
    from torchvision import models
    source = manifest.data["source"]
    if source["type"] != "torchvision" or source["model_name"] != "resnet18":
        raise ValueError("Only torchvision resnet18 source models are currently supported.")
    weights = None
    if initialize_weights and source.get("weights") is not None:
        name = source["weights"]
        try:
            weights = models.ResNet18_Weights[name]
        except KeyError as error:
            raise ValueError(f"Unknown ResNet18 weights: {name}") from error
    return models.resnet18(weights=weights).eval()


def load_source_model(manifest, path):
    import torch
    model = create_source_model(manifest)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def validate_onnx(manifest, model, onnx_path):
    import torch
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    results = []
    # Check both the requested batch and a second batch to validate dynamic export.
    for batch in sorted({manifest.batch_size, 2}):
        x = manifest.generate_input(batch_size=batch)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).detach().cpu().numpy()
        actual = session.run(None, {session.get_inputs()[0].name: x})[0]
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-3, atol=1e-4):
            raise RuntimeError("ONNX artifact validation failed against PyTorch source model.")
        results.append({"batch_size": batch, "max_abs_error": float(np.max(np.abs(actual - expected)))})
    print("ONNX validation passed against PyTorch source model.")
    return {"rtol": 1e-3, "atol": 1e-4, "checks": results}


def export_onnx(manifest, source_path, output_path):
    import torch
    import onnx
    model = load_source_model(manifest, source_path)
    x = torch.from_numpy(manifest.generate_input())
    input_name = manifest.data["inputs"][0].get("layer_name", "input")
    output_name = manifest.data["outputs"][0].get("layer_name", "output")
    torch.onnx.export(
        model, x, str(output_path),
        input_names=[input_name], output_names=[output_name],
        dynamic_axes={input_name: {0: "batch_size"}, output_name: {0: "batch_size"}},
        opset_version=manifest.data["artifacts"]["onnx"]["opset_version"],
        dynamo=False, external_data=False,
    )
    onnx.checker.check_model(str(output_path))
    return validate_onnx(manifest, model, output_path)

"""Read and validate the supported local, single-input manifest format."""

from pathlib import Path
from copy import deepcopy
import re

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


class Manifest:
    def __init__(self, data):
        if not isinstance(data, dict):
            raise ValueError("Manifest must contain a YAML mapping.")
        data = deepcopy(data)
        self.is_logical = "model" in data
        if self.is_logical:
            model = data["model"]
            if not isinstance(model, dict):
                raise ValueError("model must be a mapping.")
            for key in ("id", "name", "version", "task"):
                if not isinstance(model.get(key), str) or not model[key].strip():
                    raise ValueError(f"Missing or invalid manifest field: model.{key}")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", model["id"]):
                raise ValueError("model.id must contain only letters, digits, underscores or hyphens.")
            source = data.get("source")
            if not isinstance(source, dict) or source.get("framework") != "pytorch":
                raise ValueError("Logical models currently require source.framework: pytorch.")
            data.update({key: model[key] for key in ("name", "version", "task")})
            data["framework"] = {"name": source["framework"]}
            self.model_id = model["id"]
        else:
            self.model_id = "legacy_" + re.sub(r"[^a-z0-9_-]", "_", str(data.get("name", "model")).lower())
        self.data = data
        for field in (
            "name", "version", "task", "framework.name",
        ):
            value = self._required(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Manifest field '{field}' must be a non-empty string.")

        self.name = data["name"]
        self.version = data["version"]
        self.framework = data["framework"]["name"]
        if self.framework.lower() not in ("onnx", "onnxruntime", "pytorch"):
            raise ValueError(f"Unsupported framework: {self.framework}. Expected onnx or pytorch.")
        profiling = self._required("profiling")
        if not isinstance(profiling, dict):
            raise ValueError("Manifest profiling must be a mapping.")
        source = self._required("source")
        if not isinstance(source, dict):
            raise ValueError("Manifest source must be a mapping.")
        if self.framework.lower() == "pytorch":
            if source.get("type") != "torchvision":
                raise ValueError("PyTorch currently requires source.type: torchvision.")
            if self._required("source.model_name") != "resnet18":
                raise ValueError("Only torchvision resnet18 is supported.")
            if source.get("weights") is not None and not isinstance(source["weights"], str):
                raise ValueError("source.weights must be null or a torchvision weights enum name.")
            self.model_path = None
        else:
            if source.get("type", "local") != "local":
                raise ValueError("ONNX requires source.type: local.")
            model_path = self._required("source.path")
            if not isinstance(model_path, str) or not model_path.strip():
                raise ValueError("source.path must be a non-empty string.")
            self.model_path = resolve_repo_path(model_path)
            if not self.model_path.is_file():
                raise ValueError(f"Manifest model file not found: {self.model_path}")
        runtime = data.get("runtime")
        expected_runtime = "pytorch" if self.framework.lower() == "pytorch" else "onnxruntime"
        if runtime is not None and (not isinstance(runtime, dict) or runtime.get("name") != expected_runtime):
            raise ValueError(f"runtime.name must be {expected_runtime} for this manifest.")
        # Older manifests may carry runtime hints. AgentConfig owns execution.
        self.backend = data["profiling"].get("backend")
        self.device = data["profiling"].get("device")
        self.batch_size = self._required("profiling.batch_size")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Manifest field 'profiling.batch_size' must be a positive integer.")

        for field in ("inputs", "outputs"):
            specs = self._required(field)
            if not isinstance(specs, list) or not specs:
                raise ValueError(f"Manifest field '{field}' must be a non-empty list.")
            for index, spec in enumerate(specs):
                label = f"{field}[{index}]"
                if not isinstance(spec, dict):
                    raise ValueError(f"Manifest field '{label}' must be a mapping.")
                for key in ("shape", "element_type"):
                    if key not in spec:
                        raise ValueError(f"Missing manifest field: {label}.{key}")
                if "layer_name" in spec and (not isinstance(spec["layer_name"], str) or not spec["layer_name"].strip()):
                    raise ValueError(f"{label}.layer_name must be a non-empty string.")
                shape = spec["shape"]
                if not isinstance(shape, list) or any(
                    dim != "batch_size" and (type(dim) is not int or dim < 1)
                    for dim in shape
                ):
                    raise ValueError(f"{label}.shape must list positive integers or 'batch_size'.")
                try:
                    if not isinstance(spec["element_type"], str):
                        raise TypeError("dtype must be a string")
                    dtype = np.dtype(spec["element_type"])
                    if dtype.kind not in "fiub":
                        raise TypeError("only numeric or boolean tensors are supported")
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid {label}.element_type: {spec['element_type']}") from error
        if len(data["inputs"]) != 1:
            raise ValueError("This Agent currently supports exactly one input tensor.")
        self.input_shape = tuple(
            self.batch_size if dim == "batch_size" else dim
            for dim in data["inputs"][0]["shape"]
        )
        self.input_dtype = np.dtype(data["inputs"][0]["element_type"])
        if self.framework.lower() == "pytorch":
            if not self.is_logical and "artifacts" not in data:
                data["artifacts"] = {
                    "pytorch": {"path": f"models/{self.model_id}/pytorch/resnet18.pth"},
                    "onnx": {"path": f"models/{self.model_id}/onnx/resnet18.onnx", "opset_version": 13},
                }
            artifacts = self._required("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError("artifacts must be a mapping.")
            for target in ("pytorch", "onnx"):
                spec = artifacts.get(target)
                if not isinstance(spec, dict) or not isinstance(spec.get("path"), str) or not spec["path"].strip():
                    raise ValueError(f"Missing or invalid artifacts.{target}.path")
            opset = artifacts["onnx"].get("opset_version")
            if type(opset) is not int or opset < 13:
                raise ValueError("artifacts.onnx.opset_version must be an integer >= 13.")
            if resolve_repo_path(artifacts["pytorch"]["path"]) == resolve_repo_path(artifacts["onnx"]["path"]):
                raise ValueError("PyTorch and ONNX artifact paths must differ.")
            if self.input_dtype != np.dtype("float32"):
                raise ValueError("PyTorch source export currently requires float32 input.")
            if not data["inputs"][0]["shape"] or data["inputs"][0]["shape"][0] != "batch_size":
                raise ValueError("PyTorch source input must start with dynamic batch_size.")
        report = profiling.get("report_path")
        if report is not None and (not isinstance(report, str) or not report.strip()):
            raise ValueError("profiling.report_path must be a non-empty string.")
        self.report_path = resolve_repo_path(report) if report is not None else None
        if self.report_path is not None and self.report_path == self.model_path:
            raise ValueError("profiling.report_path must differ from source.path.")

    def _required(self, field):
        value = self.data
        for key in field.split("."):
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"Missing manifest field: {field}")
            value = value[key]
        return value

    @classmethod
    def load(cls, path):
        path = resolve_repo_path(path)
        try:
            with path.open(encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"Cannot read manifest '{path}': {error}") from error
        return cls(data)

    def generate_input(self, batch_size=None):
        """Generate synthetic data without preprocessing or dataset loading."""
        batch_size = self.batch_size if batch_size is None else batch_size
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        shape = tuple(batch_size if dim == "batch_size" else dim for dim in self.data["inputs"][0]["shape"])
        rng = np.random.default_rng(0)
        if self.input_dtype.kind == "f":
            return rng.standard_normal(shape).astype(self.input_dtype)
        return rng.integers(0, 2, size=shape).astype(self.input_dtype)

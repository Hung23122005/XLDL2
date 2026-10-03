"""Execution environment configuration, independent of model manifests."""

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PROVIDERS = {
    "onnxruntime": {"cpu": "CPUExecutionProvider", "gpu": "CUDAExecutionProvider"},
    "pytorch": {"cpu": "cpu", "gpu": "cuda"},
}
DEVICE_PROVIDERS = BACKEND_PROVIDERS["onnxruntime"]  # Legacy import.


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    backend: str
    device: str
    provider: str

    def __post_init__(self):
        for field in ("agent_id", "backend", "device", "provider"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Agent config '{field}' must be a non-empty string.")
        if self.backend not in BACKEND_PROVIDERS:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.device not in BACKEND_PROVIDERS[self.backend]:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.provider != BACKEND_PROVIDERS[self.backend][self.device]:
            raise ValueError(f"Provider {self.provider} does not match device {self.device}.")

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            with path.open(encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"Cannot read agent config '{path}': {error}") from error
        if not isinstance(data, dict) or not isinstance(data.get("runtime"), dict):
            raise ValueError("Agent config requires a runtime mapping.")
        try:
            return cls(data["agent_id"], **{
                key: data["runtime"][key] for key in ("backend", "device", "provider")
            })
        except KeyError as error:
            raise ValueError(f"Missing agent config field: {error.args[0]}") from error

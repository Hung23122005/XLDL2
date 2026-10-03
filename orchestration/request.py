"""Public request/result schemas for synchronous profiling."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from registry.models import json_value


@dataclass(frozen=True)
class ProfilingRequest:
    manifest_path: str | Path
    backend: str
    device: str
    action: str = "profile"
    report_path: str | Path | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        self.validate()

    def validate(self):
        for name in ("request_id", "backend", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        for name in ("manifest_path", "report_path"):
            value = getattr(self, name)
            if name == "report_path" and value is None:
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a non-empty str or Path.")
        if self.action != "profile":
            raise ValueError(f"Unsupported action: {self.action}. Only profile is supported.")

    def to_dict(self):
        return json_value(asdict(self))


@dataclass(frozen=True)
class ProfilingResult:
    request_id: str
    agent_id: str
    backend: str
    device: str
    action: str
    model_id: str
    status: str
    report_path: Path
    artifact_metadata: dict
    started_at: datetime
    finished_at: datetime
    duration_ms: float

    def to_dict(self):
        return json_value(asdict(self))

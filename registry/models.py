"""Detached, JSON-compatible snapshots of live Agent metadata."""
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from agent.agent import AgentStatus


def json_value(value):
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Metadata keys must be strings.")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported metadata type: {type(value).__name__}")


@dataclass(frozen=True)
class AgentRecord:
    agent_id: str
    backend: str
    device: str
    provider: str
    status: str
    hardware_info: dict
    capabilities: dict
    model_loaded: bool

    @classmethod
    def from_info(cls, info):
        if not isinstance(info, dict):
            raise ValueError("Agent.get_info() must return a dictionary.")
        fields = {}
        for name in ("agent_id", "backend", "device", "provider"):
            value = info.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing or invalid Agent metadata: {name}")
            fields[name] = value
        try:
            fields["status"] = AgentStatus(info.get("status")).value
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid Agent status: {info.get('status')}") from error
        hardware = info.get("hardware", info.get("hardware_info"))
        if not isinstance(hardware, dict) or not isinstance(info.get("capabilities"), dict):
            raise ValueError("Agent hardware and capabilities must be dictionaries.")
        if type(info.get("model_loaded")) is not bool:
            raise ValueError("Agent model_loaded must be a boolean.")
        return cls(**fields, hardware_info=json_value(hardware),
                   capabilities=json_value(info["capabilities"]), model_loaded=info["model_loaded"])

    def to_dict(self):
        return json_value(asdict(self))

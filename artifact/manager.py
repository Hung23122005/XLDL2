"""Persist source weights and reuse only artifacts with matching provenance."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .registry import default_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def artifact_path(manifest, target):
    path = Path(manifest.data["artifacts"][target]["path"]).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


class ArtifactManager:
    def __init__(self, registry=None):
        self.registry = registry or default_registry()

    def supports(self, manifest, target):
        source = manifest.framework.lower()
        return source == target or self.registry.supports(source, target)

    @staticmethod
    def _metadata_path(path):
        return path.with_suffix(path.suffix + ".json")

    def _read_metadata(self, path):
        metadata_path = self._metadata_path(path)
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a mapping.")
            return metadata
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Invalid artifact metadata: {metadata_path}") from error

    def _write_metadata(self, path, metadata):
        metadata = {**metadata, "sha256": checksum(path)}
        with TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / "metadata.json"
            temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            temporary.replace(self._metadata_path(path))

    def get_artifact(self, manifest, target):
        source = manifest.framework.lower()
        if source != target:
            converter = self.registry.get(source, target)
        if source != "pytorch":
            raise RuntimeError(f"Unsupported artifact source: {source}")
        if target not in manifest.data["artifacts"]:
            raise ValueError(f"Missing artifact declaration: artifacts.{target}")
        source_path = artifact_path(manifest, "pytorch")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_signature = signature({"model_id": manifest.model_id, "source": manifest.data["source"]})
        meta = self._read_metadata(source_path)
        if source_path.is_file():
            if meta is not None and (meta.get("signature") != source_signature or meta.get("sha256") != checksum(source_path)):
                raise RuntimeError("PyTorch artifact provenance mismatch; use a new model_id/artifact path for changed weights.")
            if meta is None:
                from .converter import load_source_model
                load_source_model(manifest, source_path)
                self._write_metadata(source_path, {"signature": source_signature, "model_id": manifest.model_id})
            print(f"Artifact reuse: {source_path}")
        else:
            import torch
            from .converter import create_source_model
            # A private seed makes fresh random source creation reproducible.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                model = create_source_model(manifest, initialize_weights=True)
            with TemporaryDirectory(dir=source_path.parent) as directory:
                temporary = Path(directory) / "source.pth"
                torch.save(model.state_dict(), temporary)
                temporary.replace(source_path)
            self._write_metadata(source_path, {"signature": source_signature, "model_id": manifest.model_id})
            print(f"Artifact created: {source_path}")
        if target == "pytorch":
            return source_path

        path = artifact_path(manifest, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_signature = signature({
            "source_sha256": checksum(source_path), "model_id": manifest.model_id,
            "inputs": manifest.data["inputs"], "outputs": manifest.data["outputs"],
            "conversion": manifest.data["artifacts"][target],
        })
        meta = self._read_metadata(path)
        if path.is_file() and meta and meta.get("signature") == expected_signature and meta.get("sha256") == checksum(path):
            print(f"Artifact reuse: {path}")
            return path
        with TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / path.name
            validation = converter(manifest, source_path, temporary)
            temporary.replace(path)
        self._write_metadata(path, {
            "signature": expected_signature, "model_id": manifest.model_id,
            "generated_from": str(source_path), "source_sha256": checksum(source_path),
            "validation": validation,
        })
        print(f"Artifact exported: {path}")
        return path

    def get_metadata(self, manifest, target, path):
        return {
            "model_id": manifest.model_id,
            "source_framework": manifest.framework.lower(),
            "artifact_format": target, "artifact_path": str(path),
            "generated_from": str(artifact_path(manifest, "pytorch")) if target != "pytorch" else None,
        }

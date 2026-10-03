"""Test persisted weights, export validation, cache reuse and invalidation."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent.manifest import Manifest
from artifact.converter import validate_onnx
from artifact.manager import ArtifactManager, checksum
from artifact.registry import default_registry


class ArtifactTests(unittest.TestCase):
    def manifest(self, directory):
        data = deepcopy(Manifest.load("manifests/resnet18.yaml").data)
        data["model"]["id"] = "test_source"
        data["inputs"][0]["shape"] = ["batch_size", 3, 8, 8]
        for target, suffix in (("pytorch", "pth"), ("onnx", "onnx")):
            data["artifacts"][target]["path"] = str(Path(directory) / f"source.{suffix}")
        return Manifest(data)

    @staticmethod
    def tiny_model(*args, **kwargs):
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(), torch.nn.Linear(3, 1000)
        ).eval()

    def test_fresh_export_reuse_and_provenance(self):
        with TemporaryDirectory() as directory, patch("artifact.converter.create_source_model", side_effect=self.tiny_model):
            manifest = self.manifest(directory)
            manager = ArtifactManager()
            source = manager.get_artifact(manifest, "pytorch")
            original = checksum(source)
            onnx = manager.get_artifact(manifest, "onnx")
            metadata = json.loads(onnx.with_suffix(".onnx.json").read_text())
            self.assertEqual(metadata["source_sha256"], original)
            self.assertEqual(len(metadata["validation"]["checks"]), 2)
            times = (source.stat().st_mtime_ns, onnx.stat().st_mtime_ns)
            with patch("artifact.converter.export_onnx", side_effect=AssertionError("must reuse")):
                self.assertEqual(ArtifactManager().get_artifact(manifest, "onnx"), onnx)
            self.assertEqual(times, (source.stat().st_mtime_ns, onnx.stat().st_mtime_ns))
            # A changed export configuration invalidates the ONNX cache only.
            manifest.data["artifacts"]["onnx"]["opset_version"] = 14
            manager.get_artifact(manifest, "onnx")
            self.assertEqual(checksum(source), original)
            with source.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                manager.get_artifact(manifest, "onnx")

    def test_reverse_conversion_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported conversion: onnx -> pytorch"):
            default_registry().get("onnx", "pytorch")

    def test_validation_failure(self):
        with TemporaryDirectory() as directory:
            manifest = self.manifest(directory)
            with patch("onnxruntime.InferenceSession") as session:
                session.return_value.run.return_value = [np.full((1, 1000), 1e6, dtype=np.float32)]
                with self.assertRaisesRegex(RuntimeError, "ONNX artifact validation failed"):
                    validate_onnx(manifest, self.tiny_model(), "unused.onnx")


if __name__ == "__main__":
    unittest.main()

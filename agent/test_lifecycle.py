"""Regression tests: python -m unittest agent.test_lifecycle -v."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort

from .agent import Agent
from .manifest import Manifest
from .predictor import Predictor


class LifecycleTests(unittest.TestCase):
    def test_switch_models_and_reuse_session(self):
        base = Manifest.load("manifests/resnet18_cpu.yaml").data
        agent = Agent()
        self.assertEqual(agent.get_status(), "CREATED")
        agent.start()
        self.assertFalse(agent.model_loaded)
        self.assertTrue(agent.supports(backend="onnxruntime", device="cpu"))
        self.assertFalse(agent.supports(device="gpu"))
        with TemporaryDirectory() as directory:
            try:
                for index, op in enumerate(("Identity", "Neg")):
                    path = Path(directory) / f"model_{index}.onnx"
                    graph = onnx.helper.make_graph(
                        [onnx.helper.make_node(op, ["input"], ["output"])], op,
                        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
                        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
                    )
                    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
                    model.ir_version = 8
                    onnx.save(model, path)
                    data = deepcopy(base)
                    data["source"]["path"] = str(path)
                    data["inputs"][0]["shape"] = ["batch_size", 2]
                    data["outputs"][0]["shape"] = ["batch_size", 2]
                    with patch("agent.predictor.ort.InferenceSession", wraps=ort.InferenceSession) as factory:
                        agent.load_model(Manifest(data))
                        x = np.array([[1, 2]], dtype=np.float32)
                        for _ in range(3):
                            np.testing.assert_array_equal(agent.predict(x)[0], x if index == 0 else -x)
                        self.assertEqual(factory.call_count, 1)
                    with patch.object(agent.predictor, "predict", side_effect=lambda x: self.assertEqual(agent.get_status(), "BUSY")):
                        agent.predict(x)
                    with self.assertRaisesRegex(RuntimeError, "already loaded"):
                        agent.load_model(Manifest(data))
                    self.assertEqual(agent.get_status(), "ERROR")
                    self.assertTrue(agent.model_loaded)
                    agent.unload_model()
                    self.assertEqual(agent.get_status(), "READY")
                    self.assertIsNone(agent.manifest)
            finally:
                agent.stop()
        self.assertEqual(agent.get_status(), "STOPPED")

    def test_missing_provider_and_no_fallback(self):
        agent = Agent("configs/agent_gpu.yaml")
        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider is not available for agent_gpu"):
                agent.start()
        self.assertEqual(agent.get_status(), "ERROR")
        agent.unload_model()
        self.assertEqual(agent.get_status(), "ERROR")
        agent.stop()
        predictor = Predictor("CUDAExecutionProvider")
        with patch("agent.predictor.ort.get_available_providers", return_value=["CUDAExecutionProvider"]), patch("agent.predictor.ort.InferenceSession") as factory:
            factory.return_value.get_providers.return_value = ["CPUExecutionProvider"]
            with self.assertRaisesRegex(RuntimeError, "refusing fallback"):
                predictor.load_model("unused.onnx")
            self.assertIsNone(predictor._session)

    def test_profile_error_preserves_previous_report(self):
        agent = Agent.from_manifest("manifests/resnet18_cpu.yaml")
        agent.start()
        try:
            agent.load_model()
            with TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                path.write_text('{"old": true}', encoding="utf-8")
                def fail(command, **kwargs):
                    self.assertEqual(agent.get_status(), "BUSY")
                    self.assertIn("providers=CPUExecutionProvider,strict_provider=true", command)
                    return subprocess.CompletedProcess(command, 1, "", "test failure")
                with patch("agent.profilers.subprocess.run", side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        agent.profile(path)
                self.assertEqual(agent.get_status(), "ERROR")
                self.assertEqual(json.loads(path.read_text()), {"old": True})
        finally:
            agent.stop()


if __name__ == "__main__":
    unittest.main()

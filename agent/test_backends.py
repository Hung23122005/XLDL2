"""Backend contracts and profiling regression tests."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .agent import Agent
from .manifest import Manifest
from .predictors import create_predictor
from .predictors.base import BasePredictor
from .profilers import BasicProfiler, PRoofProfiler


class BackendTests(unittest.TestCase):
    def test_factory_and_manifest_compatibility(self):
        for backend, device, provider in (
            ("onnxruntime", "cpu", "CPUExecutionProvider"),
            ("onnxruntime", "gpu", "CUDAExecutionProvider"),
            ("pytorch", "cpu", "cpu"), ("pytorch", "gpu", "cuda"),
        ):
            self.assertIsInstance(create_predictor(backend, device, provider), BasePredictor)
        for backend, manifest in (("pytorch", "onnx"), ("onnx", "pytorch")):
            agent = Agent(f"configs/agent_{backend}_cpu.yaml")
            agent.start()
            try:
                with self.assertRaisesRegex(ValueError, "is not supported by agent backend"):
                    agent.load_model(f"manifests/resnet18_{manifest}.yaml")
                self.assertEqual(agent.get_status(), "ERROR")
                self.assertFalse(agent.model_loaded)
            finally:
                agent.stop()

    def test_torch_model_reuse_output_and_no_grad(self):
        manifest = Manifest.load("manifests/resnet18_pytorch.yaml")
        predictor = create_predictor("pytorch", "cpu", "cpu")
        with self.assertRaises(RuntimeError):
            predictor.predict(np.zeros((1, 3, 224, 224)))
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 1000))
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        artifact_path = Path(directory.name) / "source.pth"
        torch.save(model.state_dict(), artifact_path)
        with patch("torchvision.models.resnet18", return_value=model) as create:
            predictor.load_model(manifest, artifact_path=artifact_path)
            self.assertFalse(model.training)
            for _ in range(3):
                output = predictor.predict(np.ones((2, 3, 2, 2), dtype=np.float64))
                self.assertEqual(output[0].shape, (2, 1000))
                self.assertEqual(output[0].dtype, np.float32)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                predictor.load_model(manifest)
            self.assertEqual(create.call_count, 1)
        predictor.unload_model()
        self.assertIsNone(predictor.model)

    def test_torch_cuda_unavailable_does_not_load(self):
        predictor = create_predictor("pytorch", "gpu", "cuda")
        with patch("torch.cuda.is_available", return_value=False), patch("torchvision.models.resnet18") as create:
            with self.assertRaisesRegex(RuntimeError, "refusing CPU fallback"):
                predictor.load_model(Manifest.load("manifests/resnet18_pytorch.yaml"))
            create.assert_not_called()

    def test_basic_profiler_counts_batch_and_report(self):
        agent = Agent("configs/agent_pytorch_cpu.yaml", Manifest.load("manifests/resnet18_pytorch.yaml"))
        self.assertEqual(agent._default_report_path(), "reports/resnet18_pytorch_cpu.json")
        with TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch.object(agent.predictor, "predict") as predict, patch.object(agent.predictor, "synchronize") as sync, patch("agent.profilers.time.perf_counter", side_effect=range(100)), patch("agent.profilers.subprocess.run") as subprocess_run:
                BasicProfiler().profile(agent, report, batch_size=2)
                self.assertEqual(predict.call_count, 60)
                self.assertEqual(sync.call_count, 100)
                self.assertEqual(predict.call_args.args[0].shape, (2, 3, 224, 224))
                subprocess_run.assert_not_called()
            data = json.loads(report.read_text())
            self.assertEqual(data["batch_size"], 2)
            self.assertEqual(data["performance"]["avg_latency_ms"], 1000)
            self.assertEqual(data["performance"]["throughput_samples_per_sec"], 2)

    def test_proof_publishes_fresh_report(self):
        agent = Agent("configs/agent_onnx_cpu.yaml")
        agent.model_path = Manifest.load("manifests/resnet18_onnx.yaml").model_path
        with TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text('{"old":true}')
            def run(command, **kwargs):
                from subprocess import CompletedProcess
                Path(command[command.index("-f") + 1]).write_text('{"fresh":true}')
                return CompletedProcess(command, 0, "", "")
            with patch("agent.profilers.subprocess.run", side_effect=run):
                PRoofProfiler().profile(agent, report, 1)
            self.assertEqual(json.loads(report.read_text()), {"fresh": True})


if __name__ == "__main__":
    unittest.main()

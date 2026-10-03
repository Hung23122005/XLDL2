"""A configurable execution environment with an explicit model lifecycle."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from pathlib import Path
import sys

# Direct script execution also needs access to the sibling artifact package.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact import ArtifactManager

if __package__:
    from .config import AgentConfig, REPO_ROOT
    from .hardware import detect_hardware
    from .manifest import Manifest, resolve_repo_path
    from .predictors import create_predictor
    from .profilers import create_profiler
else:
    from config import AgentConfig, REPO_ROOT
    from hardware import detect_hardware
    from manifest import Manifest, resolve_repo_path
    from predictors import create_predictor
    from profilers import create_profiler


class AgentStatus(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    BUSY = "BUSY"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class Agent:
    def __init__(self, config=None, manifest=None):
        # Keep Agent() and Agent(manifest) useful for existing CPU callers.
        if isinstance(config, Manifest):
            if manifest is not None:
                raise ValueError("Manifest provided twice.")
            manifest, config = config, None
        self.config = config if isinstance(config, AgentConfig) else AgentConfig.load(
            config or "configs/agent_cpu.yaml"
        )
        self.agent_id = self.config.agent_id
        self.backend = self.config.backend
        self.device = self.config.device
        self.provider = self.config.provider
        self.status = AgentStatus.CREATED
        self.manifest = manifest
        self.predictor = create_predictor(self.backend, self.device, self.provider)
        self.profiler = create_profiler(self.backend)
        self.artifact_manager = ArtifactManager()
        self.artifact_metadata = {}
        self.model_loaded = False
        self.model_path = None
        self.hardware_info = {}
        self._runtime_ready = False

    @classmethod
    def from_manifest(cls, manifest_path, agent_config=None):
        """Construct with a manifest; call start(), then load_model()."""
        return cls(config=agent_config, manifest=Manifest.load(manifest_path))

    @contextmanager
    def _operation(self):
        try:
            if self.status != AgentStatus.READY:
                raise RuntimeError(f"Agent {self.agent_id} is {self.status.value}; call start() first.")
            self.status = AgentStatus.BUSY
            yield
        except BaseException:
            self.status = AgentStatus.ERROR
            raise
        else:
            self.status = AgentStatus.READY
            self._runtime_ready = True

    def start(self, model_path=None):
        """Validate runtime without loading a model.

        Passing model_path retains the legacy start(path) convenience.
        """
        try:
            if self.status not in (AgentStatus.CREATED, AgentStatus.STOPPED):
                raise RuntimeError("Agent already started; call stop() before restarting.")
            self.hardware_info = detect_hardware(self.device, self.backend)
            available = self.hardware_info["available_providers"]
            if self.provider not in available:
                raise RuntimeError(
                    f"{self.provider} is not available for {self.agent_id}.\n"
                    f"Available providers: {available}"
                )
            self.status = AgentStatus.READY
            self._runtime_ready = True
            print(f"Agent start: {self.agent_id} (READY)")
            if model_path is not None:
                self.load_model(model_path=model_path)
        except Exception:
            self.status = AgentStatus.ERROR
            raise

    def load_model(self, manifest=None, *, model_path=None):
        with self._operation():
            if self.model_loaded:
                raise RuntimeError("Model already loaded. Call unload_model() first.")
            if manifest is not None and model_path is not None:
                raise ValueError("Provide either manifest or model_path, not both.")
            selected = manifest if isinstance(manifest, Manifest) else (
                Manifest.load(manifest) if manifest is not None else self.manifest
            )
            if model_path is not None:
                selected = None
                path = Path(model_path).expanduser().resolve()
            elif selected is not None:
                path = selected.model_path
            else:
                raise ValueError("Provide a model manifest or model_path.")
            compatible = selected is None or self.supports(framework=selected.framework)
            if selected is not None and selected.is_logical:
                compatible = self.artifact_manager.supports(selected, self.predictor.artifact_format)
            if not compatible:
                raise ValueError(
                    f"Manifest framework '{selected.framework}' is not supported by agent backend '{self.backend}'."
                )
            metadata = {}
            if selected is not None and (selected.is_logical or selected.framework.lower() == "pytorch"):
                path = self.artifact_manager.get_artifact(selected, target=self.predictor.artifact_format)
                self.predictor.load_model(selected, artifact_path=path)
                metadata = self.artifact_manager.get_metadata(selected, self.predictor.artifact_format, path)
            else:
                self.predictor.load_model(selected if selected is not None else path)
                metadata = {
                    "model_id": selected.model_id if selected else path.stem,
                    "source_framework": selected.framework if selected else self.predictor.artifact_format,
                    "artifact_format": self.predictor.artifact_format, "artifact_path": str(path),
                    "generated_from": None,
                }
            self.artifact_metadata = metadata
            self.manifest = selected
            self.model_path = path
            self.model_loaded = True

    def _require_model(self):
        if not self.model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

    def predict(self, input_tensor):
        with self._operation():
            self._require_model()
            return self.predictor.predict(input_tensor)

    def generate_input(self):
        try:
            if self.manifest is None:
                raise RuntimeError("Input generation requires a model manifest.")
            return self.manifest.generate_input()
        except Exception:
            self.status = AgentStatus.ERROR
            raise

    def profile(self, output_path=None, batch_size=None):
        """Delegate measurement to the backend's profiler without reloading."""
        with self._operation():
            self._require_model()
            output_path = output_path if output_path is not None else (
                self.manifest.report_path if self.manifest and self.manifest.report_path else self._default_report_path()
            )
            batch_size = batch_size if batch_size is not None else (
                self.manifest.batch_size if self.manifest else 1
            )
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError("batch_size must be a positive integer.")
            report_path = resolve_repo_path(output_path)
            protected_paths = {self.model_path}
            if self.manifest is not None:
                protected_paths.update(resolve_repo_path(spec["path"]) for spec in self.manifest.data.get("artifacts", {}).values())
            if report_path in protected_paths:
                raise ValueError("Report path must differ from model path.")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            print("Profile", flush=True)
            self.profiler.profile(self, report_path, batch_size)
            if not report_path.is_file():
                raise RuntimeError(f"Profiler did not create report: {report_path}")
            print(f"Profile completed: {report_path}")
            return report_path

    def unload_model(self, *, preserve_error=False):
        # Cleanup remains possible after an inference/profiling error.
        keep_error = preserve_error and self.status == AgentStatus.ERROR
        try:
            if self.status == AgentStatus.BUSY:
                raise RuntimeError("Cannot unload while Agent is BUSY.")
            self.predictor.unload_model()
            self.model_loaded = False
            self.model_path = None
            self.manifest = None
            self.artifact_metadata = {}
            if self._runtime_ready and not keep_error:
                self.status = AgentStatus.READY
        except BaseException:
            self.status = AgentStatus.ERROR
            raise

    def stop(self):
        self.unload_model()
        self._runtime_ready = False
        self.status = AgentStatus.STOPPED
        print(f"Agent stop: {self.agent_id} (STOPPED)")

    def get_status(self):
        return self.status.value

    def supports(self, backend=None, device=None, framework=None):
        """Match configured capabilities; start() verifies actual availability."""
        return ((backend is None or backend == self.backend)
                and (device is None or device == self.device)
                and (framework is None or self.predictor.supports_framework(framework)))

    def _default_report_path(self):
        import re
        name = self.manifest.name if self.manifest else self.model_path.stem
        name = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_") or "model"
        backend = self.predictor.frameworks[0]
        return f"reports/{name}_{backend}_{self.device}.json"

    def get_info(self):
        return {
            "agent_id": self.agent_id, "backend": self.backend, "device": self.device,
            "provider": self.provider, "status": self.get_status(),
            "model_loaded": self.model_loaded,
            "model_path": str(self.model_path) if self.model_path else None,
            "hardware": deepcopy(self.hardware_info),
            "artifact": deepcopy(self.artifact_metadata),
            "capabilities": {
                **self.predictor.get_info(), "backend": self.backend, "device": self.device,
                "model_formats": [self.predictor.artifact_format],
                "supported_actions": ["profile"],
                "provider_available": self.provider in self.hardware_info.get("available_providers", []),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Run a configured Agent with a model manifest.")
    parser.add_argument("--agent-config", default="configs/agent_cpu.yaml")
    parser.add_argument("--manifest", default="manifests/resnet18.yaml")
    parser.add_argument("--report-path", help="Override the manifest report path for this Agent.")
    args = parser.parse_args()
    agent = Agent(config=args.agent_config)
    try:
        agent.start()
        agent.load_model(args.manifest)
        print(f"Manifest loaded: {agent.manifest.name} {agent.manifest.version}")
        print(f"Backend: {agent.backend}")
        print(f"Device: {agent.device}")
        print(f"Provider: {agent.provider}")
        x = agent.generate_input()
        for index in range(1, 4):
            outputs = agent.predict(x)
            print(f"Predict {index}: output shapes = {[output.shape for output in outputs]}")
        agent.profile(output_path=args.report_path)
        agent.unload_model()
    finally:
        agent.stop()


if __name__ == "__main__":
    main()

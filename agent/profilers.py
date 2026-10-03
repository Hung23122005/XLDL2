"""Separate profiling strategies.

PRoof handles ONNX Runtime.
PyTorchProfiler handles PyTorch with:
- model-level latency
- throughput
- per-layer timing
- operator-level torch.profiler statistics
- memory usage
- FLOPs where available
"""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

if __package__:
    from .config import REPO_ROOT
else:
    from config import REPO_ROOT


# ============================================================
# ONNX / PRoof profiler
# ============================================================

class PRoofProfiler:
    def profile(self, agent, report_path, batch_size):
        # Publish only a fresh, valid report;
        # a failed run cannot reuse an old one.
        with TemporaryDirectory(
            prefix="proof_",
            dir=report_path.parent,
        ) as directory:

            temporary_report = (
                Path(directory) / "report.json"
            )

            command = [
                sys.executable,
                str(REPO_ROOT / "main.py"),

                "-B",
                agent.backend,

                "-m",
                str(agent.model_path),

                "-b",
                str(batch_size),

                "-f",
                str(temporary_report),

                "-v",

                "-s",
                "model",

                "-o",
                (
                    f"providers={agent.provider},"
                    "strict_provider=true"
                ),
            ]

            try:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )

            except OSError as error:
                raise RuntimeError(
                    f"Could not launch PRoof: {error}"
                ) from error

            details = "\n".join(
                part
                for part in (
                    result.stdout,
                    result.stderr,
                )
                if part
            )

            if (
                result.returncode != 0
                or
                "report file (if any) maybe incomplete"
                in details
            ):
                raise RuntimeError(
                    "PRoof profiling failed "
                    f"(exit code {result.returncode}):\n"
                    f"{details}"
                )

            if not temporary_report.is_file():
                raise RuntimeError(
                    "PRoof did not create report: "
                    f"{report_path}\n"
                    f"{details}"
                )

            try:
                with temporary_report.open(
                    encoding="utf-8"
                ) as stream:
                    report = json.load(stream)

            except (OSError, ValueError) as error:
                raise RuntimeError(
                    "Invalid PRoof JSON report: "
                    f"{error}"
                ) from error

            report.update(
                agent.artifact_metadata
            )

            temporary_report.write_text(
                json.dumps(
                    report,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temporary_report.replace(
                report_path
            )


# ============================================================
# PyTorch detailed profiler
# ============================================================

class PyTorchProfiler:
    """Detailed PyTorch profiling.

    Measures:
    - predict API latency
    - throughput
    - individual layer latency
    - torch operator CPU/CUDA time
    - operator memory usage
    - FLOPs where torch.profiler can estimate them
    """

    warmup_iterations = 10
    benchmark_iterations = 50

    def profile(
        self,
        agent,
        report_path,
        batch_size,
    ):
        if agent.manifest is None:
            raise RuntimeError(
                "PyTorchProfiler requires "
                "a model manifest."
            )

        predictor = agent.predictor

        if predictor is None:
            raise RuntimeError(
                "PyTorchProfiler requires "
                "a loaded predictor."
            )

        if predictor.model is None:
            raise RuntimeError(
                "PyTorch model is not loaded."
            )

        if predictor.device is None:
            raise RuntimeError(
                "PyTorch device is not initialized."
            )

        import torch

        # ====================================================
        # Generate input from manifest
        # ====================================================

        x = agent.manifest.generate_input(
            batch_size=batch_size
        )

        if (
            not x.shape
            or x.shape[0] != batch_size
        ):
            raise ValueError(
                "Manifest input batch dimension "
                "does not match profiling batch_size."
            )

        tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=predictor.device,
        )

        model = predictor.model
        device = predictor.device

        model.eval()

        # ====================================================
        # Warmup
        # ====================================================

        with torch.no_grad():
            for _ in range(
                self.warmup_iterations
            ):
                model(tensor)

        predictor.synchronize()

        # ====================================================
        # Model-level benchmark
        # ====================================================

        latencies_seconds = []

        with torch.no_grad():
            for _ in range(
                self.benchmark_iterations
            ):
                predictor.synchronize()

                start = time.perf_counter()

                model(tensor)

                predictor.synchronize()

                end = time.perf_counter()

                latencies_seconds.append(
                    end - start
                )

        average_seconds = (
            sum(latencies_seconds)
            / len(latencies_seconds)
        )

        latencies_ms = [
            value * 1000.0
            for value in latencies_seconds
        ]

        throughput = (
            batch_size
            / average_seconds
        )

        # ====================================================
        # Layer-level timing with hooks
        # ====================================================

        layer_rows = []
        layer_starts = {}
        handles = []

        def make_pre_hook(name):
            def hook(module, inputs):
                predictor.synchronize()

                layer_starts[name] = (
                    time.perf_counter()
                )

            return hook

        def make_post_hook(name):
            def hook(
                module,
                inputs,
                output,
            ):
                predictor.synchronize()

                start_time = (
                    layer_starts.get(name)
                )

                if start_time is None:
                    return

                latency_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000.0

                row = {
                    "name": name,
                    "type":
                        module
                        .__class__
                        .__name__,
                    "latency_ms":
                        latency_ms,
                }

                # Parameter count
                try:
                    row[
                        "parameter_count"
                    ] = sum(
                        parameter.numel()
                        for parameter
                        in module.parameters(
                            recurse=False
                        )
                    )
                except Exception:
                    row[
                        "parameter_count"
                    ] = None

                # Output shape
                try:
                    if isinstance(
                        output,
                        torch.Tensor,
                    ):
                        row[
                            "output_shape"
                        ] = list(
                            output.shape
                        )

                    elif (
                        isinstance(
                            output,
                            (
                                list,
                                tuple,
                            ),
                        )
                        and output
                        and isinstance(
                            output[0],
                            torch.Tensor,
                        )
                    ):
                        row[
                            "output_shape"
                        ] = list(
                            output[0].shape
                        )

                except Exception:
                    pass

                layer_rows.append(
                    row
                )

            return hook

        # Only leaf modules.
        # Avoid recording parent modules
        # that contain children.
        for name, module in (
            model.named_modules()
        ):
            if not name:
                continue

            if any(
                True
                for _ in module.children()
            ):
                continue

            handles.append(
                module
                .register_forward_pre_hook(
                    make_pre_hook(name)
                )
            )

            handles.append(
                module
                .register_forward_hook(
                    make_post_hook(name)
                )
            )

        try:
            with torch.no_grad():
                model(tensor)

            predictor.synchronize()

        finally:
            for handle in handles:
                handle.remove()

        # ====================================================
        # torch.profiler operator profiling
        # ====================================================

        activities = [
            torch.profiler
            .ProfilerActivity
            .CPU
        ]

        if device.type == "cuda":
            activities.append(
                torch.profiler
                .ProfilerActivity
                .CUDA
            )

        profiler_kwargs = {
            "activities":
                activities,

            "record_shapes":
                True,

            "profile_memory":
                True,

            "with_stack":
                False,

            "with_flops":
                True,
        }

        with torch.profiler.profile(
            **profiler_kwargs
        ) as prof:

            with torch.no_grad():
                model(tensor)

            predictor.synchronize()

        operator_rows = []

        for event in prof.key_averages():
            row = {
                "name":
                    event.key,

                "calls":
                    event.count,

                "cpu_time_total_us":
                    event.cpu_time_total,

                "cpu_time_self_us":
                    event.self_cpu_time_total,

                "cpu_memory_bytes":
                    event.cpu_memory_usage,

                "flops":
                    getattr(
                        event,
                        "flops",
                        0,
                    )
                    or 0,
            }

            # Different torch versions expose
            # CUDA timing under slightly
            # different properties.
            cuda_total = 0
            cuda_self = 0
            cuda_memory = 0

            for attribute in (
                "device_time_total",
                "cuda_time_total",
            ):
                if hasattr(
                    event,
                    attribute,
                ):
                    try:
                        cuda_total = (
                            getattr(
                                event,
                                attribute,
                            )
                            or 0
                        )
                        break
                    except Exception:
                        pass

            for attribute in (
                "self_device_time_total",
                "self_cuda_time_total",
            ):
                if hasattr(
                    event,
                    attribute,
                ):
                    try:
                        cuda_self = (
                            getattr(
                                event,
                                attribute,
                            )
                            or 0
                        )
                        break
                    except Exception:
                        pass

            for attribute in (
                "device_memory_usage",
                "cuda_memory_usage",
            ):
                if hasattr(
                    event,
                    attribute,
                ):
                    try:
                        cuda_memory = (
                            getattr(
                                event,
                                attribute,
                            )
                            or 0
                        )
                        break
                    except Exception:
                        pass

            row[
                "device_time_total_us"
            ] = cuda_total

            row[
                "device_time_self_us"
            ] = cuda_self

            row[
                "device_memory_bytes"
            ] = cuda_memory

            operator_rows.append(
                row
            )

        # Sort slowest first
        layer_rows.sort(
            key=lambda row:
                row.get(
                    "latency_ms",
                    0,
                ),
            reverse=True,
        )

        operator_rows.sort(
            key=lambda row:
                max(
                    row.get(
                        "device_time_total_us",
                        0,
                    ),
                    row.get(
                        "cpu_time_total_us",
                        0,
                    ),
                ),
            reverse=True,
        )

        # ====================================================
        # Memory summary
        # ====================================================

        memory = {}

        if device.type == "cuda":
            try:
                memory = {
                    "allocated_bytes":
                        torch.cuda
                        .memory_allocated(
                            device
                        ),

                    "reserved_bytes":
                        torch.cuda
                        .memory_reserved(
                            device
                        ),

                    "max_allocated_bytes":
                        torch.cuda
                        .max_memory_allocated(
                            device
                        ),

                    "max_reserved_bytes":
                        torch.cuda
                        .max_memory_reserved(
                            device
                        ),
                }

            except Exception:
                memory = {}

        # ====================================================
        # Final report
        # ====================================================

        report = {
            **agent.artifact_metadata,

            "agent_id":
                agent.agent_id,

            "model":
                agent.manifest.name,

            "framework":
                agent.manifest.framework,

            "backend":
                agent.backend,

            "device":
                agent.device,

            "provider":
                agent.provider,

            "batch_size":
                batch_size,

            "profiler":
                "PyTorchProfiler",

            "measurement_scope":
                (
                    "model execution using "
                    "PyTorch native profiler "
                    "and module forward hooks"
                ),

            "warmup_iterations":
                self.warmup_iterations,

            "benchmark_iterations":
                self.benchmark_iterations,

            "weights":
                agent.manifest
                .data["source"]
                .get("weights"),

            "performance": {
                "avg_latency_ms":
                    average_seconds
                    * 1000.0,

                "throughput_samples_per_sec":
                    throughput,

                "latencies_ms":
                    latencies_ms,

                "min_latency_ms":
                    min(latencies_ms),

                "max_latency_ms":
                    max(latencies_ms),
            },

            "memory":
                memory,

            "layers":
                layer_rows,

            "operators":
                operator_rows,
        }

        # ====================================================
        # Atomic report publish
        # ====================================================

        with TemporaryDirectory(
            prefix="pytorch_",
            dir=report_path.parent,
        ) as directory:

            temporary_report = (
                Path(directory)
                / "report.json"
            )

            temporary_report.write_text(
                json.dumps(
                    report,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temporary_report.replace(
                report_path
            )


# ============================================================
# Factory
# ============================================================

def create_profiler(backend):
    profilers = {
        "onnxruntime":
            PRoofProfiler,

        "pytorch":
            PyTorchProfiler,
    }

    if backend not in profilers:
        raise ValueError(
            f"No profiler for backend: {backend}"
        )

    return profilers[backend]()
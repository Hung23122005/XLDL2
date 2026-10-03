"""Torchvision inference without automatic downloads or device fallback."""

from .base import BasePredictor


class PyTorchPredictor(BasePredictor):
    artifact_format = "pytorch"
    frameworks = ("pytorch",)

    def __init__(self, provider):
        self.provider = provider

        self.model = None
        self.device = None

    def load_model(
        self,
        manifest,
        artifact_path=None,
    ):
        if self.model is not None:
            raise RuntimeError(
                "Model already loaded. "
                "Call unload_model() first."
            )

        import torch

        if (
            not hasattr(
                manifest,
                "data",
            )
            or not self.supports_framework(
                manifest.framework
            )
        ):
            raise ValueError(
                "PyTorchPredictor requires "
                "a PyTorch model manifest."
            )

        source = manifest.data[
            "source"
        ]

        if (
            source["type"]
            != "torchvision"
            or
            source["model_name"]
            != "resnet18"
        ):
            raise ValueError(
                "Currently only torchvision "
                "resnet18 is supported."
            )

        if (
            self.provider == "cuda"
            and
            not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "PyTorch CUDA is not available; "
                "refusing CPU fallback."
            )

        self.device = torch.device(
            self.provider
        )

        from artifact import (
            ArtifactManager
        )

        from artifact.converter import (
            load_source_model
        )

        if artifact_path is None:
            artifact_path = (
                ArtifactManager()
                .get_artifact(
                    manifest,
                    "pytorch",
                )
            )

        model = load_source_model(
            manifest,
            artifact_path,
        )

        model.eval()
        model.to(self.device)

        self.model = model

        print("ModelLoad")

    def predict(
        self,
        input_tensor,
    ):
        if self.model is None:
            raise RuntimeError(
                "No model loaded. "
                "Call load_model() "
                "before predict()."
            )

        import torch

        tensor = torch.as_tensor(
            input_tensor,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            output = self.model(
                tensor
            )

        return [
            output
            .detach()
            .cpu()
            .numpy()
        ]

    def synchronize(self):
        if self.provider == "cuda":
            import torch

            torch.cuda.synchronize(
                self.device
            )

    def unload_model(self):
        if self.model is not None:
            self.model = None

            if self.provider == "cuda":
                import torch

                torch.cuda.empty_cache()

            self.device = None

            print("ModelUnload")
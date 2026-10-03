"""Common inference contract: one loaded model, list of NumPy outputs."""
from abc import ABC, abstractmethod


class BasePredictor(ABC):
    frameworks = ()

    @abstractmethod
    def load_model(self, manifest, artifact_path=None):
        pass

    @abstractmethod
    def predict(self, input_tensor):
        pass

    @abstractmethod
    def unload_model(self):
        pass

    def supports_framework(self, framework):
        return framework.lower() in self.frameworks

    def synchronize(self):
        """Wait for device work when the backend uses asynchronous execution."""

    def get_info(self):
        return {"frameworks": list(self.frameworks), "provider": self.provider}

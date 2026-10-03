"""Backend-specific predictors, loaded lazily by the factory."""
from .base import BasePredictor
from .factory import create_predictor

__all__ = ["BasePredictor", "create_predictor"]

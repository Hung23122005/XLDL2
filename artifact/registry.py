"""Directed conversion registry; reverse conversions are never inferred."""


class ConversionRegistry:
    def __init__(self):
        self._converters = {}

    def register(self, source, target, converter):
        self._converters[(source, target)] = converter

    def supports(self, source, target):
        return (source, target) in self._converters

    def get(self, source, target):
        try:
            return self._converters[(source, target)]
        except KeyError:
            raise RuntimeError(f"Unsupported conversion: {source} -> {target}") from None


def default_registry():
    from .converter import export_onnx
    registry = ConversionRegistry()
    registry.register("pytorch", "onnx", export_onnx)
    return registry

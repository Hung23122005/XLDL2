import torch
import torchvision.models as models

model = models.resnet18(weights=None)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet18_clean.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=13,
    dynamo=False
)

print("Saved resnet18_clean.onnx")
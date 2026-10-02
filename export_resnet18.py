import torch
import torchvision.models as models

model = models.resnet18(weights=None)
model.eval()

x = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    x,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=17,
    dynamo=False
)

print("Saved resnet18.onnx")
from classifier.models.cnn import GestureCNN
import torch

model = GestureCNN(num_classes=4)
x = torch.randn(8, 3, 128, 128)
out = model(x)
assert out.shape == (8, 4)
print(f"Output shape: {out.shape}")       # should be torch.Size([8, 4])
print(f"Params: {model.get_num_params()}") # should be roughly 8-9 million
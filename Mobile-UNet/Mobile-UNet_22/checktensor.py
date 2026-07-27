import torch
from Wildfire_models import UNet2D
model = UNet2D(in_channels=4, out_channels=2)
x = torch.randn(1, 4, 128, 128)
out = model(x)
print(out.shape)  # debe ser [1, 2, 128, 128]
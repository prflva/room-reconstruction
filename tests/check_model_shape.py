import sys; sys.path.insert(0, ".")
import torch
from src.models.unet import UNet

# проверка размерностей + количество параметров
model = UNet()
x = torch.randn(2, 3, 240, 320)
y = model(x)
print("вход:", x.shape, "| выход:", y.shape)       # ожидаем (2,1,240,320)
print("параметров:", sum(p.numel() for p in model.parameters()))
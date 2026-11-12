import sys; sys.path.insert(0, ".")
import torch
from src.data.nyu import read_pairs, split_by_scene, NYUDepth
from src.models.unet import UNet
from src.metrics.depth import depth_metrics

# обучение на одном батче

ROOT = "/kaggle/input/datasets/soumikrakshit/nyu-depth-v2/nyu_data/"
train, val = split_by_scene(read_pairs(ROOT, ROOT + "data/nyu2_train.csv"))
ds = NYUDepth(train[:8], size=(240, 320))
rgb, gt, mask = [torch.stack(t) for t in zip(*[ds[i] for i in range(8)])]   # 8 картинок одним батчем

device = "cuda"
rgb, gt, mask = rgb.to(device), gt.to(device), mask.to(device)
model = UNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(301):
    pred = torch.exp(model(rgb))                       # 1) прямой проход; exp делает глубину положительной
    loss = (pred - gt).abs()[mask].mean()              # 2) лосс L1 только по валидным пикселям
    opt.zero_grad()                                    # 3) обнуляем старые градиенты
    loss.backward()                                    # 4) считаем градиенты лосса по всем весам
    opt.step()                                         # 5) сдвигаем веса в сторону уменьшения лосса
    if step % 50 == 0:
        print(step, f"loss={loss.item():.4f}", depth_metrics(pred.detach(), gt, mask)["abs_rel"])
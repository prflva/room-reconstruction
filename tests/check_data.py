import sys
sys.path.insert(0, ".")
import torch
from src.data.dataset import read_pairs, split_by_scene, NYUDepth
from src.metrics.depth import depth_metrics

ROOT = "/kaggle/input/datasets/soumikrakshit/nyu-depth-v2/nyu_data/"
pairs = read_pairs(ROOT, ROOT + "data/nyu2_train.csv")
train, val = split_by_scene(pairs)
print("train:", len(train), "| val:", len(val))

ds = NYUDepth(val[:8], size=(240, 320))
rgb, depth, mask = ds[0]
print(rgb.shape, depth.shape, mask.shape)
print("rgb:", rgb.min().item(), rgb.max().item())
print("глубина, м:", depth.min().item(), depth.max().item(), "| валидных:", mask.float().mean().item())

batch = [ds[i] for i in range(8)]
gt = torch.stack([b[1] for b in batch]); m = torch.stack([b[2] for b in batch])
print("идеальное предсказание:", depth_metrics(gt.clone(), gt, m))
print("предсказание x1.3:     ", depth_metrics(gt * 1.3, gt, m))
import csv, random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DEPTH_MAX_8BIT = 10.0   # в train значение 255 соответствует 10 м (допущение, см. проверку)


def read_pairs(root, csv_path):
    """Читает CSV вида 'путь_к_фото,путь_к_глубине' и делает пути абсолютными."""
    pairs = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) == 2:
                pairs.append((str(Path(root) / row[0]), str(Path(root) / row[1])))
    return pairs


def split_by_scene(pairs, val_fraction=0.05, seed=123):
    """Делит train на train/val ПО СЦЕНАМ, а не по кадрам.
    Соседние кадры одной сцены почти одинаковы; если разнести их по разным выборкам,
    валидация станет нечестно лёгкой (утечка)."""
    scenes = sorted({Path(p[0]).parent.name for p in pairs})
    random.Random(seed).shuffle(scenes)
    val_scenes = set(scenes[: max(1, int(len(scenes) * val_fraction))])
    train = [p for p in pairs if Path(p[0]).parent.name not in val_scenes]
    val = [p for p in pairs if Path(p[0]).parent.name in val_scenes]
    return train, val


class NYUDepth(Dataset):
    def __init__(self, pairs, size=(240, 320), augment=False):
        self.pairs, self.size, self.augment = pairs, size, augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i): # чтение и нормализация rgb изображения
        rgb_path, depth_path = self.pairs[i]
        h, w = self.size                      # cv2.resize принимает (ширина, высота)

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # формат определяем по ТИПУ данных, а не по max() конкретной картинки
        depth = raw.astype(np.float32) * (DEPTH_MAX_8BIT / 255.0 if raw.dtype == np.uint8 else 1 / 1000.0)
        # для глубины только nearest: билинейная интерполяция выдумывает значения между
        # передним и задним планом, и в 3D появляются «летающие» точки
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:         # горизонтальное отражение: фото и глубина вместе
                rgb, depth = rgb[:, ::-1], depth[:, ::-1]
            # яркость и контраст меняем только у фото; глубина от освещения не зависит
            rgb = np.clip((rgb - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1), 0, 1)

        mask = (depth > 0.01) & (depth < 10.0)  # валидные пиксели; по ним считаем лосс и метрики
        return (torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1),   # (3,H,W)
                torch.from_numpy(np.ascontiguousarray(depth))[None],            # (1,H,W) метры
                torch.from_numpy(np.ascontiguousarray(mask))[None])             # (1,H,W) bool
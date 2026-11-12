import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import read_pairs, split_by_scene, NYUDepth
from src.metrics.depth import depth_metrics
from src.models.unet import UNet


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    # фиксируем случайность, чтобы запуск можно было повторить
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    out = Path(cfg["out_dir"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # --- данные: train для обучения, val для контроля (test пока не трогаем)
    root = cfg["data_root"]
    train_pairs, val_pairs = split_by_scene(read_pairs(root, root + "data/nyu2_train.csv"))
    size = tuple(cfg["size"])
    train_dl = DataLoader(NYUDepth(train_pairs, size, augment=True), batch_size=cfg["batch_size"],
                          shuffle=True, num_workers=cfg["num_workers"], drop_last=True)
    val_dl = DataLoader(NYUDepth(val_pairs, size, augment=False), batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=cfg["num_workers"])

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best, history = float("inf"), []
    for epoch in range(cfg["epochs"]):
        t0 = time.time()

        # --- обучение
        model.train()                                    # включаем режим обучения
        run = 0.0
        for rgb, gt, mask in train_dl:
            rgb, gt, mask = rgb.to(device), gt.to(device), mask.to(device)
            pred = torch.exp(model(rgb))
            loss = (pred - gt).abs()[mask].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()
        train_loss = run / len(train_dl)

        # --- проверка на val
        model.eval()                                     # режим оценки
        sums, n = {}, 0
        with torch.no_grad():                            # градиенты не нужны, экономим память
            for rgb, gt, mask in val_dl:
                rgb, gt, mask = rgb.to(device), gt.to(device), mask.to(device)
                m = depth_metrics(torch.exp(model(rgb)), gt, mask)
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + v * len(rgb)
                n += len(rgb)
        val = {k: v / n for k, v in sums.items()}

        rec = {"epoch": epoch + 1, "train_loss": train_loss, "sec": time.time() - t0, **val}
        history.append(rec)
        print(json.dumps({k: round(v, 4) for k, v in rec.items()}))

        if val["abs_rel"] < best:                        # запоминаем лучшую эпоху
            best = val["abs_rel"]
            torch.save(model.state_dict(), out / "best.pth")
        json.dump(history, open(out / "history.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    main(p.parse_args().config)
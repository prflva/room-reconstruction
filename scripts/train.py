import argparse, json, random, sys, time, shutil
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import NYUCached
from src.metrics.depth import depth_metrics
from src.models.unet import UNet

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    # фиксируем случайность, чтобы запуск можно было повторить
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    out = Path(cfg["out_dir"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out / "config.yaml")
    device = "cuda"

    # --- данные: train для обучения, val для контроля (test пока не трогаем)
    cache = cfg["cache_dir"]
    train_dl = DataLoader(NYUCached(cache, "train", augment=True), batch_size=cfg["batch_size"],
                          shuffle=True, num_workers=cfg["num_workers"], drop_last=True, pin_memory=True)
    val_dl = DataLoader(NYUCached(cache, "val"), batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    model = UNet().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, history = float("inf"), []
    scaler = torch.amp.GradScaler("cuda")


    for epoch in range(cfg["epochs"]):
        t0 = time.time()

        # --- обучение
        model.train()                                    # включаем режим обучения
        run = 0.0
        for rgb, gt, mask in train_dl:
            rgb = rgb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            gt, mask = (rgb.to(device, non_blocking=True), gt.to(device, non_blocking=True),
                             mask.to(device, non_blocking=True))
            with torch.autocast("cuda", dtype=torch.float16):
                raw = model(rgb)
            pred = torch.exp(raw.float().clamp(max=3.0))
            loss = (pred - gt).abs()[mask].mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # градиенты масштабируются
            scaler.step(opt)  # шаг оптимизатора (с обратным масштабом)
            scaler.update()
            run += loss.item()
        train_loss = run / len(train_dl)

        # --- проверка на val
        model.eval()                                     # режим оценки
        sums, n = {}, 0
        with torch.no_grad():                            # градиенты не нужны, экономим память
            for rgb, gt, mask in val_dl:
                rgb = rgb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                gt, mask = gt.to(device), mask.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    raw = model(rgb)
                pred = torch.exp(raw.float().clamp(max=3.0))
                m = depth_metrics(pred, gt, mask)
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + v * len(rgb)
                n += len(rgb)
        val = {k: v / n for k, v in sums.items()}

        rec = {"epoch": epoch + 1, "train_loss": train_loss, "sec": time.time() - t0, **val}
        history.append(rec)
        print(json.dumps({k: round(v, 4) for k, v in rec.items()}), flush=True)

        if val["abs_rel"] < best:                        # запоминаем лучшую эпоху
            best = val["abs_rel"]
            torch.save(model.state_dict(), out / "best.pth")
        json.dump(history, open(out / "history.json", "w"), indent=1)
        torch.save(model.state_dict(), out / "last.pth")
    shutil.make_archive("/kaggle/working/" + cfg["name"], "zip", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    main(p.parse_args().config)
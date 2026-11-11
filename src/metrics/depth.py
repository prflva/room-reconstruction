import torch

# метрики для оценки качества работы модели

def depth_metrics(pred, gt, mask, eps=1e-6):
    """pred, gt: тензоры (B,1,H,W) в МЕТРАХ; mask: (B,1,H,W) bool.
    Метрики считаются отдельно для каждого изображения, затем усредняются по батчу."""
    out = {k: [] for k in ["abs_rel", "rmse", "log10", "d1", "d2", "d3"]}
    for p, g, m in zip(pred, gt, mask):
        p, g = p[m].clamp(min=eps), g[m]          # берём только валидные пиксели
        if g.numel() == 0:                         # в кадре нет валидных пикселей
            continue
        ratio = torch.maximum(p / g, g / p)        # во сколько раз ошиблись (всегда >= 1)
        out["abs_rel"].append(((p - g).abs() / g).mean())
        out["rmse"].append(((p - g) ** 2).mean().sqrt())
        out["log10"].append((p.log10() - g.log10()).abs().mean())
        out["d1"].append((ratio < 1.25).float().mean())        # доля пикселей с ошибкой < 25%
        out["d2"].append((ratio < 1.25 ** 2).float().mean())
        out["d3"].append((ratio < 1.25 ** 3).float().mean())
    return {k: torch.stack(v).mean().item() for k, v in out.items()}
import sys, time
sys.path.insert(0, ".")
from pathlib import Path
import cv2, numpy as np
from src.data.dataset import read_pairs, split_by_scene

ROOT = "/kaggle/input/datasets/soumikrakshit/nyu-depth-v2/nyu_data/"
OUT = Path("/kaggle/working/cache"); OUT.mkdir(exist_ok=True)
H, W, STEP = 240, 320, 2          # STEP=2: берём каждый второй кадр

def build(pairs, name):
    n = len(pairs)
    rgb = np.lib.format.open_memmap(OUT / f"{name}_rgb.npy", mode="w+", dtype=np.uint8, shape=(n, H, W, 3))
    dep = np.lib.format.open_memmap(OUT / f"{name}_depth.npy", mode="w+", dtype=np.uint16, shape=(n, H, W))
    t0 = time.time()
    for i, (rp, dp) in enumerate(pairs):
        img = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        rgb[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        raw = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        # храним глубину в МИЛЛИМЕТРАХ в uint16: единый формат для train и test
        mm = raw.astype(np.float32) * (10000.0 / 255.0) if raw.dtype == np.uint8 else raw.astype(np.float32)
        dep[i] = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        if i % 2000 == 0:
            print(name, i, "/", n, round(time.time() - t0), "с", flush=True)
    rgb.flush(); dep.flush()

pairs = read_pairs(ROOT, ROOT + "data/nyu2_train.csv")
train, val = split_by_scene(pairs)
build(train[::STEP], "train")
build(val, "val")
test = read_pairs(ROOT, ROOT + "data/nyu2_test.csv")
build(test, "test")
print("готово")
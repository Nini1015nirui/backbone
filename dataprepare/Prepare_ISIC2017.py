# dataprepare/prepare_isic17_fixed.py
import os
from pathlib import Path

import numpy as np
import imageio.v2 as imageio      # Pip install imageio
import cv2                         # Pip install opencv-python
from tqdm import tqdm

# ---------- 参数 ----------
HEIGHT, WIDTH = 256, 256
DATA_ROOT = Path('./data/ISIC2017')       # ← 根据截图
IMG_DIR   = DATA_ROOT/'images'
MSK_DIR   = DATA_ROOT/'masks'
OUT_DIR   = Path('./data')                # 覆盖原有 .npy 就在 data/

# ---------- 收集文件 ----------
img_paths = sorted(IMG_DIR.glob('*.jpg'))
assert img_paths, f'❌ 找不到 JPG：{IMG_DIR}'

N = len(img_paths)
imgs = np.zeros((N, HEIGHT, WIDTH, 3), dtype=np.uint8)
msks = np.zeros((N, HEIGHT, WIDTH),     dtype=np.uint8)

# ---------- 读取 & 预处理 ----------
for i, p in enumerate(tqdm(img_paths, desc='Reading')):
    img = imageio.imread(p)
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    imgs[i] = img

    mask_p = MSK_DIR / f"{p.stem}_segmentation.png"
    assert mask_p.is_file(), f'Missing mask: {mask_p}'
    m = imageio.imread(mask_p)
    if m.ndim == 3: m = m[..., 0]
    m = cv2.resize(m, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    msks[i] = (m > 127).astype(np.uint8)  # 归一化 0/1

# ---------- 拆分 ----------
n_train, n_val = 1250, 150
train_i, val_i, test_i = np.split(imgs,  [n_train, n_train + n_val])
train_m, val_m, test_m = np.split(msks, [n_train, n_train + n_val])

# ---------- 保存 (覆盖) ----------
OUT_DIR.mkdir(exist_ok=True, parents=True)
np.save(OUT_DIR/'data_train.npy',  train_i)
np.save(OUT_DIR/'data_val.npy',    val_i)
np.save(OUT_DIR/'data_test.npy',   test_i)
np.save(OUT_DIR/'mask_train.npy',  train_m)
np.save(OUT_DIR/'mask_val.npy',    val_m)
np.save(OUT_DIR/'mask_test.npy',   test_m)

print('✅ 生成 .npy 完成，输出目录:', OUT_DIR.resolve())

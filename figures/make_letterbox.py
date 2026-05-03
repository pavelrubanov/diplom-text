"""Сгенерировать letterbox-версию leaf_crop для рисунка конвейера классификатора.

Берёт исходный канонический фрагмент, скейлит до 512×max(...) с сохранением
пропорций и кладёт по центру чёрного квадрата 512×512 --- ровно как
classifier/mobilenet.py:_fit_to_square.
"""
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).parent / "121_170_0035_leaf_crop_0002.png"
DST = Path(__file__).parent / "121_170_0035_leaf_crop_0002_letterbox.png"
SIZE = 512

img = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
h, w = img.shape[:2]
scale = SIZE / max(h, w)
new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
out = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
y0, x0 = (SIZE - new_h) // 2, (SIZE - new_w) // 2
out[y0:y0 + new_h, x0:x0 + new_w] = resized
cv2.imwrite(str(DST), out)
print(f"saved: {DST} ({SIZE}x{SIZE}, scaled leaf {new_w}x{new_h})")

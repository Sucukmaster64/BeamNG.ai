

import cv2, numpy as np
p = "C:\\Users\\salih\\Desktop\\BeamNg.ai\\data\\labels\\1769523967041.png"
lbl = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print("shape:", lbl.shape, "dtype:", lbl.dtype)
print("min/max:", lbl.min(), lbl.max())
print("unique (first 20):", np.unique(lbl)[:20])

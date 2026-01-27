import os
import cv2
import numpy as np
from glob import glob
from collections import Counter

LABEL_DIR = "data/labels"

def main():
    paths = sorted(glob(os.path.join(LABEL_DIR, "*.png")))
    if not paths:
        print("No label files found in", LABEL_DIR)
        return

    total = Counter()
    maxv = 0
    n = 0

    for p in paths:
        lbl = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if lbl is None:
            continue
        n += 1
        maxv = max(maxv, int(lbl.max()))
        vals, cnts = np.unique(lbl, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            total[int(v)] += int(c)

    pixels = sum(total.values())
    print("Files read:", n)
    print("Max label value:", maxv)
    print("Class distribution:")
    for k in sorted(total.keys()):
        print(f"  class {k}: {total[k] / pixels:.4f}")

if __name__ == "__main__":
    main()

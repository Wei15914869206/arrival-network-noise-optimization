# -*- coding: utf-8 -*-
"""
make_fig456.py — 把三权重轨迹图 PNG 转成论文用的 Figure4/5/6.pdf

源：result/traj_wN{0.0,0.5,1.0}_2d.png（300 dpi，与表 3 的 seed=0 单链解一致）
映射：w_N=0.0 → Figure4.pdf，w_N=0.5 → Figure5.pdf，w_N=1.0 → Figure6.pdf
"""

import os
import shutil

from PIL import Image

RESULT_DIR = "result"
PAPER_DIR = "aerospace-4517773"

# 300 dpi → PDF 物理尺寸 = 像素 / 300
MAPPING = [
    ("result/traj_wN0.0_2d.png", "Figure4.pdf"),
    ("result/traj_wN0.5_2d.png", "Figure5.pdf"),
    ("result/traj_wN1.0_2d.png", "Figure6.pdf"),
]


def main():
    for src, dst in MAPPING:
        im = Image.open(src)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

        out = os.path.join(RESULT_DIR, dst)
        im.save(out, "PDF", resolution=300.0)
        shutil.copyfile(out, os.path.join(PAPER_DIR, dst))
        print(f"{src}  ->  {dst}   ({im.width}x{im.height} px, 300 dpi)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
plot_multi_seed_convergence.py — 30 条 SA 链的收敛曲线图

数据源：result/multi_seed_beta090_trace.csv（不重跑算法）
画法：横轴迭代次数，纵轴目标函数值 F；30 条链逐代 best-so-far 的
      中位数曲线 + 四分位距带，另叠加全局最优单链。

输出：
  result/fig_multi_seed_convergence.pdf   矢量，期刊用
  result/fig_multi_seed_convergence.png   位图预览
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = "result"
TRACE = os.path.join(RESULT_DIR, "multi_seed_beta090_trace.csv")
FIG_PDF = os.path.join(RESULT_DIR, "fig_multi_seed_convergence.pdf")
FIG_PNG = os.path.join(RESULT_DIR, "fig_multi_seed_convergence.png")

df = pd.read_csv(TRACE)

pivot = df.pivot_table(index="gen", columns="seed", values="F_best")
gens = pivot.index.to_numpy()
q1 = pivot.quantile(0.25, axis=1).to_numpy()
q2 = pivot.quantile(0.50, axis=1).to_numpy()
q3 = pivot.quantile(0.75, axis=1).to_numpy()
lo = pivot.min(axis=1).to_numpy()
best_seed = int(pivot.iloc[-1].idxmin())
best_curve = pivot[best_seed].to_numpy()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "legend.fontsize": 7.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6.0 / 2.54))

ax.fill_between(gens, q1, q3, color="#9DC3E6", alpha=0.55, lw=0, zorder=2,
                label="IQR (30 seeds)")
ax.plot(gens, q2, color="#1F4E79", lw=1.1, zorder=3, label="Median")
ax.plot(gens, best_curve, color="#C00000", lw=0.8, ls="--", zorder=3,
        label=f"Best seed ({best_seed})")

ax.set_xlabel("Iteration")
ax.set_ylabel("Best-so-far objective $F$")
ax.set_xlim(0, int(gens.max()))
ax.set_ylim(lo[-1] - 0.006, max(q3[0], best_curve[0]) + 0.02)
ax.grid(True, color="0.88", lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.legend(loc="upper right", frameon=False, handlelength=1.6,
          borderaxespad=0.2)

fig.savefig(FIG_PDF)
fig.savefig(FIG_PNG, dpi=600)
plt.close(fig)

print(f"Median F* = {q2[-1]:.4f}  IQR = [{q1[-1]:.4f}, {q3[-1]:.4f}]  "
      f"best = {lo[-1]:.4f}")
print(f"Figure: {os.path.abspath(FIG_PDF)}")
print(f"        {os.path.abspath(FIG_PNG)}")

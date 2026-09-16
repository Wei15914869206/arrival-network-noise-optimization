# -*- coding: utf-8 -*-
"""
make_fig7_r2_8.py — 生成 R2-8 的新 Figure 7（双子图）

(a) 暴露人口 vs SEL 阈值：重算后的基线曲线 + 三个权重的固定轨迹事后重分类曲线
    + w_N=0.3 的完整重优化点（虚线、空心标记、IQR 误差条）
(b) 阈值特定改善率 R_N,θ：三个权重的重分类曲线 + w_N=0.3 的完整重优化点

配色：dataviz 验证调色板的第 1-3 色位（blue / orange / aqua），
      基线用中性灰（参考线，非类别序列），并以虚线做二次编码。
      该三色位在明暗两种模式下均通过 all-pairs CVD 检验（最差 ΔE 9.2）。
字体：与论文其它插图一致，使用 matplotlib 默认无衬线体。

输出：result/Figure7.pdf / result/Figure7.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULT_DIR = "result"
FIG7_CSV = os.path.join(RESULT_DIR, "fig7_data.csv")

# ── 调色板 ──
C_BASE = "#898781"        # 中性灰：基线（参考线）
C_W0 = "#2a78d6"          # blue   slot 1
C_W3 = "#eb6834"          # orange slot 2
C_W1 = "#1baf7a"          # aqua   slot 3
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

SERIES = [("w_N=0.0", C_W0), ("w_N=0.3", C_W3), ("w_N=1.0", C_W1)]
WCOL = {s: c for s, c in SERIES}


def load():
    rows = []
    with open(FIG7_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(dict(series=r["series"], thr=float(r["thr"]),
                             N=float(r["N"]) if r["N"] else None,
                             q1=float(r["N_q1"]) if r["N_q1"] else None,
                             q3=float(r["N_q3"]) if r["N_q3"] else None,
                             kind=r["kind"]))
    return rows


def main():
    rows = load()
    base = {r["thr"]: r["N"] for r in rows if r["kind"] == "baseline"}
    recl = {(r["series"], r["thr"]): r["N"]
            for r in rows if r["kind"] == "reclassification"}
    reopt = {r["thr"]: r for r in rows if r["kind"] == "reoptimized"}
    thrs = sorted(base)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 7.0), dpi=300,
                                   gridspec_kw=dict(hspace=0.32))

    def style(ax):
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=9, length=3)
        ax.set_xticks(thrs)

    # ── (a) 暴露人口 vs 阈值 ──
    ax1.plot(thrs, [base[t] for t in thrs], color=C_BASE, linewidth=2.0,
             linestyle="--", marker="o", markersize=5, markerfacecolor="white",
             markeredgewidth=1.4, zorder=3, label="Published baseline")
    for name, col in SERIES:
        ys = [recl[(name, t)] / 1e6 for t in thrs]
        ax1.plot(thrs, ys, color=col, linewidth=2.0, marker="o",
                 markersize=5, zorder=4,
                 label=r"$w_N=%s$ (reclassified)" % name.split("=")[1])

    ro_t = [t for t in thrs if t in reopt]
    ax1.errorbar(ro_t, [reopt[t]["N"] / 1e6 for t in ro_t],
                 yerr=[[(reopt[t]["N"] - reopt[t]["q1"]) / 1e6 for t in ro_t],
                       [(reopt[t]["q3"] - reopt[t]["N"]) / 1e6 for t in ro_t]],
                 color=C_W3, linestyle="--", linewidth=1.6, marker="o",
                 markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                 capsize=3, elinewidth=1.0, zorder=5,
                 label=r"$w_N=0.3$ (full re-optimization)")

    # 选择性直接标注：曲线在 65 dB 以上高度重合，只在最分离的左端标注，
    # 其余靠图例识别（避免标注互相遮挡）。
    x0 = thrs[0]
    ax1.annotate("Published baseline", (x0, base[x0] / 1e6),
                 xytext=(4, 9), textcoords="offset points",
                 fontsize=8.5, color=INK2)
    ax1.annotate(r"$w_N=0.0$", (x0, recl[("w_N=0.0", x0)] / 1e6),
                 xytext=(4, 8), textcoords="offset points",
                 fontsize=8.5, color=C_W0)

    ax1.set_xlabel(r"$L_{AE}$ threshold (dB)", fontsize=10, color=INK)
    ax1.set_ylabel("Exposed population (millions)", fontsize=10, color=INK)
    ax1.set_ylim(0, 11.6)
    ax1.set_xlim(53.5, 77.0)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax1.legend(fontsize=8.5, frameon=False, loc="upper right",
               labelcolor=INK2, handlelength=2.2)
    ax1.text(-0.085, 1.045, "(a)", transform=ax1.transAxes,
             fontsize=11, fontweight="bold", color=INK)
    style(ax1)

    # ── (b) 阈值特定改善率 ──
    for name, col in SERIES:
        ys = [(base[t] - recl[(name, t)]) / base[t] * 100.0 for t in thrs]
        ax2.plot(thrs, ys, color=col, linewidth=2.0, marker="o",
                 markersize=5, zorder=4, label=r"$w_N=%s$" % name.split("=")[1])
    ys = [(base[t] - reopt[t]["N"]) / base[t] * 100.0 for t in ro_t]
    ax2.plot(ro_t, ys, color=C_W3, linestyle="--", linewidth=1.6,
             marker="o", markersize=5, markerfacecolor="white",
             markeredgewidth=1.4, zorder=5,
             label=r"$w_N=0.3$, re-optimized")

    ax2.axhline(0, color=AXIS, linewidth=1.0, zorder=2)
    ax2.set_xlabel(r"$L_{AE}$ threshold (dB)", fontsize=10, color=INK)
    ax2.set_ylabel(r"Improvement $R_{N,\theta}$ (%)", fontsize=10, color=INK)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(53.5, 77.0)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right",
               labelcolor=INK2, handlelength=2.2)
    ax2.text(-0.085, 1.045, "(b)", transform=ax2.transAxes,
             fontsize=11, fontweight="bold", color=INK)
    style(ax2)

    for p, ext in (("Figure7.pdf", "pdf"), ("Figure7.png", "png")):
        out = os.path.join(RESULT_DIR, p)
        fig.savefig(out, format=ext, bbox_inches="tight",
                    facecolor="white", dpi=300)
        print("-> %s" % os.path.abspath(out))
    plt.close(fig)


if __name__ == "__main__":
    main()

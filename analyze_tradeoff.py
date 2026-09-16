# -*- coding: utf-8 -*-
"""
analyze_tradeoff.py — R2-4 后处理：非支配集、膝点、归一化敏感性

输入：result/w_n_sweep_decoupled_per_run.csv
      result/w_n_sweep_decoupled_summary.csv

只使用固定解码器（A* 内层权重固定 0.5）下的 11 点 × 20 seeds 扫描数据。

做三件事：
  1. 每个 w_N 取 20 条链的中位数 → 11 个 (N, L) 点，判定支配关系
  2. 在非支配子集上求膝点（归一化后到首尾连线距离最大的点）
  3. 换三种归一化各求一次膝点，检查膝点位置是否移动

输出：
  result/tradeoff_report.txt      文字报告
  result/fig_pareto_frontier.png  非支配前沿图
"""

import os
import csv

import numpy as np

RESULT_DIR = "result"
PER_RUN_CSV = os.path.join(RESULT_DIR, "w_n_sweep_decoupled_per_run.csv")
SUMMARY_CSV = os.path.join(RESULT_DIR, "w_n_sweep_decoupled_summary.csv")
REPORT_TXT = os.path.join(RESULT_DIR, "tradeoff_report.txt")
FIG_PDF = os.path.join(RESULT_DIR, "fig_pareto_frontier.pdf")
FIG_PNG = os.path.join(RESULT_DIR, "fig_pareto_frontier.png")

N_REF = 6169767.0
L_REF = 829.2

_lines = []


def out(s=""):
    print(s, flush=True)
    _lines.append(s)


def read_summary():
    pts = []
    with open(SUMMARY_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pts.append(dict(
                w_N=float(r["w_N"]),
                n_valid=int(r["n_valid"]),
                N=float(r["N_median"]),
                N_q1=float(r["N_IQR"]),      # 仅用于画误差棒时换算，见下
                L=float(r["L_median"]),
                R_N=float(r["R_N_median"]),
                R_L=float(r["R_L_median"]),
            ))
    return sorted(pts, key=lambda p: p["w_N"])


def read_quartiles():
    """从逐链明细直接算每点的 Q1/Q3（summary 只存了 IQR 宽度）。"""
    acc = {}
    with open(PER_RUN_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["success"] != "True":
                continue
            w = float(r["w_N"])
            acc.setdefault(w, ([], []))
            acc[w][0].append(float(r["N_exposed"]))
            acc[w][1].append(float(r["L_total"]))
    q = {}
    for w, (Ns, Ls) in acc.items():
        q1n, _, q3n = np.percentile(Ns, [25, 50, 75])
        q1l, _, q3l = np.percentile(Ls, [25, 50, 75])
        q[w] = (q1n, q3n, q1l, q3l)
    return q


def dominates(a, b):
    """a 支配 b：N 与 L 都不差，且至少一个严格更好。"""
    return (a["N"] <= b["N"] and a["L"] <= b["L"]
            and (a["N"] < b["N"] or a["L"] < b["L"]))


def nondominated_set(pts):
    nd, dom_by = [], {}
    for p in pts:
        killers = [q["w_N"] for q in pts if q is not p and dominates(q, p)]
        if killers:
            dom_by[p["w_N"]] = killers
        else:
            nd.append(p)
    return sorted(nd, key=lambda p: p["N"]), dom_by


def knee_point(front, nrm):
    """归一化后到首尾连线距离最大的点。front 已按 N 升序。"""
    if len(front) < 3:
        return None, []
    P = np.array([[nrm(p)[0], nrm(p)[1]] for p in front], dtype=float)
    a, b = P[0], P[-1]
    ab = b - a
    abn = np.linalg.norm(ab)
    if abn < 1e-12:
        return None, []
    rel = P - a
    d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / abn
    i = int(np.argmax(d))
    return front[i], d


def main():
    pts = read_summary()
    quart = read_quartiles()

    # ── 基线参照 ──
    base = dict(w_N=None, N=N_REF, L=L_REF)

    out("=" * 78)
    out("R2-4 权衡分析：固定 A* 解码器（内层权重 = 0.5）")
    out("=" * 78)
    out(f"基线: N0 = {N_REF:,.0f}   L0 = {L_REF} km")
    out("")
    out(f"{'w_N':>5} {'n':>3} {'N (median)':>13} {'N IQR':>9} "
        f"{'L (median)':>11} {'L IQR':>7} {'R_N%':>7} {'R_L%':>7}  支配")
    out("-" * 78)

    nd, dom_by = nondominated_set(pts)
    nd_wn = {p["w_N"] for p in nd}

    for p in pts:
        flag = "" if p["w_N"] in nd_wn else f"被 {dom_by[p['w_N']]} 支配"
        q1n, q3n, q1l, q3l = quart[p["w_N"]]
        out(f"{p['w_N']:>5.1f} {p['n_valid']:>3} {p['N']:>13,.0f} "
            f"{q3n - q1n:>9,.0f} {p['L']:>11.1f} "
            f"{q3l - q1l:>7.1f} {p['R_N']:>7.2f} {p['R_L']:>7.2f}  {flag}")
    out("-" * 78)

    out("")
    out(f"非支配集（{len(nd)} 个点）：")
    for p in nd:
        out(f"  w_N = {p['w_N']:.1f}   N = {p['N']:>11,.0f}   L = {p['L']:>7.1f} km   "
            f"R_N = {p['R_N']:+.2f}%   R_L = {p['R_L']:+.2f}%")

    # ── 同时优于基线的窗口 ──
    wplus = [p["w_N"] for p in pts if p["R_N"] > 0 and p["R_L"] > 0]
    out("")
    out(f"W+ = {{w_N : R_N > 0 且 R_L > 0}} = {wplus}")
    if wplus:
        out(f"     连续区间: [{min(wplus):.1f}, {max(wplus):.1f}]")

    # ── 膝点：三种归一化 ──
    nimax = max(p["N"] for p in nd)
    nimin = min(p["N"] for p in nd)
    limax = max(p["L"] for p in nd)
    limin = min(p["L"] for p in nd)

    schemes = {
        "S1 基线归一 (N/N0, L/L0)":
            lambda p: (p["N"] / N_REF, p["L"] / L_REF),
        "S2 前沿极差归一 (min-max)":
            lambda p: ((p["N"] - nimin) / max(nimax - nimin, 1e-9),
                       (p["L"] - limin) / max(limax - limin, 1e-9)),
        "S3 改善率空间 (R_N, R_L)":
            lambda p: (p["R_N"], p["R_L"]),
    }

    out("")
    out("膝点（归一化后到前沿首尾连线距离最大的点）：")
    knee_by_scheme = {}
    for name, nrm in schemes.items():
        k, dist = knee_point(nd, nrm)
        knee_by_scheme[name] = k["w_N"] if k else None
        out(f"  {name:<34} → w_N = {k['w_N']:.1f}" if k
            else f"  {name:<34} → 无（非支配点少于 3 个）")
    stable = len(set(knee_by_scheme.values())) == 1
    out(f"  膝点是否对归一化稳定: {'是' if stable else '否'}"
        f"  （出现过的位置: {sorted(set(knee_by_scheme.values()))}）")

    # ── 落盘 ──
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n报告已保存: {os.path.abspath(REPORT_TXT)}")

    # ── 出图（风格对齐 plot_multi_seed_convergence.py：serif 8pt / 无标题 /
    #            去 top-right 边框 / PDF 矢量 + 600 dpi PNG）──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 10.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        })

        # Okabe-Ito 色盲友好配色
        COLOR = "#0072B2"
        KNEE_C = "#D55E00"
        FS = 10.0

        # 被支配的 w_N = 0.5 不入图（其支配关系只在表 6 中报告）
        pts_fig = [p for p in pts if abs(p["w_N"] - 0.5) > 1e-9]

        fig, ax = plt.subplots(figsize=(15.0 / 2.54, 9.6 / 2.54))

        def _draw_points(axes, plist, with_err):
            for p in plist:
                if with_err:
                    q1n, q3n, q1l, q3l = quart[p["w_N"]]
                    axes.errorbar(p["N"] / 1e6, p["L"],
                                  xerr=[[p["N"] / 1e6 - q1n / 1e6],
                                        [q3n / 1e6 - p["N"] / 1e6]],
                                  yerr=[[p["L"] - q1l], [q3l - p["L"]]],
                                  fmt="none", ecolor="0.82", elinewidth=0.4,
                                  capsize=1.2, zorder=1)

        def _draw_scatter(axes, plist, size=24):
            axes.plot([p["N"] / 1e6 for p in nd], [p["L"] for p in nd],
                      color="0.35", lw=0.7, zorder=2)
            for p in plist:
                axes.scatter(p["N"] / 1e6, p["L"], s=size, zorder=3,
                             facecolors=COLOR, edgecolors=COLOR,
                             linewidths=0.4, marker="o")

        _draw_points(ax, pts_fig, False)
        _draw_scatter(ax, pts_fig)

        ax.set_xlim(1.52, 2.92)
        ax.set_ylim(733, 1035)

        # ── 放大内嵌图：0.3–0.8 密集簇（无坐标，仅放大细节）──
        ZX = (1.685, 1.860)
        ZY = (776, 826)
        INS = [0.415, 0.395, 0.570, 0.560]      # 内嵌图在 ax 的轴分数位置
        axins = ax.inset_axes(INS)
        _draw_points(axins, pts_fig, False)
        _draw_scatter(axins, pts_fig, size=18)
        axins.set_xlim(*ZX)
        axins.set_ylim(*ZY)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.patch.set_facecolor("white")
        axins.patch.set_alpha(1.0)              # 不透明，遮住主图网格
        for s in axins.spines.values():         # 极细边界，仅用于界定面板
            s.set_visible(True)
            s.set_linewidth(0.4)
            s.set_color("0.75")

        # ── 逐点标注 w_N 值：贪心避让 + 细引出线 ──
        k = knee_by_scheme.get("S2 前沿极差归一 (min-max)")
        fig.canvas.draw()

        CANDS = [(6, 1), (-6, 1), (6, 7), (-6, 7), (6, -7), (-6, -7),
                 (11, 2), (-11, 2), (11, 10), (-11, 10), (11, -10),
                 (-11, -10), (0, 12), (0, -12), (17, 3), (-17, 3),
                 (0, 19), (0, -19), (17, 13), (-17, 13), (17, -13),
                 (-17, -13), (23, 3), (-23, 3), (0, 26), (0, -26)]

        def _ov(a, b):
            return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
                        or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])

        def _label(axes, order, fs):
            """贪心避让标注：order 为 [(w_N, 文本, 颜色), ...]，按给定顺序放置。

            约束：不与已放标签、数据点、坐标轴边界重叠。
            """
            trans = axes.transData
            bb = axes.get_window_extent(fig.canvas.get_renderer())
            lim = (bb.x0 + 2.0, bb.y0 + 2.0, bb.x1 - 2.0, bb.y1 - 2.0)
            by_w = {p["w_N"]: p for p in pts_fig}
            obstacles = []
            for p in pts_fig:
                xd, yd = trans.transform((p["N"] / 1e6, p["L"]))
                obstacles.append((xd - 2.8, yd - 2.8, 5.6, 5.6))
            # 前沿折线本身也作为障碍，避免标签压线
            ndx = [p["N"] / 1e6 for p in nd]
            ndy = [p["L"] for p in nd]
            for i in range(len(ndx) - 1):
                for t in np.linspace(0.0, 1.0, 24):
                    px = ndx[i] + t * (ndx[i + 1] - ndx[i])
                    py = ndy[i] + t * (ndy[i + 1] - ndy[i])
                    xd, yd = trans.transform((px, py))
                    obstacles.append((xd - 1.8, yd - 1.8, 3.6, 3.6))
            placed = []
            for w, txt, col in order:
                p = by_w[w]
                xd, yd = trans.transform((p["N"] / 1e6, p["L"]))
                tw = len(txt) * fs * 0.60
                th = fs * 1.15
                for dx, dy in CANDS:
                    # dx<0 时文本右对齐，向左铺开
                    bx = xd + dx if dx >= 0 else xd + dx - tw
                    box = (bx, yd + dy - th / 2.0, tw, th)
                    if box[0] < lim[0] or box[0] + box[2] > lim[2]:
                        continue
                    if box[1] < lim[1] or box[1] + box[3] > lim[3]:
                        continue
                    if any(_ov(box, b) for b in placed + obstacles):
                        continue
                    placed.append(box)
                    far = abs(dx) > 8 or abs(dy) > 8
                    axes.annotate(txt, (p["N"] / 1e6, p["L"]),
                                  textcoords="offset points", xytext=(dx, dy),
                                  fontsize=fs, color=col,
                                  ha="left" if dx >= 0 else "right",
                                  va="center", zorder=8,
                                  arrowprops=dict(arrowstyle="-", lw=0.4,
                                                  color="0.6", shrinkA=0,
                                                  shrinkB=2.5) if far else None)
                    break

        ktext = ("knee ($w_N$=%.1f)" % k) if k is not None else None

        # 膝点星标（主图 + 内嵌图），先画星再放标签
        if k is not None:
            kp = {p["w_N"]: p for p in pts_fig}[k]
            for axes in (ax, axins):
                axes.scatter([kp["N"] / 1e6], [kp["L"]], s=110, marker="*",
                             color=KNEE_C, edgecolors=KNEE_C, zorder=7)

        # 主图：只标开口大的点；密集簇交给内嵌图
        _label(ax, [(w, f"{w:.1f}", COLOR)
                    for w in (1.0, 0.9, 0.2, 0.1, 0.0)], FS)
        # 内嵌图：膝点优先，再标密集簇其余点
        _label(axins, [(k, ktext, KNEE_C)] +
               [(w, f"{w:.1f}", COLOR) for w in (0.4, 0.6, 0.7, 0.8)], FS)

        # 放大区域方框 + 一条引出线（方框右上角 → 内嵌图左下角）
        from matplotlib.patches import Rectangle
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0],
                               facecolor="none", edgecolor="0.55",
                               lw=0.5, zorder=4))
        ax.plot([ZX[1], x0 + INS[0] * (x1 - x0)],
                [ZY[1], y0 + INS[1] * (y1 - y0)],
                color="0.55", lw=0.5, zorder=4, clip_on=False)

        ax.set_xlabel("Exposed population $N$ (millions)")
        ax.set_ylabel("Total route length $L$ (km)")
        ax.grid(True, color="0.88", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        fig.savefig(FIG_PDF)
        fig.savefig(FIG_PNG, dpi=600)
        plt.close(fig)
        print(f"图已保存: {os.path.abspath(FIG_PDF)}")
        print(f"          {os.path.abspath(FIG_PNG)}")
    except Exception as e:
        print(f"出图失败（不影响分析结果）: {e}")


if __name__ == "__main__":
    main()

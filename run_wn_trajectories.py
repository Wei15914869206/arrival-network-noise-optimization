# -*- coding: utf-8 -*-
"""
run_wn_trajectories.py — 三个权重各跑一条 SA 链（并行），输出 2D 轨迹图

w_N ∈ {0.0, 0.5, 1.0}，每个权重 1 条链（seed=0, 150 代），3 个进程并行。
绘图复用 test.visualize_result（只保留 2D，删掉其附带的 3D 图）。
输出：result/traj_wN0.0_2d.png / traj_wN0.5_2d.png / traj_wN1.0_2d.png
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")

import test as _t

WN_VALUES = [0.0, 0.5, 1.0]
MAX_ITER = _t.NS_MAX_ITER
SEED = 0
N_WORKERS = len(WN_VALUES)
RESULT_DIR = "result"

# 每个 worker 进程的私有上下文
_G = {}


def log(*a):
    print(*a, flush=True)


def _init_worker():
    import test as t
    pop, _ = t.load_population_grid("population_data_square.xlsx")
    _G["t"] = t
    _G["pop"] = pop
    _G["nm"] = t.SELNoiseModel(pop)
    _G["pf"] = t.PathfinderConfig()
    _G["astar"] = t.AStar2DNoise(_G["nm"])


def _run_one(w_n):
    """单权重单链：SA 搜索 + 2D 轨迹图。返回汇总 dict。"""
    t = _G["t"]
    t.WN = float(w_n)   # 外层目标与 A* 内层噪声权重同步（ASTAR_WN_AUTO_SYNC=True）

    t0 = time.time()
    F_best, tree_best, _theta, info, _hist = t.neighborhood_search_run(
        t.numbered_points, t.IF_PT, t.P, _G["pf"], _G["nm"], _G["astar"],
        init_tree=None, max_iter=MAX_ITER, seed=SEED,
        T0=t.SA_T0, beta=t.SA_BETA, p_topo=t.P_TOPO_MOVE, log_every=0)
    runtime = time.time() - t0

    N = info.get("N_exposed")
    L = info.get("L_total")
    topo = t.tree_to_bracket(tree_best)

    prefix = os.path.join(RESULT_DIR, f"traj_wN{w_n}")
    t.visualize_result(
        info["decoded_tree"], info["all_paths"], _G["pf"], t.FAF, t.P,
        node_height_ranges=info["node_height_ranges"], pop=_G["pop"],
        title=f"w_N={w_n}   F={F_best:.4f}   N={N:,.0f}   L={L:.1f} km",
        fname_prefix=prefix)

    p3d = f"{prefix}_3d.png"          # 只要 2D 轨迹图
    if os.path.exists(p3d):
        os.remove(p3d)

    return dict(w_N=w_n, F=F_best, N=N, L=L, topo=topo,
                R_N=info.get("R_N"), R_L=info.get("R_L"),
                png=f"{prefix}_2d.png", runtime=runtime)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    t_wall = time.time()

    log("=" * 76)
    log(f"三权重轨迹图  w_N = {WN_VALUES}  |  每权重 1 条链 (seed={SEED}, "
        f"max_iter={MAX_ITER})  |  {N_WORKERS} 进程并行")
    log(f"SA: T0={_t.SA_T0} beta={_t.SA_BETA} p_topo={_t.P_TOPO_MOVE}  |  "
        f"A*: w_L={_t.ASTAR_W_L} (w_N 跟随 WN)")
    log(f"约束: CONSTRAIN_E1E2_FIRST={_t.CONSTRAIN_E1E2_FIRST}  "
        f"FAF_NO_CROSS={_t.FAF_NO_CROSS}  "
        f"INIT_THETA_MIDPOINT={_t.INIT_THETA_MIDPOINT}")
    log(f"基线: N0={_t.N_REF:,}  L0={_t.L_REF} km")
    log("=" * 76)

    out = []
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_run_one, w): w for w in WN_VALUES}
        for fut in as_completed(futs):
            r = fut.result()
            out.append(r)
            log(f"  [完成] w_N={r['w_N']:.1f}  F={r['F']:.4f}  "
                f"N={r['N']:,.0f} (R_N={r['R_N']:+.1f}%)  "
                f"L={r['L']:.1f}km (R_L={r['R_L']:+.1f}%)  "
                f"topo={r['topo']}  [{r['runtime']:.0f}s]")

    wall = time.time() - t_wall

    log("")
    log("─" * 76)
    log(f"{'w_N':>5}{'F*':>10}{'N':>14}{'L(km)':>10}{'R_N(%)':>10}"
        f"{'R_L(%)':>10}  topology")
    log("─" * 76)
    for r in sorted(out, key=lambda r: r["w_N"]):
        log(f"{r['w_N']:>5.1f}{r['F']:>10.4f}{r['N']:>14,.0f}{r['L']:>10.1f}"
            f"{r['R_N']:>+10.1f}{r['R_L']:>+10.1f}  {r['topo']}")
    log("─" * 76)
    log(f"并行耗时 {wall:.0f}s（{N_WORKERS} 进程）")
    log("")
    for r in sorted(out, key=lambda r: r["w_N"]):
        log(f"  {os.path.abspath(r['png'])}")


if __name__ == "__main__":
    main()

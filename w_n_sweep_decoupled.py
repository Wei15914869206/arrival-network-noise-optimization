# -*- coding: utf-8 -*-
"""
w_n_sweep_decoupled.py — R2-4: 固定 A* 内层噪声权重的 w_N 扫描

问题背景：原 w_n_sweep.py 中 ASTAR_W_N 自动跟随外层 WN，导致改变 w_N 同时
改变了标量目标与下层路径解码器，因此不是对固定问题的权扫描。

本脚本解耦：
    ASTAR_WN_AUTO_SYNC = False
    ASTAR_W_N 固定为 ASTAR_W_N_FIXED = 0.5（全程不变，含 A* 路径缓存键）
    外层 w_N 只进入目标函数 F = w_N·N/N₀ + (1−w_N)·L/L₀

实验协议（与 4.1.4 一致）：
    w_N ∈ {0.0, 0.1, ..., 1.0}         11 点
    每点独立 SA 链 20 条（seed = 0..19）
    T0 = 0.5, beta = 0.90, K_max = 150, p_topo = 0.25, step = 0.15
    E1/E2 首层汇聚约束关闭；FAF → P 禁越线约束关闭
    单链墙钟上限 900 s（超时判失败，避免病态拓扑下 A* 长时间空转）

输出：
    result/w_n_sweep_decoupled_per_run.csv    每条链一行
    result/w_n_sweep_decoupled_summary.csv    每个 w_N 一行（中位数/均值/标准差/IQR/极值）
"""

import os
import time
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ── 实验参数 ──
WN_VALUES = [round(i * 0.1, 1) for i in range(11)]
N_SEEDS = 20
MAX_ITER = 150
STEP = 0.15
P_TOPO = 0.25
T0 = 0.5
BETA = 0.90
ASTAR_W_N_FIXED = 0.5
TIME_LIMIT_S = 900
N_WORKERS = 12
MAX_INIT_TRIES = 20

POP_FILE = "population_data_square.xlsx"
RESULT_DIR = "result"
PER_RUN_CSV = os.path.join(RESULT_DIR, "w_n_sweep_decoupled_per_run.csv")
SUMMARY_CSV = os.path.join(RESULT_DIR, "w_n_sweep_decoupled_summary.csv")

# ── 约束开关：与 4.1.4 拓扑穷举一致 ──
CONSTRAIN_E1E2_FIRST = False   # False -> 拓扑空间 = 14 种保序二叉树
FAF_NO_CROSS = True            # True  -> 关闭 FAF→P 禁越线约束

_G = {}


def log(*a):
    print(*a, flush=True)


def _init_worker():
    """每个 worker 进程只执行一次：载入模型并锁定 A* 内层权重。"""
    import test as t
    t.CONSTRAIN_E1E2_FIRST = CONSTRAIN_E1E2_FIRST
    t.FAF_NO_CROSS = FAF_NO_CROSS
    t.ASTAR_WN_AUTO_SYNC = False          # ← 解耦的关键
    t.ASTAR_W_N = ASTAR_W_N_FIXED         # ← 内层权重固定
    pop, _ = t.load_population_grid(POP_FILE)
    _G["t"] = t
    _G["nm"] = t.SELNoiseModel(pop)
    _G["pf"] = t.PathfinderConfig()
    _G["astar"] = t.AStar2DNoise(_G["nm"])
    _G["NP"] = t.numbered_points
    _G["IFPT"] = t.IF_PT
    _G["P"] = t.P
    _G["trees"] = t.generate_convergence_structures()
    return len(_G["trees"])


def _evaluate(theta, tree):
    t = _G["t"]
    return t.evaluate_X(theta, tree, _G["NP"], _G["IFPT"], _G["P"],
                        _G["pf"], _G["nm"], _G["astar"])


def _run_chain(args):
    """单条 SA 链：固定 w_N，跑 MAX_ITER 代，带墙钟上限。"""
    w_n, seed = args
    t = _G["t"]
    t.WN = float(w_n)
    rng = np.random.RandomState(seed)

    t_start = time.time()

    # ── 初始解：随机拓扑 + Θ 从扇环中点出发，不可行则重采样 ──
    tree = theta = F = info = None
    for _ in range(MAX_INIT_TRIES):
        tree = copy.deepcopy(_G["trees"][rng.randint(0, len(_G["trees"]))])
        dim = t.count_internal_params(tree)
        theta = np.full(dim, 0.5) if t.INIT_THETA_MIDPOINT \
            else rng.uniform(0.0, 1.0, size=dim)
        F, info = _evaluate(theta, tree)
        if info.get("is_valid"):
            break
    init_valid = bool(info.get("is_valid"))

    def _snap(infov):
        return dict(N=infov.get("N_exposed"),
                    L=infov.get("L_total"),
                    R_N=infov.get("R_N"),
                    R_L=infov.get("R_L"))

    best = _snap(info)
    best_F = F
    best_topo = t.tree_to_bracket(tree)
    n_acc = 0
    n_worse = 0
    last_improve_gen = 0
    timed_out = False

    for gen in range(1, MAX_ITER + 1):
        if time.time() - t_start > TIME_LIMIT_S:
            timed_out = True
            break

        T = T0 * (BETA ** (gen - 1))

        do_topo = rng.random_sample() < P_TOPO
        if do_topo:
            high = [r for r in info.get("ratios", []) if r["rho"] > t.RHO_MAX]
            cands = t.targeted_rotations(tree, high) if high \
                else t.tree_rotations(tree)
            if not cands:
                do_topo = False
            else:
                tree_new = cands[rng.randint(0, len(cands))]
                theta_new = t._inherit_theta(tree, theta, tree_new, rng)
        if not do_topo:
            tree_new = tree
            theta_new = t.neighbor_theta(theta, rng, STEP)

        F_new, info_new = _evaluate(theta_new, tree_new)
        dF = F_new - F

        if t.sa_accept(dF, T, rng):
            tree, theta, F, info = tree_new, theta_new, F_new, info_new
            n_acc += 1
            if dF > 0:
                n_worse += 1
            if F < best_F:
                best_F = F
                best = _snap(info)
                best_topo = t.tree_to_bracket(tree)
                last_improve_gen = gen

    runtime = time.time() - t_start
    success = bool(best_F < 1.0e5)

    return dict(
        w_N=float(w_n), seed=seed,
        F_best=best_F if success else None,
        N_exposed=best["N"] if success else None,
        L_total=best["L"] if success else None,
        R_N=best["R_N"] if success else None,
        R_L=best["R_L"] if success else None,
        topology=best_topo,
        n_accepted=n_acc,
        cum_acceptance=n_acc / MAX_ITER,
        n_worse_accepted=n_worse,
        last_improve_gen=last_improve_gen,
        runtime_s=runtime,
        success=success,
        timed_out=timed_out,
        init_valid=init_valid,
    )


def _stats(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"),
                    median=float("nan"), q1=float("nan"), q3=float("nan"),
                    iqr=float("nan"), best=float("nan"), worst=float("nan"))
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(n=int(a.size), mean=float(a.mean()),
                std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                median=float(med), q1=float(q1), q3=float(q3),
                iqr=float(q3 - q1), best=float(a.min()), worst=float(a.max()))


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    t_wall = time.time()

    log("=" * 78)
    log("w_n_sweep_decoupled.py — 固定 A* 内层权重的 w_N 扫描")
    log(f"  内层 A* 噪声权重 FIXED = {ASTAR_W_N_FIXED} (AUTO_SYNC=False)")
    log(f"  外层 w_N = {WN_VALUES}")
    log(f"  seeds/点 = {N_SEEDS} | K_max = {MAX_ITER} | "
        f"T0 = {T0} | beta = {BETA} | 单链上限 = {TIME_LIMIT_S}s")
    log(f"  约束: E1E2_first={CONSTRAIN_E1E2_FIRST}, no_cross={FAF_NO_CROSS}")
    log(f"  workers = {N_WORKERS} | 总链数 = {len(WN_VALUES) * N_SEEDS}")
    log("=" * 78)

    tasks = [(w, s) for w in WN_VALUES for s in range(N_SEEDS)]
    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_run_chain, tk): tk for tk in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            if r["success"]:
                log(f"  [{done:>3}/{len(tasks)}] w_N={r['w_N']:<4} seed={r['seed']:<3} "
                    f"F={r['F_best']:.4f}  N={r['N_exposed']:>9,.0f}  "
                    f"L={r['L_total']:>6.1f}  topo={r['topology']:<18} "
                    f"t={r['runtime_s']:.0f}s{'  TIMEOUT' if r['timed_out'] else ''}")
            else:
                log(f"  [{done:>3}/{len(tasks)}] w_N={r['w_N']:<4} seed={r['seed']:<3} "
                    f"FAILED  t={r['runtime_s']:.0f}s"
                    f"{'  TIMEOUT' if r['timed_out'] else ''}")

    wall = time.time() - t_wall

    # ── 落盘：逐链明细 ──
    fields = ["w_N", "seed", "F_best", "N_exposed", "L_total", "R_N", "R_L",
              "topology", "n_accepted", "cum_acceptance", "n_worse_accepted",
              "last_improve_gen", "runtime_s", "success", "timed_out",
              "init_valid"]
    rows.sort(key=lambda r: (r["w_N"], r["seed"]))
    with open(PER_RUN_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([
                f"{r['w_N']:.1f}", r["seed"],
                f"{r['F_best']:.6f}" if r["F_best"] is not None else "",
                f"{r['N_exposed']:.0f}" if r["N_exposed"] is not None else "",
                f"{r['L_total']:.2f}" if r["L_total"] is not None else "",
                f"{r['R_N']:.3f}" if r["R_N"] is not None else "",
                f"{r['R_L']:.3f}" if r["R_L"] is not None else "",
                r["topology"] or "",
                r["n_accepted"], f"{r['cum_acceptance']:.6f}",
                r["n_worse_accepted"], r["last_improve_gen"],
                f"{r['runtime_s']:.1f}", r["success"], r["timed_out"],
                r["init_valid"]])

    # ── 落盘：每个 w_N 的汇总统计 ──
    summary_rows = []
    for w_n in WN_VALUES:
        g = [r for r in rows if r["w_N"] == w_n and r["success"]]
        st = {k: _stats([r[k] for r in g])
              for k in ("F_best", "N_exposed", "L_total", "R_N", "R_L",
                        "runtime_s")}
        topo_counts = {}
        for r in g:
            topo_counts[r["topology"]] = topo_counts.get(r["topology"], 0) + 1
        topo_best = max(topo_counts.items(), key=lambda kv: kv[1])[0] \
            if topo_counts else ""
        summary_rows.append(dict(
            w_N=w_n, n_valid=len(g),
            F_med=st["F_best"]["median"], F_mean=st["F_best"]["mean"],
            F_std=st["F_best"]["std"], F_iqr=st["F_best"]["iqr"],
            F_min=st["F_best"]["best"], F_max=st["F_best"]["worst"],
            N_med=st["N_exposed"]["median"], N_mean=st["N_exposed"]["mean"],
            N_std=st["N_exposed"]["std"], N_iqr=st["N_exposed"]["iqr"],
            N_min=st["N_exposed"]["best"], N_max=st["N_exposed"]["worst"],
            L_med=st["L_total"]["median"], L_mean=st["L_total"]["mean"],
            L_std=st["L_total"]["std"], L_iqr=st["L_total"]["iqr"],
            L_min=st["L_total"]["best"], L_max=st["L_total"]["worst"],
            RN_med=st["R_N"]["median"], RL_med=st["R_L"]["median"],
            RN_min=st["R_N"]["best"], RL_min=st["R_L"]["best"],
            topo_modal=topo_best,
            runtime_med=st["runtime_s"]["median"]))

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["w_N", "n_valid", "F_median", "F_mean", "F_std", "F_IQR",
                    "F_min", "F_max", "N_median", "N_mean", "N_std", "N_IQR",
                    "N_min", "N_max", "L_median", "L_mean", "L_std", "L_IQR",
                    "L_min", "L_max", "R_N_median", "R_L_median",
                    "R_N_best", "R_L_best", "topology_modal", "runtime_median_s"])
        for s in summary_rows:
            w.writerow([
                f"{s['w_N']:.1f}", s["n_valid"],
                f"{s['F_med']:.6f}", f"{s['F_mean']:.6f}", f"{s['F_std']:.6f}",
                f"{s['F_iqr']:.6f}", f"{s['F_min']:.6f}", f"{s['F_max']:.6f}",
                f"{s['N_med']:.0f}", f"{s['N_mean']:.0f}", f"{s['N_std']:.0f}",
                f"{s['N_iqr']:.0f}", f"{s['N_min']:.0f}", f"{s['N_max']:.0f}",
                f"{s['L_med']:.2f}", f"{s['L_mean']:.2f}", f"{s['L_std']:.2f}",
                f"{s['L_iqr']:.2f}", f"{s['L_min']:.2f}", f"{s['L_max']:.2f}",
                f"{s['RN_med']:.2f}", f"{s['RL_med']:.2f}",
                f"{s['RN_min']:.2f}", f"{s['RL_min']:.2f}",
                s["topo_modal"], f"{s['runtime_med']:.0f}"])

    # ── 屏幕汇总 ──
    log("")
    log("=" * 78)
    log(f"{'w_N':>4} {'n':>3} {'N_med':>10} {'N_IQR':>8} {'L_med':>8} "
        f"{'L_IQR':>7} {'R_N%':>7} {'R_L%':>7} {'F_med':>8}  modal topology")
    log("-" * 78)
    for s in summary_rows:
        log(f"{s['w_N']:>4.1f} {s['n_valid']:>3} {s['N_med']:>10,.0f} "
            f"{s['N_iqr']:>8,.0f} {s['L_med']:>8.1f} {s['L_iqr']:>7.1f} "
            f"{s['RN_med']:>7.2f} {s['RL_med']:>7.2f} {s['F_med']:>8.4f}  "
            f"{s['topo_modal']}")
    log("-" * 78)
    n_timeout = sum(1 for r in rows if r["timed_out"])
    n_fail = sum(1 for r in rows if not r["success"])
    log(f"完成 {len(rows)} 条链 | 失败 {n_fail} | 其中超时 {n_timeout} "
        f"| 墙钟 {wall / 60:.1f} min")
    log(f"逐链明细: {os.path.abspath(PER_RUN_CSV)}")
    log(f"汇总    : {os.path.abspath(SUMMARY_CSV)}")


if __name__ == "__main__":
    main()

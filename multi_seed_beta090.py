# -*- coding: utf-8 -*-
"""
multi_seed_beta090.py — 多 seed 统计实验（R2-5 运行级分布）

参数与单链温度实验完全一致：
    T0 = 0.5，beta = 0.90，w_N = 0.5，K_max = 150，step = 0.15，p_topo = 0.25
约束（与单链实验一致）：
    E1/E2 首层汇聚约束关闭；FAF -> P 禁越线约束关闭
    → 拓扑空间 = 14 种保序二叉树的全部

N_SEEDS 条独立 SA 链（seed = 0 .. N_SEEDS-1），每条链：
    记录逐代 T / F_current / F_best / 累计接受率 / 累计更差解接受率 / 最优拓扑，
    并独立计时。seed=0 的链应与 sa_temperature_scale.py 的单链结果逐代一致。

并行：ProcessPoolExecutor，每个 worker 进程只构建一次噪声模型。

输出：
  result/multi_seed_beta090_trace.csv     逐代轨迹（seed, gen, T, F_current, F_best,
                                          cum_acceptance, cum_worse, best_topology）
  result/multi_seed_beta090_per_seed.csv  每条链的汇总指标
  result/multi_seed_beta090_stats.csv     总体统计量
"""

import os
import sys
import time
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ── 实验参数（与单链实验一致）──
T0 = 0.5
BETA = 0.90
WN = 0.5
MAX_ITER = 150
STEP = 0.15
P_TOPO = 0.25
EVERY = 10

N_SEEDS = 30
N_WORKERS = 8

POP_FILE = "population_data_square.xlsx"
RESULT_DIR = "result"
TRACE_CSV = os.path.join(RESULT_DIR, "multi_seed_beta090_trace.csv")
PER_SEED_CSV = os.path.join(RESULT_DIR, "multi_seed_beta090_per_seed.csv")
STATS_CSV = os.path.join(RESULT_DIR, "multi_seed_beta090_stats.csv")

# ── 约束开关：模块级，调用时读取 ──
CONSTRAIN_E1E2_FIRST = False   # False -> 不要求 E1/E2 在第一层汇聚
FAF_NO_CROSS = True            # True  -> 关闭 FAF->P 禁越线约束

# 每个 worker 进程的私有上下文（不跨进程共享）
_G = {}


def log(*a):
    print(*a, flush=True)


def _init_worker():
    """每个 worker 进程只执行一次：载入模型、生成拓扑空间。"""
    import test as t
    t.CONSTRAIN_E1E2_FIRST = CONSTRAIN_E1E2_FIRST
    t.FAF_NO_CROSS = FAF_NO_CROSS
    t.WN = WN
    pop, _ = t.load_population_grid(POP_FILE)
    _G["t"] = t
    _G["nm"] = t.SELNoiseModel(pop)
    _G["pf"] = t.PathfinderConfig()
    _G["astar"] = t.AStar2DNoise(_G["nm"])
    _G["NP"] = t.numbered_points
    _G["IFPT"] = t.IF_PT
    _G["P"] = t.P
    _G["trees"] = t.generate_convergence_structures()


def _evaluate(theta, tree):
    t = _G["t"]
    return t.evaluate_X(theta, tree, _G["NP"], _G["IFPT"], _G["P"],
                        _G["pf"], _G["nm"], _G["astar"])


def _build_initial_solution(rng):
    """随机取一个拓扑，Θ 从扇环中点出发；无效则重采样（最多 20 次）。"""
    t = _G["t"]
    trees = _G["trees"]
    tree = theta = F = info = None
    for _ in range(20):
        tree = copy.deepcopy(trees[rng.randint(0, len(trees))])
        dim = t.count_internal_params(tree)
        theta = (np.full(dim, 0.5) if t.INIT_THETA_MIDPOINT
                 else rng.uniform(0.0, 1.0, size=dim))
        F, info = _evaluate(theta, tree)
        if info.get("is_valid"):
            break
    return tree, theta, F, info


def _run_seed(seed):
    """单条 SA 链。返回 (seed, trace, summary_dict)。

    trace 每行：(gen, T, F_current, F_best, n_acc, n_worse, best_topo)
    """
    t = _G["t"]
    rng = np.random.RandomState(seed)

    t_start = time.time()
    tree, theta, F, info = _build_initial_solution(rng)
    init_valid = bool(info.get("is_valid"))

    best_F = F
    best_topo = t.tree_to_bracket(tree)
    best_N = info.get("N_exposed")
    best_L = info.get("L_total")
    best_RN = info.get("R_N")
    best_RL = info.get("R_L")

    n_acc = 0
    n_worse = 0
    last_improve_gen = 0
    trace = [(0, T0, F, best_F, 0, 0, best_topo)]

    for gen in range(1, MAX_ITER + 1):
        T = T0 * (BETA ** (gen - 1))          # 该代温度（确定性指数衰减）

        do_topo = rng.random_sample() < P_TOPO
        if do_topo:
            high = [r for r in info.get("ratios", []) if r["rho"] > t.RHO_MAX]
            cands = t.targeted_rotations(tree, high) if high else t.tree_rotations(tree)
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
                best_topo = t.tree_to_bracket(tree)
                best_N = info.get("N_exposed")
                best_L = info.get("L_total")
                best_RN = info.get("R_N")
                best_RL = info.get("R_L")
                last_improve_gen = gen

        trace.append((gen, T, F, best_F, n_acc, n_worse, best_topo))

    runtime = time.time() - t_start

    # 收敛到 final F_best 的 1% 邻域所需的代数
    thresh = best_F * 1.01 if best_F > 0 else best_F
    gen_1pct = MAX_ITER
    for gen, _T, _Fc, Fb, _na, _nw, _tp in trace:
        if Fb <= thresh:
            gen_1pct = gen
            break

    success = bool(best_F < 1.0e5)   # 未触发 BIG_PENALTY

    summary = dict(
        seed=seed,
        F_best=best_F,
        best_topology=best_topo,
        N_exposed=best_N,
        L_total=best_L,
        R_N=best_RN,
        R_L=best_RL,
        n_accepted=n_acc,
        cum_acceptance=n_acc / MAX_ITER,
        n_worse_accepted=n_worse,
        cum_worse_acceptance=n_worse / MAX_ITER,
        last_improve_gen=last_improve_gen,
        gen_to_1pct=gen_1pct,
        runtime_s=runtime,
        success=success,
        init_valid=init_valid,
    )
    return seed, trace, summary


def _stats(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"), median=float("nan"),
                    q1=float("nan"), q3=float("nan"), iqr=float("nan"),
                    best=float("nan"), worst=float("nan"))
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(n=int(a.size), mean=float(a.mean()),
                std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                median=float(med), q1=float(q1), q3=float(q3),
                iqr=float(q3 - q1), best=float(a.min()), worst=float(a.max()))


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    t_wall = time.time()

    log("=" * 78)
    log(f"Multi-seed run: T0={T0}, beta={BETA}, w_N={WN}, "
        f"K_max={MAX_ITER}, step={STEP}, p_topo={P_TOPO}")
    log(f"Constraints OFF: E1E2_first={CONSTRAIN_E1E2_FIRST}, "
        f"no_cross={FAF_NO_CROSS}")
    log(f"Seeds = {N_SEEDS} (0..{N_SEEDS - 1}) | workers = {N_WORKERS}")
    log("=" * 78)

    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_run_seed, s): s for s in range(N_SEEDS)}
        for fut in as_completed(futs):
            s = futs[fut]
            seed, trace, summary = fut.result()
            results[seed] = (trace, summary)
            done += 1
            log(f"  [{done:>2}/{N_SEEDS}] seed={seed:<3} "
                f"F_best={summary['F_best']:.4f}  "
                f"topo={summary['best_topology']:<16} "
                f"acc={summary['cum_acceptance'] * 100:5.1f}%  "
                f"conv={summary['last_improve_gen']:>3}  "
                f"t={summary['runtime_s']:.0f}s")

    wall = time.time() - t_wall

    # ── 逐代轨迹落盘 ──
    with open(TRACE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "gen", "T", "F_current", "F_best",
                    "n_accepted", "n_worse_accepted",
                    "cum_acceptance", "cum_worse_acceptance", "best_topology"])
        for seed in range(N_SEEDS):
            for gen, T, Fc, Fb, na, nw, topo in results[seed][0]:
                w.writerow([seed, gen, f"{T:.8f}", f"{Fc:.6f}", f"{Fb:.6f}",
                            na, nw,
                            f"{na / gen:.6f}" if gen else "0.000000",
                            f"{nw / gen:.6f}" if gen else "0.000000", topo])

    summaries = [results[s][1] for s in range(N_SEEDS)]
    fields = ["seed", "F_best", "best_topology", "N_exposed", "L_total",
              "R_N", "R_L", "n_accepted", "cum_acceptance",
              "n_worse_accepted", "cum_worse_acceptance", "last_improve_gen",
              "gen_to_1pct", "runtime_s", "success", "init_valid"]
    with open(PER_SEED_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for sm in summaries:
            w.writerow([sm["seed"],
                        f"{sm['F_best']:.6f}",
                        sm["best_topology"],
                        sm["N_exposed"], f"{sm['L_total']:.1f}",
                        f"{sm['R_N']:.2f}", f"{sm['R_L']:.2f}",
                        sm["n_accepted"], f"{sm['cum_acceptance']:.6f}",
                        sm["n_worse_accepted"], f"{sm['cum_worse_acceptance']:.6f}",
                        sm["last_improve_gen"], sm["gen_to_1pct"],
                        f"{sm['runtime_s']:.1f}", sm["success"],
                        sm["init_valid"]])

    # ── 汇总统计 ──
    n_succ = sum(1 for sm in summaries if sm["success"])
    st_F = _stats([sm["F_best"] for sm in summaries])
    st_rt = _stats([sm["runtime_s"] for sm in summaries])
    st_conv = _stats([sm["last_improve_gen"] for sm in summaries])
    st_acc = _stats([sm["cum_acceptance"] for sm in summaries])
    st_worse = _stats([sm["cum_worse_acceptance"] for sm in summaries])
    st_N = _stats([sm["N_exposed"] for sm in summaries])
    st_L = _stats([sm["L_total"] for sm in summaries])

    topo_counts = {}
    for sm in summaries:
        topo_counts[sm["best_topology"]] = topo_counts.get(sm["best_topology"], 0) + 1

    stats_rows = []
    for key, st in [("F_best", st_F), ("N_exposed", st_N), ("L_total", st_L),
                    ("cum_acceptance", st_acc),
                    ("cum_worse_acceptance", st_worse),
                    ("last_improve_gen", st_conv), ("runtime_s", st_rt)]:
        stats_rows.append([key, st["n"], f"{st['mean']:.4f}", f"{st['std']:.4f}",
                           f"{st['median']:.4f}", f"{st['q1']:.4f}",
                           f"{st['q3']:.4f}", f"{st['iqr']:.4f}",
                           f"{st['best']:.4f}", f"{st['worst']:.4f}"])

    with open(STATS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mean", "std", "median", "q1", "q3", "iqr",
                    "best", "worst"])
        w.writerows(stats_rows)
        w.writerow([])
        w.writerow(["success_rate", f"{n_succ / N_SEEDS:.4f}"])
        w.writerow(["n_success", n_succ])
        w.writerow(["total_wall_s", f"{wall:.1f}"])
        w.writerow(["sum_runtime_s", f"{sum(sm['runtime_s'] for sm in summaries):.1f}"])
        w.writerow([])
        w.writerow(["topology", "count", "share"])
        for topo, cnt in sorted(topo_counts.items(), key=lambda kv: -kv[1]):
            w.writerow([topo, cnt, f"{cnt / N_SEEDS:.2f}"])

    log("")
    log(f"{'metric':<24}{'mean':>10}{'std':>10}{'median':>10}"
        f"{'IQR':>18}{'best':>10}{'worst':>10}")
    log("-" * 92)
    for key, st in [("F_best", st_F), ("N_exposed", st_N), ("L_total", st_L),
                    ("cum_acceptance", st_acc),
                    ("cum_worse_acceptance", st_worse),
                    ("last_improve_gen", st_conv), ("runtime_s", st_rt)]:
        log(f"{key:<24}{st['mean']:>10.4f}{st['std']:>10.4f}{st['median']:>10.4f}"
            f"{'[' + format(st['q1'], '.4f') + ', ' + format(st['q3'], '.4f') + ']':>18}"
            f"{st['best']:>10.4f}{st['worst']:>10.4f}")
    log("-" * 92)
    log(f"Success rate: {n_succ}/{N_SEEDS} = {n_succ / N_SEEDS * 100:.1f}%")
    log("Best-found topology counts:")
    for topo, cnt in sorted(topo_counts.items(), key=lambda kv: -kv[1]):
        log(f"  {topo:<18} {cnt:>3}  ({cnt / N_SEEDS * 100:.0f}%)")
    log("")
    log(f"Wall time (parallel, {N_WORKERS} workers): {wall:.0f}s "
        f"| sum of chain times: {sum(sm['runtime_s'] for sm in summaries):.0f}s")
    log(f"Trace   : {os.path.abspath(TRACE_CSV)}")
    log(f"Per-seed: {os.path.abspath(PER_SEED_CSV)}")
    log(f"Stats   : {os.path.abspath(STATS_CSV)}")


if __name__ == "__main__":
    main()

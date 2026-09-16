# -*- coding: utf-8 -*-
"""
exhaustive_topology_beta090.py — 14 种拓扑全穷举（R2-5 拓扑完备性）

对 5 进场点保序二叉树的全部 14 种拓扑，各自固定拓扑、只优化汇聚点位置 Θ
（扇环参数），报告每种拓扑的路径长度 L、人口影响 N_exposed、目标函数 F。

参数：T0 = 0.5，beta = 0.90，w_N = 0.5，K_max = 150，step = 0.15
约束：E1/E2 首层汇聚约束关闭（开启时只剩 5 种拓扑）；
      FAF -> P 禁越线约束关闭
初始解：Θ ~ U(0,1) 随机抽取，不可行则重抽，最多 20 次
提前中止：若一条链跑到第 ABORT_GEN 代仍未出现任何可行解，判定该拓扑
          在本次协议下不可行并结束该链（避免 150 代空转）

输出：
  result/exhaustive_topology_beta090.xlsx
    sheet "topology_summary"  每种拓扑一行：最优 F / N / L / R_N / R_L / 可行率
    sheet "per_seed"          每 (拓扑, seed) 一行明细
"""

import os
import time
import copy
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ── 实验参数 ──
T0 = 0.5
BETA = 0.90
WN = 0.5
MAX_ITER = 150
STEP = 0.15
N_SEEDS = 5
ABORT_GEN = 30
MAX_INIT_TRIES = 20
N_WORKERS = 8

POP_FILE = "population_data_square.xlsx"
RESULT_DIR = "result"
XLSX = os.path.join(RESULT_DIR, "exhaustive_topology_beta090.xlsx")

CONSTRAIN_E1E2_FIRST = False   # 必须为 False，否则拓扑空间只有 5 种
FAF_NO_CROSS = True            # True -> 关闭 FAF->P 禁越线约束

_G = {}


def log(*a):
    print(*a, flush=True)


def _init_worker():
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
    return len(_G["trees"])


def _evaluate(theta, tree):
    t = _G["t"]
    return t.evaluate_X(theta, tree, _G["NP"], _G["IFPT"], _G["P"],
                        _G["pf"], _G["nm"], _G["astar"])


def _run_one(args):
    """固定拓扑、只优化 Θ 的一条 SA 链。"""
    topo_idx, seed = args
    t = _G["t"]
    rng = np.random.RandomState(seed)
    tree = copy.deepcopy(_G["trees"][topo_idx])
    topo = t.tree_to_bracket(tree)

    t_start = time.time()

    # 随机初始 Θ，不可行则重抽
    dim = t.count_internal_params(tree)
    theta = np.full(dim, 0.5)
    F, info = t.evaluate_X(theta, tree, _G["NP"], _G["IFPT"], _G["P"],
                           _G["pf"], _G["nm"], _G["astar"])
    init_tries = 0
    for init_tries in range(1, MAX_INIT_TRIES + 1):
        theta = rng.uniform(0.0, 1.0, size=dim)
        F, info = _evaluate(theta, tree)
        if info.get("is_valid"):
            break

    def _snap(Fv, infov):
        return dict(F=Fv,
                    N=infov.get("N_exposed"),
                    L=infov.get("L_total"),
                    R_N=infov.get("R_N"),
                    R_L=infov.get("R_L"))

    best = _snap(F, info)
    valid = bool(info.get("is_valid"))
    seen_valid = valid
    n_acc = 0
    aborted = False

    for gen in range(1, MAX_ITER + 1):
        T = T0 * (BETA ** (gen - 1))
        theta_new = t.neighbor_theta(theta, rng, STEP)
        F_new, info_new = _evaluate(theta_new, tree)
        dF = F_new - F
        if t.sa_accept(dF, T, rng):
            theta, F, info = theta_new, F_new, info_new
            n_acc += 1
            if info.get("is_valid"):
                seen_valid = True
            if F < best["F"]:
                best = _snap(F, info)
        if not seen_valid and gen >= ABORT_GEN:
            aborted = True
            break

    runtime = time.time() - t_start
    success = bool(best["F"] < 1.0e5)

    return dict(topo_idx=topo_idx, topology=topo, seed=seed,
                feasible=success, aborted=aborted,
                F_best=best["F"], N_exposed=best["N"], L_total=best["L"],
                R_N=best["R_N"], R_L=best["R_L"],
                init_tries=init_tries, n_accepted=n_acc,
                gens_run=(gen if aborted else MAX_ITER),
                runtime_s=runtime)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    t_wall = time.time()

    log("=" * 84)
    log(f"Exhaustive topology run: T0={T0}, beta={BETA}, w_N={WN}, "
        f"K_max={MAX_ITER}, 14 topologies x {N_SEEDS} seeds")
    log(f"Constraints OFF: E1E2_first={CONSTRAIN_E1E2_FIRST}, "
        f"no_cross={FAF_NO_CROSS} | abort if no feasible by gen {ABORT_GEN}")
    log("=" * 84)

    n_topo = 14   # 5 进场点保序二叉树 = Catalan(4) = 14
    records = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker) as ex:
        jobs = [(i, s) for i in range(n_topo) for s in range(N_SEEDS)]
        futs = {ex.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception:
                log(f"  !! worker failed for {futs[fut]}:")
                log(traceback.format_exc())
                continue
            records.append(rec)
            done += 1
            tag = "OK " if rec["feasible"] else ("ABORT" if rec["aborted"] else "FAIL ")
            log(f"  [{done:>2}/{len(jobs)}] T{rec['topo_idx']:>2} seed={rec['seed']} "
                f"{tag} F={rec['F_best']:>10.4f} "
                f"N={rec['N_exposed']} L={rec['L_total']} "
                f"t={rec['runtime_s']:.0f}s")

    wall = time.time() - t_wall
    if not records:
        log("No records produced; aborting.")
        return

    per_seed = pd.DataFrame(records).sort_values(["topo_idx", "seed"])
    per_seed = per_seed[["topo_idx", "topology", "seed", "feasible", "aborted",
                         "F_best", "N_exposed", "L_total", "R_N", "R_L",
                         "init_tries", "n_accepted", "gens_run", "runtime_s"]]

    # ── 按拓扑汇总 ──
    rows = []
    for i in range(n_topo):
        sub = per_seed[per_seed["topo_idx"] == i]
        if sub.empty:
            continue
        ok = sub[sub["feasible"]]
        topo = sub["topology"].iloc[0]
        if ok.empty:
            rows.append(dict(topo_idx=i, topology=topo, n_seeds=len(sub),
                             n_feasible=0, feasible_rate=0.0,
                             F_best=np.nan, F_mean=np.nan, F_std=np.nan,
                             N_exposed=np.nan, L_total=np.nan,
                             R_N=np.nan, R_L=np.nan,
                             mean_runtime_s=round(sub["runtime_s"].mean(), 1)))
        else:
            b = ok.loc[ok["F_best"].idxmin()]
            rows.append(dict(topo_idx=i, topology=topo, n_seeds=len(sub),
                             n_feasible=len(ok),
                             feasible_rate=len(ok) / len(sub),
                             F_best=b["F_best"],
                             F_mean=ok["F_best"].mean(),
                             F_std=ok["F_best"].std(ddof=1) if len(ok) > 1 else 0.0,
                             N_exposed=b["N_exposed"], L_total=b["L_total"],
                             R_N=b["R_N"], R_L=b["R_L"],
                             mean_runtime_s=round(sub["runtime_s"].mean(), 1)))
    summary = pd.DataFrame(rows).sort_values("F_best", na_position="last")

    with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="topology_summary", index=False)
        per_seed.to_excel(xw, sheet_name="per_seed", index=False)

    log("")
    log(f"{'idx':>3} {'topology':<20}{'feas':>6}{'F_best':>10}{'N_exposed':>12}"
        f"{'L_km':>10}{'R_N%':>8}{'R_L%':>8}")
    log("-" * 84)
    for _, r in summary.iterrows():
        f_f = f"{r['F_best']:.4f}" if pd.notna(r["F_best"]) else "infeasible"
        n_s = f"{int(r['N_exposed'])}" if pd.notna(r["N_exposed"]) else "-"
        l_s = f"{r['L_total']:.1f}" if pd.notna(r["L_total"]) else "-"
        rn_s = f"{r['R_N']:.1f}" if pd.notna(r["R_N"]) else "-"
        rl_s = f"{r['R_L']:.1f}" if pd.notna(r["R_L"]) else "-"
        log(f"{int(r['topo_idx']):>3} {r['topology']:<20}"
            f"{int(r['n_feasible'])}/{int(r['n_seeds']):<4}{f_f:>10}{n_s:>12}"
            f"{l_s:>10}{rn_s:>8}{rl_s:>8}")
    log("-" * 84)
    n_feas_topo = int((summary["n_feasible"] > 0).sum())
    log(f"Feasible topologies: {n_feas_topo}/{n_topo}")
    log(f"Wall time: {wall:.0f}s ({N_WORKERS} workers) | "
        f"sum of chain times: {per_seed['runtime_s'].sum():.0f}s")
    log(f"Excel: {os.path.abspath(XLSX)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
sa_temperature_scale.py — 温度尺度实验（R2-5 温度整定）

参数：T0 = 0.5，beta = 0.90，w_N = 0.5，K_max = 150
约束：E1/E2 首层汇聚约束关闭；FAF->P 禁越线约束关闭
单条 SA 链（seed = 0），逐代记录温度 / 是否接受 / 累计接受率 / 收敛，
每 10 代输出一次统计行。

输出：
  result/sa_t0_05_beta_090_trace.csv      逐代完整轨迹（供出图）
  result/sa_t0_05_beta_090_every10.csv    每 10 代统计表
"""

import os
import time
import copy
import csv
import numpy as np

import test as t

# ── 实验参数 ──
T0 = 0.5
BETA = 0.90
WN = 0.5
MAX_ITER = 150
STEP = 0.15
P_TOPO = 0.25
SEED = 0
EVERY = 10

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

TRACE_CSV = os.path.join(RESULT_DIR, "sa_t0_05_beta_090_trace.csv")
EVERY_CSV = os.path.join(RESULT_DIR, "sa_t0_05_beta_090_every10.csv")

# ── 关闭两个约束（模块级开关，调用时读取）──
t.CONSTRAIN_E1E2_FIRST = False   # False -> 不要求 E1/E2 在第一层汇聚
t.FAF_NO_CROSS = True            # True  -> 关闭 FAF->P 禁越线约束
t.WN = WN                        # 目标函数权重；A* 内层权重自动跟随


def log(*a):
    print(*a, flush=True)


def build_initial_solution(rng):
    """随机取一个拓扑，Θ 从扇环中点出发；无效则重采样（最多 20 次）。"""
    tree = theta = F = info = None
    for _ in range(20):
        tree = copy.deepcopy(trees[rng.randint(0, len(trees))])
        dim = t.count_internal_params(tree)
        theta = (np.full(dim, 0.5) if t.INIT_THETA_MIDPOINT
                 else rng.uniform(0.0, 1.0, size=dim))
        F, info = t.evaluate_X(theta, tree, NP, IFPT, P, pf, nm, astar)
        if info.get("is_valid"):
            break
    return tree, theta, F, info


def run_chain(seed):
    """单条 SA 链，返回逐代轨迹 [(gen, T, F_cur, F_best, n_acc, n_worse, topo)]。"""
    rng = np.random.RandomState(seed)
    tree, theta, F, info = build_initial_solution(rng)

    best_F = F
    n_acc = 0
    n_worse = 0
    trace = [(0, T0, F, best_F, 0, 0, t.tree_to_bracket(tree))]

    for gen in range(1, MAX_ITER + 1):
        # 该代使用的温度（确定性指数衰减）
        T = T0 * (BETA ** (gen - 1))

        # 邻域移动：以 P_TOPO 概率做拓扑旋转，否则扰动 Θ
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

        F_new, info_new = t.evaluate_X(theta_new, tree_new, NP, IFPT, P, pf, nm, astar)
        dF = F_new - F

        if t.sa_accept(dF, T, rng):
            tree, theta, F, info = tree_new, theta_new, F_new, info_new
            n_acc += 1
            if dF > 0:
                n_worse += 1
            if F < best_F:
                best_F = F

        trace.append((gen, T, F, best_F, n_acc, n_worse, t.tree_to_bracket(tree)))

    return trace, info


# ── 准备模型 ──
pop, _ = t.load_population_grid("population_data_square.xlsx")
nm = t.SELNoiseModel(pop)
pf = t.PathfinderConfig()
astar = t.AStar2DNoise(nm)
NP = t.numbered_points
IFPT = t.IF_PT
P = t.P
trees = t.generate_convergence_structures()

log("=" * 74)
log(f"Temperature scale run: T0={T0}, beta={BETA}, w_N={WN}, "
    f"K_max={MAX_ITER}, seed={SEED}")
log(f"Constraints OFF: E1E2_first={t.CONSTRAIN_E1E2_FIRST}, "
    f"no_cross={t.FAF_NO_CROSS} | topology space = {len(trees)}")
log("=" * 74)

t_start = time.time()
trace, final_info = run_chain(SEED)
elapsed = time.time() - t_start

# ── 逐代轨迹落盘 ──
with open(TRACE_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["gen", "T", "F_current", "F_best", "n_accepted",
                "n_worse_accepted", "cum_acceptance", "cum_worse_acceptance",
                "topology"])
    for gen, T, Fc, Fb, na, nw, topo in trace:
        w.writerow([gen, f"{T:.8f}", f"{Fc:.6f}", f"{Fb:.6f}", na, nw,
                    f"{na / gen:.6f}" if gen else "0.000000",
                    f"{nw / gen:.6f}" if gen else "0.000000", topo])

# ── 每 10 代统计表 ──
rows = [rec for rec in trace if rec[0] % EVERY == 0]
log("")
log(f"{'gen':>5} {'T':>10} {'cumAcc':>8} {'cumWorse':>9} "
    f"{'F_current':>10} {'F_best':>10}  topology")
log("-" * 74)
for gen, T, Fc, Fb, na, nw, topo in rows:
    acc = na / gen if gen else 0.0
    wor = nw / gen if gen else 0.0
    log(f"{gen:>5} {T:>10.6f} {acc * 100:>7.1f}% {wor * 100:>8.1f}% "
        f"{Fc:>10.4f} {Fb:>10.4f}  {topo}")
log("-" * 74)

with open(EVERY_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["gen", "T", "cum_acceptance", "cum_worse_acceptance",
                "F_current", "F_best", "topology"])
    for gen, T, Fc, Fb, na, nw, topo in rows:
        w.writerow([gen, f"{T:.8f}",
                    f"{na / gen:.6f}" if gen else "0.000000",
                    f"{nw / gen:.6f}" if gen else "0.000000",
                    f"{Fc:.6f}", f"{Fb:.6f}", topo])

g_last, T_last, F_last, Fb_last, na_last, nw_last, topo_last = trace[-1]
log("")
log(f"Final: F_current={F_last:.4f}  F_best={Fb_last:.4f}  "
    f"topology={topo_last}")
log(f"Acceptance over chain: total={na_last / MAX_ITER * 100:.1f}%, "
    f"worse-move={nw_last / MAX_ITER * 100:.1f}%")
log(f"T range: {T0:.4f} -> {T_last:.3e}  |  valid={final_info.get('is_valid')}")
log(f"Elapsed: {elapsed:.0f}s")
log(f"Trace : {os.path.abspath(TRACE_CSV)}")
log(f"Table : {os.path.abspath(EVERY_CSV)}")

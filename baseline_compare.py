# -*- coding: utf-8 -*-
"""
baseline_compare.py — 实验 D：与替代优化器对比（R2-5）

在【相同评估预算】下对比三种方法：
  SA      —— 完整方法（拓扑+位置，模拟退火）
  greedy  —— 贪心局部搜索（同样邻域算子，但只接受更优解，即 T→0）
  random  —— 随机重启（每步随机拓扑+随机位置，取最优）

贪心与随机复用了与 SA 完全相同的邻域算子（tree_rotations/neighbor_theta）和
evaluate_X，唯一差异在"接受准则"，从而干净地检验退火机制的价值。
E1/E2 约束关闭。

输出：result/baseline_compare.xlsx + 控制台对比表。
"""

import test as _t
import numpy as np
import os, time, copy

RESULT_DIR = "result"; os.makedirs(RESULT_DIR, exist_ok=True)


def log(*a, **kw):
    print(*a, **kw, flush=True)


WN = 0.5
N_SEEDS = 10
MAX_ITER = 150       # 每条链评估预算（SA/greedy 每代 1 次 evaluate；random 每步 1 次）
STEP = 0.15

_t.CONSTRAIN_E1E2_FIRST = False
_t.WN = WN

log("=" * 60)
log("实验 D：SA vs 贪心 vs 随机重启（同预算，E1/E2 约束关闭）")
log(f"w_N={WN}, {N_SEEDS} seed × {MAX_ITER} 评估")
log("=" * 60)

pop, _ = _t.load_population_grid("population_data_square.xlsx")
nm = _t.SELNoiseModel(pop)
pf = _t.PathfinderConfig()
astar = _t.AStar2DNoise(nm)

NP = _t.numbered_points
IFPT = _t.IF_PT
P = _t.P


def _new_solution(rng, enum_trees):
    """随机生成一个 (tree, theta) 解。"""
    tree = copy.deepcopy(enum_trees[rng.randint(0, len(enum_trees))])
    dim = _t.count_internal_params(tree)
    theta = rng.uniform(0.0, 1.0, size=dim)
    return tree, theta


def _resample_valid(rng, enum_trees, tries=20):
    """重采样直到得到可行解，返回 (F, tree, theta, info)。"""
    for _ in range(tries):
        tree, theta = _new_solution(rng, enum_trees)
        F, info = _t.evaluate_X(theta, tree, NP, IFPT, P, pf, nm, astar)
        if info.get('is_valid'):
            return F, tree, theta, info
    return None, None, None, None


def run_sa(seed):
    Fb, tb, thetab, infob, _ = _t.neighborhood_search_run(
        NP, IFPT, P, pf, nm, astar, init_tree=None, max_iter=MAX_ITER,
        step=STEP, seed=seed, T0=_t.SA_T0, beta=_t.SA_BETA,
        p_topo=_t.P_TOPO_MOVE, log_every=0)
    return Fb, infob


def run_greedy(seed):
    """贪心：与 SA 相同邻域，但只接受更优解（不接受更差）。"""
    rng = np.random.RandomState(seed)
    enum_trees = _t.generate_convergence_structures()
    F, tree, theta, info = _resample_valid(rng, enum_trees)
    if F is None:
        return None, None
    best_F, best_info = F, info
    for _ in range(MAX_ITER):
        do_topo = (rng.random_sample() < _t.P_TOPO_MOVE)
        if do_topo:
            high = [r for r in info.get('ratios', []) if r['rho'] > _t.RHO_MAX]
            cands = _t.targeted_rotations(tree, high) if high else _t.tree_rotations(tree)
            if not cands:
                do_topo = False
            else:
                tree_new = cands[rng.randint(0, len(cands))]
                theta_new = _t._inherit_theta(tree, theta, tree_new, rng)
        if not do_topo:
            tree_new = tree
            theta_new = _t.neighbor_theta(theta, rng, STEP)
        F_new, info_new = _t.evaluate_X(theta_new, tree_new, NP, IFPT, P, pf, nm, astar)
        if F_new < F:   # 贪心：只接受更优
            tree, theta, F, info = tree_new, theta_new, F_new, info_new
            if F < best_F:
                best_F, best_info = F, info
    return best_F, best_info


def run_random(seed):
    """随机重启：每步全新随机解，取最优。"""
    rng = np.random.RandomState(seed)
    enum_trees = _t.generate_convergence_structures()
    best_F, best_info = float('inf'), None
    for _ in range(MAX_ITER):
        tree, theta = _new_solution(rng, enum_trees)
        F, info = _t.evaluate_X(theta, tree, NP, IFPT, P, pf, nm, astar)
        if info.get('is_valid') and F < best_F:
            best_F, best_info = F, info
    return (best_F if best_info else None), best_info


METHODS = [('SA', run_sa), ('greedy', run_greedy), ('random', run_random)]

rows = []
for mid, mfunc in METHODS:
    fs, ts, ns, ls = [], [], [], []
    for seed in range(N_SEEDS):
        t0 = time.time()
        Fb, infob = mfunc(seed)
        dt = time.time() - t0
        if infob and infob.get('is_valid') and Fb is not None:
            fs.append(Fb)
            ts.append(dt)
            ns.append(float(infob.get('N_exposed', 0)))
            ls.append(float(infob.get('L_total', 0)))
    if fs:
        a = np.array(fs)
        row = dict(method=mid, n=len(fs),
                   mean=float(a.mean()), std=float(a.std()),
                   median=float(np.median(a)), best=float(a.min()),
                   worst=float(a.max()),
                   best_N=float(min(ns)), best_L=float(min(ls)),
                   mean_time=float(np.mean(ts)))
        log(f"  {mid:8s}: n={len(fs)} F* mean={row['mean']:.4f} "
            f"median={row['median']:.4f} best={row['best']:.4f} "
            f"worst={row['worst']:.4f}  平均耗时 {row['mean_time']:.1f}s")
    else:
        row = dict(method=mid, n=0, mean=None, std=None, median=None,
                   best=None, worst=None, best_N=None, best_L=None,
                   mean_time=None)
        log(f"  {mid:8s}: 全部 FAILED")
    rows.append(row)

# ── 保存 xlsx ──
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

wb = openpyxl.Workbook()
hf = Font(bold=True)
hf2 = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
bd = Border(left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin'))
ws = wb.active
ws.title = "方法对比"
for c, h in enumerate(['方法', '可行数', 'mean_F*', 'std_F*', 'median_F*',
                       'best_F*', 'worst_F*', 'best_N', 'best_L(km)',
                       'mean_time(s)'], 1):
    cl = ws.cell(row=1, column=c, value=h)
    cl.font = hf; cl.fill = hf2; cl.border = bd
    cl.alignment = Alignment(horizontal='center')
for ri, r in enumerate(rows, 2):
    vals = [r['method'], r['n'],
            round(r['mean'], 6) if r['mean'] is not None else None,
            round(r['std'], 6) if r['std'] is not None else None,
            round(r['median'], 6) if r['median'] is not None else None,
            round(r['best'], 6) if r['best'] is not None else None,
            round(r['worst'], 6) if r['worst'] is not None else None,
            round(r['best_N'], 0) if r['best_N'] is not None else None,
            round(r['best_L'], 2) if r['best_L'] is not None else None,
            round(r['mean_time'], 1) if r['mean_time'] is not None else None]
    for c, v in enumerate(vals, 1):
        cl = ws.cell(row=ri, column=c, value=v); cl.border = bd

out = os.path.join(RESULT_DIR, "baseline_compare.xlsx")
wb.save(out)
log(f"\n结果: {os.path.abspath(out)}")

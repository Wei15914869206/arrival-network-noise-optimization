# -*- coding: utf-8 -*-
"""
r2_8_robustness.py — R2-8 敏感性 / 稳健性实验（精简版，260 链，约 3 小时）

【锚点】所有扰动都是对下面这套配置的偏离，锚点等于产出已发表结果的配置：
    RHO_ASTAR_MAX = 1.25   RHO_MAX = 1.5   SECTOR_R_MIN_FRAC = 0.5
    MAX_TURN_ANGLE = 60    MIN_TURN_DISTANCE = 5 km   MAX_TURNS_PER_SEGMENT = 3
    A* 邻域半径 = 3 (48 邻接)   A* 内层噪声权重 = 0.5 (固定, AUTO_SYNC=False)
    SA: T0=0.5, beta=0.90, K_max=150, p_topo=0.25, step=0.15
    声学: T=20C, RH=70%, v=80 m/s, L_A_ref=140 dB, f_ref=500 Hz
    评估: SEL_threshold=65 dB, n_samples=30
    人口: 1/120 deg (~0.92 km) 原生栅格
    外层权重: w_N = 0.3
所有注入均为模块级常量 / 实例属性 / 构造参数覆盖，不改动 test.py。

【实验设计】两阶段"筛选—确认"架构
  E1  阈值口径修正：65 dB x 3 权重 x 10 = 30 链
      - 5 阈值下的基线重算（自校验 65 dB = 6,169,767）
      - 事后重分类（固定轨迹，只改判定阈值，基线与优化轨迹同时重算）
      - 图 7 三条曲线（w_N = 0.0 / 0.3 / 1.0）
  E2  阈值完整重优化：另外 4 个阈值 x w_N=0.3 x 10 = 40 链（65 dB 复用 E1）
      用于区分"事后重分类"与"各阈值下完整重优化"
  E3  单因素敏感性（w_N=0.3）：扇环下界 / 绕行上界 / 最大转弯角 / A* 邻域
      各两方向 -> 8 配置 x 10 = 80 链
  E4  数值离散化（合并因子，粗/默认/细）：2 配置 x 10 = 20 链
      coarse = n_samples 15 + 人口 2km 块 + 邻域半径 2
      fine   = n_samples 60 + 人口原生   + 邻域半径 4
  E5  联合稳健性：9 个情景 x w_N=0.3 x 10 = 90 链（J0 复用 E1 的 65 dB 锚点）

  合计 260 链。单链中位数 ~540 s，14 进程墙钟约 2.8 h。
  零成本部分：5 阈值基线重算、事后重分类、n_samples 三档重算。

【稳健性判据】
  S_T  拓扑族保持率 = 取得锚点拓扑族（根分裂叶子集）的运行数 / 全部可行运行数
  S_D  支配保持率   = 优化解在 N 与 L 上同时优于基线的情景数 / 总情景数

【用法】
  python -u r2_8_robustness.py --stage e1
  python -u r2_8_robustness.py --stage e2   （E1 必须先完成）
  ... --stage e3 / e4 / e5 / report / all
  断点续跑：每块落盘，重启自动跳过已完成。

【产物】
  result/r2_8_tracks.pkl        全部航迹
  result/r2_8_E1.csv ... E5.csv 各实验逐链明细
  result/r2_8_robustness.xlsx   表 A-E
  result/fig7_data.csv          图 7 作图数据
"""

import os
import sys
import time
import copy
import math
import csv
import gc
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ── 运行协议 ──
N_SEEDS = 10
N_WORKERS = 14
CHUNK = 20                     # 每跑这么多链重建一次进程池，防 worker RSS 累积
MODEL_CACHE_MAX = 3
TIME_LIMIT_S = 7200          # 单链墙钟上限（原 900 s；80 邻接配置踩线，放宽 8 倍）
MAX_INIT_TRIES = 20
CONSTRAIN_E1E2_FIRST = False
FAF_NO_CROSS = True
ASTAR_W_N_FIXED = 0.5          # 内层 A* 权重全程固定（解耦）

WN_MAIN = 0.3
WN_E1 = [0.0, 0.3, 1.0]

THRESHOLDS = [55.0, 60.0, 65.0, 70.0, 75.0]
N_SAMPLES_LEVELS = [15, 30, 60]

POP_FILE = "population_data_square.xlsx"
RESULT_DIR = "result"
TRACKS_PKL = os.path.join(RESULT_DIR, "r2_8_tracks.pkl")
OUT_XLSX = os.path.join(RESULT_DIR, "r2_8_robustness.xlsx")
FIG7_CSV = os.path.join(RESULT_DIR, "fig7_data.csv")

# ── 锚点：模块级常量的默认值 ──
_DEFAULTS = {
    "SECTOR_R_MIN_FRAC": 0.5,
    "RHO_ASTAR_MAX": 1.25,
    "RHO_MAX": 1.5,
    "MAX_TURN_ANGLE": 60,
    "MIN_TURN_DISTANCE": 5,
    "MAX_TURNS_PER_SEGMENT": 3,
}
SA_ANCHOR = dict(T0=0.5, beta=0.90, p_topo=0.25, step=0.15, max_iter=150)
MODEL_ANCHOR = dict(thr=65.0, T=20.0, RH=70.0, v=80.0,
                    L_A_ref=140.0, f_ref=500.0, coarse=1, lateral=True)
NBR_ANCHOR = 3


def _cfg(name, layer, group, param, level,
         overrides=None, sa=None, model=None, nbr=None, note=""):
    return dict(name=name, layer=layer, group=group, param=param, level=level,
                overrides=dict(overrides or {}),
                sa=dict(SA_ANCHOR, **(sa or {})),
                model=dict(MODEL_ANCHOR, **(model or {})),
                nbr=NBR_ANCHOR if nbr is None else nbr,
                note=note)


ANCHOR_NAME = "THR65"          # 锚点在 E1 里，被 E2/E5 复用


# ============================================================
# 实验配置
# ============================================================

def cfg_e1():
    return [_cfg(ANCHOR_NAME, "E1", "阈值", "SEL 阈值", "65 dB (锚点)")]


def cfg_e2():
    return [_cfg("THR%d" % int(t), "E2", "阈值", "SEL 阈值", "%.0f dB" % t,
                 model=dict(thr=t)) for t in THRESHOLDS if t != 65.0]


def cfg_e3():
    """单因素：四个参数，各取锚点两侧各一档。"""
    out = []
    SF = [
        ("扇环半径下界比例", "SECTOR_R_MIN_FRAC", 0.25, 0.75),
        ("段内绕行率上界",   "RHO_ASTAR_MAX",    1.1,  1.5),
        ("最大转弯角 (deg)", "MAX_TURN_ANGLE",   45,   75),
        ("最小转弯间距 (km)", "MIN_TURN_DISTANCE", 3,   7),
    ]
    for label, attr, lo, hi in SF:
        for v in (lo, hi):
            out.append(_cfg("%s=%s" % (attr, v), "E3", "结构性", label,
                            str(v), overrides={attr: v}))
    for r in (2, 4):          # A* 邻域半径：2 -> 24 邻接，4 -> 80 邻接
        out.append(_cfg("NBR_RADIUS=%d" % r, "E3", "算法性",
                        "A* 邻域半径", "%d (%d 邻接)" % (r, _nbrs_count(r)),
                        nbr=r))
    return out


def _nbrs_count(r):
    return (2 * r + 1) ** 2 - 1


def cfg_e4():
    """数值离散化合并因子：粗 / 细（默认档即锚点，不单列）。"""
    return [
        _cfg("DISC=coarse", "E4", "数值离散化", "粗档",
             "ns=15, 人口2km块, 邻域r=2",
             model=dict(coarse=2), nbr=2),
        _cfg("DISC=fine", "E4", "数值离散化", "细档",
             "ns=60, 人口原生, 邻域r=4",
             model=dict(coarse=1), nbr=4),
    ]


# 结构 / 算法 / 声学 情景（用于联合实验）
S_STRICT = dict(SECTOR_R_MIN_FRAC=0.75, RHO_ASTAR_MAX=1.1,
                MAX_TURN_ANGLE=45, MIN_TURN_DISTANCE=7)
S_LOOSE = dict(SECTOR_R_MIN_FRAC=0.25, RHO_ASTAR_MAX=1.5,
               MAX_TURN_ANGLE=75, MIN_TURN_DISTANCE=3)
M_LOW = dict(T=5.0, RH=30.0, L_A_ref=137.0, v=70.0)
M_HIGH = dict(T=35.0, RH=90.0, L_A_ref=143.0, v=90.0)


def cfg_e5():
    """联合稳健性 9 个情景（J0 复用锚点，J3/J7 已删：分别与单因素重复、物理无意义）。"""
    S = [
        ("J1", "结构收紧", dict(overrides=S_STRICT)),
        ("J2", "结构放宽", dict(overrides=S_LOOSE)),
        ("J4", "算法/数值粗", dict(nbr=2, model=dict(coarse=2))),
        ("J5", "算法/数值细", dict(nbr=4, model=dict(coarse=1))),
        ("J6", "人口粗化",   dict(model=dict(coarse=2))),
        ("J8", "低传播声学", dict(model=M_LOW)),
        ("J9", "高传播声学", dict(model=M_HIGH)),
        ("J10", "代表性不利", dict(overrides=S_STRICT, nbr=2,
                                 model=dict(coarse=2, **M_LOW))),
        ("J11", "代表性有利", dict(overrides=S_LOOSE, nbr=4,
                                 model=dict(coarse=1, **M_HIGH))),
    ]
    return [_cfg(k, "E5", "联合情景", "联合", desc, note=desc, **kw)
            for k, desc, kw in S]


# ============================================================
# Worker
# ============================================================

_G = {}
_MODEL_CACHE = {}

try:
    import psutil as _psutil
except ImportError:
    _psutil = None


def _mem_tag():
    if _psutil is None:
        return ""
    vm = _psutil.virtual_memory()
    return "  内存 %.0f%% (%.1f/%.1f GB)" % (vm.percent, vm.used / 1e9,
                                             vm.total / 1e9)


def log(*a):
    print(*a, flush=True)


def _init_worker():
    import test as t
    t.CONSTRAIN_E1E2_FIRST = CONSTRAIN_E1E2_FIRST
    t.FAF_NO_CROSS = FAF_NO_CROSS
    t.ASTAR_WN_AUTO_SYNC = False
    t.ASTAR_W_N = ASTAR_W_N_FIXED
    pop, _ = t.load_population_grid(POP_FILE)
    _G["t"] = t
    _G["pop"] = pop
    _G["NP"] = t.numbered_points
    _G["IFPT"] = t.IF_PT
    _G["P"] = t.P
    _G["pf"] = t.PathfinderConfig()
    _G["trees"] = t.generate_convergence_structures()


def _coarsen_block(pop, k):
    """k x k 块内取均值回填：网格尺寸与总人口守恒（实测守恒误差 0%）。"""
    if k <= 1:
        return pop
    ny, nx = pop.shape
    ny2, nx2 = (ny // k) * k, (nx // k) * k
    core = pop[:ny2, :nx2].reshape(ny2 // k, k, nx2 // k, k)
    bm = core.mean(axis=(1, 3))
    out = pop.copy()
    out[:ny2, :nx2] = np.repeat(np.repeat(bm, k, axis=0), k, axis=1)
    return out


def _nbrs_for_radius(r):
    return [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
            if dx != 0 or dy != 0]


def _get_model(mk):
    t = _G["t"]
    key = (mk["thr"], mk["T"], mk["RH"], mk["v"], mk["L_A_ref"],
           mk["f_ref"], mk["coarse"], mk["lateral"])
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    nm = t.SELNoiseModel(_coarsen_block(_G["pop"], mk["coarse"]),
                         SEL_threshold=float(mk["thr"]), T=float(mk["T"]),
                         RH=float(mk["RH"]), v_approach=float(mk["v"]),
                         L_A_ref=float(mk["L_A_ref"]), f_ref=float(mk["f_ref"]))
    if not mk["lateral"]:
        nm._lateral_attenuation_db = lambda beta: (
            0.0 if np.ndim(beta) == 0 else np.zeros_like(np.asarray(beta, float)))
    astar = t.AStar2DNoise(nm)
    if len(_MODEL_CACHE) >= MODEL_CACHE_MAX:
        _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
    _MODEL_CACHE[key] = (nm, astar)
    return _MODEL_CACHE[key]


def _apply_overrides(overrides, nbr):
    """先整体复位到锚点再施加偏离（worker 进程会被复用）。"""
    t = _G["t"]
    for name, val in _DEFAULTS.items():
        setattr(t, name, val)
    for k, v in overrides.items():
        setattr(t, k, v)
    _G["nbr"] = int(nbr)


def _fail_row(cfg, w_n, seed, reason):
    return dict(cfg=cfg["name"], layer=cfg["layer"], group=cfg["group"],
                param=cfg["param"], level=cfg["level"], note=cfg.get("note", ""),
                w_N=float(w_n), seed=seed, F_best=None, N_exposed=None,
                L_total=None, topology=None, segments=None, n_accepted=0,
                runtime_s=0.0, success=False, timed_out=0, init_valid=False,
                model=cfg["model"], nbr=cfg["nbr"], error=reason)


def _run_chain(task):
    """异常一律降级为失败记录，绝不让单条链杀死整个长任务。"""
    cfg, w_n, seed = task
    try:
        return _run_chain_inner(task)
    except MemoryError:
        gc.collect()
        log("  [MemoryError] %s w_N=%s seed=%s 记为失败" % (cfg["name"], w_n, seed))
        return _fail_row(cfg, w_n, seed, "MemoryError")
    except Exception as e:
        gc.collect()
        log("  [%s] %s w_N=%s seed=%s: %s" % (type(e).__name__, cfg["name"],
                                              w_n, seed, e))
        return _fail_row(cfg, w_n, seed, "%s: %s" % (type(e).__name__, e))


def _run_chain_inner(task):
    cfg, w_n, seed = task
    t = _G["t"]
    _apply_overrides(cfg["overrides"], cfg["nbr"])
    nm, astar = _get_model(cfg["model"])
    astar._nbrs = _nbrs_for_radius(cfg["nbr"])
    # A* 路径缓存 key 不含 MAX_TURN_ANGLE / MIN_TURN_DISTANCE / RHO_ASTAR_MAX /
    # _nbrs，跨配置复用同一实例会命中脏缓存，故每条链清空。
    astar._cache = {}
    sa = cfg["sa"]
    t.WN = float(w_n)
    rng = np.random.RandomState(seed)
    t_start = time.time()

    tree = theta = F = info = None
    for _ in range(MAX_INIT_TRIES):
        tree = copy.deepcopy(_G["trees"][rng.randint(0, len(_G["trees"]))])
        dim = t.count_internal_params(tree)
        theta = (np.full(dim, 0.5) if t.INIT_THETA_MIDPOINT
                 else rng.uniform(0.0, 1.0, size=dim))
        F, info = t.evaluate_X(theta, tree, _G["NP"], _G["IFPT"], _G["P"],
                               _G["pf"], nm, astar)
        if info.get("is_valid"):
            break
    init_valid = bool(info.get("is_valid"))

    def _snap(iv):
        return dict(N=iv.get("N_exposed"), L=iv.get("L_total"))

    best, best_F = _snap(info), F
    best_topo = t.tree_to_bracket(tree)
    best_paths = info.get("all_paths")
    n_acc = timed_out = 0

    for gen in range(1, sa["max_iter"] + 1):
        if time.time() - t_start > TIME_LIMIT_S:
            timed_out = 1
            break
        T = sa["T0"] * (sa["beta"] ** (gen - 1))
        do_topo = rng.random_sample() < sa["p_topo"]
        if do_topo:
            high = [r for r in info.get("ratios", []) if r["rho"] > t.RHO_MAX]
            cands = (t.targeted_rotations(tree, high) if high
                     else t.tree_rotations(tree))
            if not cands:
                do_topo = False
            else:
                tree_new = cands[rng.randint(0, len(cands))]
                theta_new = t._inherit_theta(tree, theta, tree_new, rng)
        if not do_topo:
            tree_new = tree
            theta_new = t.neighbor_theta(theta, rng, sa["step"])

        F_new, info_new = t.evaluate_X(theta_new, tree_new, _G["NP"], _G["IFPT"],
                                       _G["P"], _G["pf"], nm, astar)
        if t.sa_accept(F_new - F, T, rng):
            tree, theta, F, info = tree_new, theta_new, F_new, info_new
            n_acc += 1
            if F < best_F:
                best_F, best = F, _snap(info)
                best_topo = t.tree_to_bracket(tree)
                best_paths = info.get("all_paths")

    runtime = time.time() - t_start
    success = bool(best_F < 1.0e5)
    segs = t.paths_to_noise_segments(best_paths) if (success and best_paths) else None
    gc.collect()          # 回收本次链的循环垃圾，防止 worker RSS 跨链累积
    return dict(cfg=cfg["name"], layer=cfg["layer"], group=cfg["group"],
                param=cfg["param"], level=cfg["level"], note=cfg.get("note", ""),
                w_N=float(w_n), seed=seed,
                F_best=best_F if success else None,
                N_exposed=best["N"] if success else None,
                L_total=best["L"] if success else None,
                topology=best_topo if success else None,
                segments=segs, n_accepted=n_acc,
                runtime_s=runtime, success=success, timed_out=timed_out,
                init_valid=init_valid, model=cfg["model"], nbr=cfg["nbr"])


# ============================================================
# 批处理：分块 + 断点续跑 + 失败隔离
# ============================================================

def _task_key(layer, cfg, w_n, seed):
    return "%s|%s|%.1f|%d" % (layer, cfg, w_n, seed)


def _load_done():
    done = set()
    if os.path.exists(TRACKS_PKL):
        try:
            with open(TRACKS_PKL, "rb") as f:
                for r in pickle.load(f):
                    done.add(_task_key(r.get("layer", ""), r["cfg"],
                                       r["w_N"], r["seed"]))
        except Exception as e:
            log("  [警告] 读取已有航迹失败，将从头跑: %s" % e)
    return done


def _append_tracks(rows):
    old = []
    if os.path.exists(TRACKS_PKL):
        with open(TRACKS_PKL, "rb") as f:
            old = pickle.load(f)
    old.extend(rows)
    with open(TRACKS_PKL, "wb") as f:
        pickle.dump(old, f)


def _run_batch(cfgs, weights, label):
    seeds = list(range(N_SEEDS))
    done = _load_done()
    todo = []
    for c in cfgs:
        for w in weights:
            for s in seeds:
                if _task_key(c["layer"], c["name"], w, s) not in done:
                    todo.append((c, w, s))
    total = len(cfgs) * len(weights) * len(seeds)
    if total - len(todo):
        log("  %s: 断点续跑，跳过已完成 %d 条" % (label, total - len(todo)))
    if not todo:
        log("  %s: 全部已完成" % label)
        return []

    rows, t_wall = [], time.time()
    for ci in range(0, len(todo), CHUNK):
        chunk = todo[ci:ci + CHUNK]
        got = []
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(chunk)),
                                 initializer=_init_worker) as ex:
            futs = {ex.submit(_run_chain, tk): tk for tk in chunk}
            for fut in as_completed(futs):
                tk = futs[fut]
                try:
                    got.append(fut.result())
                except Exception as e:
                    log("  [%s] 整条链丢失: %s w_N=%s seed=%s"
                        % (type(e).__name__, tk[0]["name"], tk[1], tk[2]))
                    got.append(_fail_row(tk[0], tk[1], tk[2], type(e).__name__))
        rows.extend(got)
        _append_tracks(got)
        gc.collect()
        ok = sum(1 for r in rows if r["success"])
        log("  [%s %d/%d] ok=%d/%d  墙钟 %.1f min%s"
            % (label, ci + len(chunk), len(todo), ok, len(rows),
               (time.time() - t_wall) / 60, _mem_tag()))
    log("  %s 完成 %d 链，墙钟 %.1f min" % (label, len(rows),
                                          (time.time() - t_wall) / 60))
    return rows


STAGE_CFG = {
    "e1": (cfg_e1, WN_E1, "E1"),
    "e2": (cfg_e2, [WN_MAIN], "E2"),
    "e3": (cfg_e3, [WN_MAIN], "E3"),
    "e4": (cfg_e4, [WN_MAIN], "E4"),
    "e5": (cfg_e5, [WN_MAIN], "E5"),
}


def run_stage(name):
    fn, weights, label = STAGE_CFG[name]
    cfgs = fn()
    log("=" * 78)
    log("%s — %d 配置 x %d 权重 x %d 次 = %d 链"
        % (label, len(cfgs), len(weights), N_SEEDS,
           len(cfgs) * len(weights) * N_SEEDS))
    log("=" * 78)
    return _run_batch(cfgs, weights, label)


# ============================================================
# 统计工具
# ============================================================

def _med(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    return float(np.median(a)) if a.size else None


def _iqr(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size < 2:
        return None
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def _root_family(topo):
    """拓扑族 = 根节点左子树包含的进场点集合。

    ((1,2),((3,4),5)) 与 ((1,2),(3,(4,5))) 的根分裂同为 {1,2}，归为同族。
    """
    if not topo:
        return None
    s = str(topo).strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner, depth = s[1:-1], 0
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return "".join(sorted(c for c in inner[:i] if c.isdigit()))
    return None


def _modal_topo(rows):
    tc = {}
    for r in rows:
        if r.get("topology"):
            tc[r["topology"]] = tc.get(r["topology"], 0) + 1
    if not tc:
        return None, {}
    return max(tc.items(), key=lambda kv: kv[1])[0], tc


def _baseline_segments():
    import r2_1_vertical_profile as r21
    return r21._union_segments(r21._published_alt_map())


def _save_csv(rows, path):
    fields = ["cfg", "group", "param", "level", "w_N", "seed", "F_best",
              "N_exposed", "L_total", "topology", "runtime_s", "success",
              "timed_out", "nbr", "error"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r["cfg"], r["group"], r["param"], r["level"],
                        "%.1f" % r["w_N"], r["seed"],
                        "" if r["F_best"] is None else "%.6f" % r["F_best"],
                        "" if r["N_exposed"] is None else "%.0f" % r["N_exposed"],
                        "" if r["L_total"] is None else "%.2f" % r["L_total"],
                        r["topology"] or "", "%.1f" % r["runtime_s"],
                        r["success"], r["timed_out"], r["nbr"],
                        r.get("error", "")])
    log("  明细 -> %s" % path)


# ============================================================
# 报告
# ============================================================

def stage_report():
    import test as t
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

    with open(TRACKS_PKL, "rb") as f:
        rows = pickle.load(f)
    pop, _ = t.load_population_grid(POP_FILE)
    base_segs = _baseline_segments()
    log("载入航迹 %d 条；基线段 %d 段" % (len(rows), len(base_segs)))

    for lay in ("E1", "E2", "E3", "E4", "E5"):
        sub = [r for r in rows if r.get("layer") == lay]
        if sub:
            _save_csv(sub, os.path.join(RESULT_DIR, "r2_8_%s.csv" % lay))

    def mk(mk_kw):
        nm = t.SELNoiseModel(_coarsen_block(pop, mk_kw["coarse"]),
                             SEL_threshold=float(mk_kw["thr"]),
                             T=float(mk_kw["T"]), RH=float(mk_kw["RH"]),
                             v_approach=float(mk_kw["v"]),
                             L_A_ref=float(mk_kw["L_A_ref"]),
                             f_ref=float(mk_kw["f_ref"]))
        if not mk_kw["lateral"]:
            nm._lateral_attenuation_db = lambda beta: (
                0.0 if np.ndim(beta) == 0
                else np.zeros_like(np.asarray(beta, float)))
        return nm

    def ok(rs, layer=None, cfg=None, w=None):
        out = [r for r in rs if r.get("success") and r.get("segments")]
        if layer:
            out = [r for r in out if r.get("layer") == layer]
        if cfg:
            out = [r for r in out if r["cfg"] == cfg]
        if w is not None:
            out = [r for r in out if r["w_N"] == w]
        return out

    # ---- 锚点 ----
    anchor = ok(rows, layer="E1", cfg=ANCHOR_NAME, w=WN_MAIN)
    nm65 = mk(MODEL_ANCHOR)
    n0_65, _, _ = nm65.exposed_population(base_segs)
    log("自校验: N0(65dB) = %s (期望 6,169,767, 偏差 %s)"
        % (format(n0_65, ",.0f"), format(n0_65 - 6169767, "+,.0f")))
    aN = _med([r["N_exposed"] for r in anchor])
    aL = _med([r["L_total"] for r in anchor])
    a_topo, a_dist = _modal_topo(anchor)
    a_fam = _root_family(a_topo)
    log("锚点 w_N=%.1f: n=%d  N=%.0f  L=%.1f  拓扑=%s (族 %s)"
        % (WN_MAIN, len(anchor), aN, aL, a_topo, a_fam))

    # ---- 表 A：阈值基线重算 + 事后重分类 ----
    tableA, fig7 = [], []
    for thr in THRESHOLDS:
        nmm = mk(dict(MODEL_ANCHOR, thr=thr))
        n0, _, _ = nmm.exposed_population(base_segs)
        rec = dict(thr=thr, N0=n0)
        for w in WN_E1:
            g = ok(rows, layer="E1", cfg=ANCHOR_NAME, w=w)
            vals = [nmm.exposed_population(r["segments"])[0] for r in g]
            nmd = _med(vals)
            rec["N_%.1f" % w] = nmd
            rec["R_%.1f" % w] = ((n0 - nmd) / n0 * 100.0
                                 if nmd is not None else None)
            fig7.append(dict(series="w_N=%.1f" % w, thr=thr, N=nmd,
                             N_q1=None, N_q3=None, kind="reclassification"))
        fig7.append(dict(series="baseline", thr=thr, N=n0, N_q1=None,
                         N_q3=None, kind="baseline"))
        tableA.append(rec)
        log("  thr=%.0fdB N0=%12s   %s" % (thr, format(n0, ",.0f"),
            "  ".join("R(w=%.1f)=%s" % (w, "n/a" if rec["R_%.1f" % w] is None
                                        else "%+.2f%%" % rec["R_%.1f" % w])
                      for w in WN_E1)))

    # ---- 表 B：各阈值完整重优化（65 dB 复用 E1 锚点） ----
    tableB = []
    for thr in THRESHOLDS:
        if thr == 65.0:
            g, n0 = anchor, n0_65
        else:
            g = ok(rows, layer="E2", cfg="THR%d" % int(thr), w=WN_MAIN)
            if not g:
                continue
            n0 = [r["N0"] for r in tableA if r["thr"] == thr][0]
        Ns = [r["N_exposed"] for r in g]
        Ls = [r["L_total"] for r in g]
        topo, dist = _modal_topo(g)
        nmd, lmd = _med(Ns), _med(Ls)
        tableB.append(dict(
            thr=thr, n=len(g), topo=topo,
            topo_dist=" / ".join("%s:%d" % kv for kv in
                                 sorted(dist.items(), key=lambda kv: -kv[1])),
            N=nmd, N_std=float(np.std(Ns, ddof=1)) if len(Ns) > 1 else 0.0,
            L=lmd, L_std=float(np.std(Ls, ddof=1)) if len(Ls) > 1 else 0.0,
            R_N=(n0 - nmd) / n0 * 100.0,
            R_L=(829.2 - lmd) / 829.2 * 100.0))
        fig7.append(dict(series="reopt w_N=%.1f" % WN_MAIN, thr=thr,
                         N=nmd,
                         N_q1=float(np.percentile(Ns, 25)),
                         N_q3=float(np.percentile(Ns, 75)),
                         kind="reoptimized"))

    # ---- 采样密度（纯评估侧，零成本） ----
    ns_rows = []
    for ns in N_SAMPLES_LEVELS:
        vals = [nm65.exposed_population(r["segments"], n_samples=ns)[0]
                for r in anchor]
        ns_rows.append(dict(n_samples=ns, N=_med(vals), IQR=_iqr(vals)))

    # ---- 表 C：单因素 ----
    def summarize(g, n0):
        Ns = [r["N_exposed"] for r in g]
        Ls = [r["L_total"] for r in g]
        topo, dist = _modal_topo(g)
        nmd, lmd = _med(Ns), _med(Ls)
        feas = len(g) / float(len(g) + sum(
            1 for r in rows if r.get("layer") == g[0]["layer"]
            and r["cfg"] == g[0]["cfg"] and r["w_N"] == g[0]["w_N"]
            and not r["success"])) * 100.0
        return dict(
            n=len(g), feasible=feas, N=nmd, L=lmd, topo=topo,
            topo_dist=" / ".join("%s:%d" % kv for kv in
                                 sorted(dist.items(), key=lambda kv: -kv[1])),
            dN=(nmd - aN) / aN * 100.0, dL=lmd - aL,
            R_N=(n0 - nmd) / n0 * 100.0, R_L=(829.2 - lmd) / 829.2 * 100.0,
            topo_same="是" if topo == a_topo else "否",
            family_same="是" if _root_family(topo) == a_fam else "否",
            runtime=_med([r["runtime_s"] for r in g]))

    tableC = []
    for c in cfg_e3() + cfg_e4():
        g = ok(rows, layer=c["layer"], cfg=c["name"], w=WN_MAIN)
        if not g:
            tableC.append(dict(cfg=c["name"], group=c["group"],
                               param=c["param"], level=c["level"],
                               feasible=0.0, n=0))
            continue
        s = summarize(g, n0_65)
        s.update(cfg=c["name"], group=c["group"], param=c["param"],
                 level=c["level"], note=c.get("note", ""))
        tableC.append(s)

    # ---- 表 D：联合稳健性 ----
    tableD = []
    scen = [dict(name=ANCHOR_NAME, desc="J0 基准（默认）", layer="E1")] + \
           [dict(name=c["name"], desc=c["note"], layer="E5") for c in cfg_e5()]
    for sc in scen:
        g = ok(rows, layer=sc["layer"], cfg=sc["name"], w=WN_MAIN)
        if not g:
            tableD.append(dict(cfg=sc["name"], desc=sc["desc"], infeasible=100.0))
            continue
        nmm = mk(g[0]["model"])
        n0, _, _ = nmm.exposed_population(base_segs)
        s = summarize(g, n0)
        s.update(cfg=sc["name"], desc=sc["desc"],
                 dominate="是" if (s["R_N"] > 0 and s["R_L"] > 0) else "否")
        tableD.append(s)
    if tableD:
        fin = [r for r in tableD if r.get("S_T_family") is None and r.get("n")]
        if fin:
            sd = np.mean([1.0 if r["dominate"] == "是" else 0.0 for r in fin]) * 100
            st = np.mean([1.0 if r["family_same"] == "是" else 0.0
                          for r in fin]) * 100
            rn = [r["R_N"] for r in fin]
            rl = [r["R_L"] for r in fin]
            log("\n联合稳健性: S_D=%.0f%%  S_T(族)=%.0f%%  "
                "R_N 范围 [%+.1f%%, %+.1f%%]  R_L 范围 [%+.1f%%, %+.1f%%]"
                % (sd, st, min(rn), max(rn), min(rl), max(rl)))

    # ---- 写 xlsx ----
    hf = Font(bold=True, size=11)
    fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    bd = Border(*[Side(style="thin")] * 4)

    def sheet(ws, header, body, widths=None, numfmt=None):
        ws.append(header)
        for c in ws[1]:
            c.font, c.fill, c.border = hf, fill, bd
            c.alignment = Alignment(horizontal="center")
        for r in body:
            ws.append(r)
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.border = bd
                if numfmt and c.column in numfmt:
                    c.number_format = numfmt[c.column]
        if widths:
            for i, wd in enumerate(widths, 1):
                ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = wd

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "A_阈值重分类"
    hdr = ["SEL(dB)", "基线 N0"]
    for w in WN_E1:
        hdr += ["w_N=%.1f N" % w, "w_N=%.1f R_N(%%)" % w]
    sheet(ws, hdr, [[r["thr"], r["N0"]] +
                    sum([[r.get("N_%.1f" % w), r.get("R_%.1f" % w)]
                         for w in WN_E1], []) for r in tableA],
          [10, 14] + [15] * (2 * len(WN_E1)), {2: "#,##0"})

    ws = wb.create_sheet("B_阈值重优化")
    sheet(ws, ["SEL(dB)", "n", "最优拓扑", "N_中位数", "N_std", "L_中位数",
               "L_std", "R_N(%)", "R_L(%)", "拓扑分布"],
          [[r["thr"], r["n"], r["topo"], r["N"], r["N_std"], r["L"],
            r["L_std"], r["R_N"], r["R_L"], r["topo_dist"]] for r in tableB],
          [9, 5, 20, 12, 10, 10, 10, 9, 9, 44])

    ws = wb.create_sheet("C_单因素与离散化")
    sheet(ws, ["配置", "类别", "参数", "水平", "可行率(%)", "N_中位数", "L_中位数",
               "ΔN vs 锚点(%)", "ΔL vs 锚点(km)", "R_N(%)", "R_L(%)",
               "众数拓扑", "拓扑不变", "族不变", "中位耗时(s)"],
          [[r.get("cfg"), r.get("group"), r.get("param"), r.get("level"),
            r.get("feasible"), r.get("N"), r.get("L"), r.get("dN"),
            r.get("dL"), r.get("R_N"), r.get("R_L"), r.get("topo"),
            r.get("topo_same"), r.get("family_same"), r.get("runtime")]
           for r in tableC],
          [20, 12, 20, 20, 11, 12, 10, 15, 16, 9, 9, 20, 9, 8, 12],
          {6: "#,##0"})

    ws = wb.create_sheet("D_联合稳健性")
    sheet(ws, ["情景", "说明", "可行率(%)", "支配保持", "N_中位数", "L_中位数",
               "R_N(%)", "R_L(%)", "众数拓扑", "拓扑族", "族不变"],
          [[r.get("cfg"), r.get("desc"), r.get("feasible"), r.get("dominate"),
            r.get("N"), r.get("L"), r.get("R_N"), r.get("R_L"),
            r.get("topo"), _root_family(r.get("topo")), r.get("family_same")]
           for r in tableD],
          [10, 26, 11, 10, 12, 10, 9, 9, 20, 8, 8], {5: "#,##0"})

    ws = wb.create_sheet("n_samples")
    sheet(ws, ["采样点数", "N_中位数", "IQR"],
          [[r["n_samples"], r["N"], r["IQR"]] for r in ns_rows],
          [12, 14, 12], {2: "#,##0"})

    wb.save(OUT_XLSX)
    log("\n汇总 -> %s" % os.path.abspath(OUT_XLSX))

    with open(FIG7_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["series", "thr", "N", "N_q1",
                                          "N_q3", "kind"])
        w.writeheader()
        for r in fig7:
            w.writerow(r)
    log("图 7 数据 -> %s" % os.path.abspath(FIG7_CSV))

    log("\n── C 表摘要 ──")
    for r in tableC:
        if not r.get("N"):
            log("  %-20s 无可行解" % r.get("cfg"))
            continue
        log("  %-20s ΔN=%+6.2f%%  ΔL=%+6.1fkm  可行=%.0f%%  拓扑不变=%s  族不变=%s"
            % (r["cfg"], r["dN"], r["dL"], r["feasible"],
               r["topo_same"], r["family_same"]))
    log("\n── D 表摘要 ──")
    for r in tableD:
        if not r.get("N"):
            log("  %-8s %-24s 无可行解" % (r.get("cfg"), r.get("desc", "")))
            continue
        log("  %-8s %-24s R_N=%+6.2f%%  R_L=%+6.2f%%  支配=%s  族不变=%s"
            % (r["cfg"], r.get("desc", ""), r["R_N"], r["R_L"],
               r["dominate"], r["family_same"]))


# ============================================================
# 主入口
# ============================================================

def main():
    args = sys.argv[1:]
    stage = "all"
    if "--stage" in args:
        stage = args[args.index("--stage") + 1]
    os.makedirs(RESULT_DIR, exist_ok=True)
    log("锚点: RHO_ASTAR_MAX=%.2f  SECTOR_R_MIN_FRAC=%.2f  MAX_TURN=%d  "
        "MIN_TURN_DIST=%d  w_N=%.1f  workers=%d"
        % (_DEFAULTS["RHO_ASTAR_MAX"], _DEFAULTS["SECTOR_R_MIN_FRAC"],
           _DEFAULTS["MAX_TURN_ANGLE"], _DEFAULTS["MIN_TURN_DISTANCE"],
           WN_MAIN, N_WORKERS))
    t0 = time.time()
    if stage == "all":
        for s in ("e1", "e2", "e3", "e4", "e5"):
            run_stage(s)
        stage_report()
    elif stage in STAGE_CFG:
        run_stage(stage)
    elif stage == "report":
        stage_report()
    else:
        log("未知 stage: %s; 可选 e1..e5 / report / all" % stage)
        return
    log("\n总墙钟 %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()

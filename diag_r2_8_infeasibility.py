# -*- coding: utf-8 -*-
"""诊断：严格结构情景下不可行的具体原因。

把 evaluate_X 拆成两段分别统计：
  A. 几何检查 _check_node_distances（汇聚点/子节点间距 < MIN_TURN_DISTANCE）—— 不需 A*
  B. A* 解码 plan_hybrid_paths（含绕行预算与转弯角约束）—— 需要 A*
"""
import copy
import numpy as np
import test as t
import r2_8_robustness as m

m._init_worker()
G = m._G
NP, IFPT, P = G["NP"], G["IFPT"], G["P"]

CASES = [
    ("锚点", {}),
    ("仅 扇环=0.75", dict(SECTOR_R_MIN_FRAC=0.75)),
    ("仅 绕行=1.1", dict(RHO_ASTAR_MAX=1.1)),
    ("仅 转弯=45", dict(MAX_TURN_ANGLE=45)),
    ("仅 间距=7", dict(MIN_TURN_DISTANCE=7)),
    ("严格组合", dict(SECTOR_R_MIN_FRAC=0.75, RHO_ASTAR_MAX=1.1,
                  MAX_TURN_ANGLE=45, MIN_TURN_DISTANCE=7)),
]

N_GEO = 200
N_AST = 20


def node_dists_ok(decoded):
    def rec(node):
        if node is None or node.is_entry:
            return True
        for ch in (node.left, node.right):
            if ch is None:
                continue
            if np.hypot(node.x - ch.x, node.y - ch.y) < t.MIN_TURN_DISTANCE:
                return False
        return rec(node.left) and rec(node.right)
    return rec(decoded)


print("%-14s %10s %10s %10s %10s" %
      ("配置", "几何失败", "A*失败", "成功", "最小间距中位"))
print("-" * 62)
for name, ov in CASES:
    m._apply_overrides(ov, 3)
    nm, astar = m._get_model(m.MODEL_ANCHOR)
    astar._nbrs = m._nbrs_for_radius(3)
    astar._cache = {}

    rng = np.random.RandomState(0)
    geo_fail = ok = 0
    mind = []
    decs = []
    for _ in range(N_GEO):
        tree = copy.deepcopy(G["trees"][rng.randint(0, len(G["trees"]))])
        dim = t.count_internal_params(tree)
        th = np.full(dim, 0.5)
        d = t.decode_tree(copy.deepcopy(tree), list(th), NP, IFPT)
        # 记录最小间距
        best = [9e9]

        def rec(n):
            if n is None or n.is_entry:
                return
            for ch in (n.left, n.right):
                if ch is not None:
                    best[0] = min(best[0], float(np.hypot(n.x - ch.x, n.y - ch.y)))
            rec(n.left)
            rec(n.right)
        rec(d)
        mind.append(best[0])
        if node_dists_ok(d):
            ok += 1
            if len(decs) < N_AST:
                decs.append(d)
        else:
            geo_fail += 1

    astar_fail = astar_ok = 0
    for d in decs:
        astar._cache = {}
        _, is_valid, _ = t.plan_hybrid_paths(
            d, G["pf"], IFPT, P, astar=astar, astar_segment_id=True,
            max_turns=t.MAX_TURNS_PER_SEGMENT)
        if is_valid:
            astar_ok += 1
        else:
            astar_fail += 1

    print("%-14s %7d/%d %7d/%d %7d  %8.2f km" %
          (name, geo_fail, N_GEO, astar_fail, len(decs), astar_ok,
           np.median(mind)))

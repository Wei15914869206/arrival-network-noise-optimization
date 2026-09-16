# -*- coding: utf-8 -*-
"""
r2_1_vertical_profile.py — R2-1 垂直剖面单因子对比（A / B / D 三格）

A = 已发布水平航路 + 已发布阶梯剖面   → N_A（基线）
B = 已发布水平航路 + 3° CDO 剖面      → N_B（水平不动，只换垂直剖面）
D = 优化水平航路   + 3° CDO 剖面      → N_D(w_N)（现有三权重结果）

分解（单因子、另一因子固定）：
    B − A = 垂直剖面贡献（水平 = 已发布，固定）
    D − B = 水平重构贡献（垂直 = 3° CDO，固定）
总效应 D − A = (B−A) + (D−B)。

B 格与 w_N 无关（已发布水平 + 3° CDO），故 A、B 三行相同，仅 D 随 w_N 变化。
D 值来自 run_wn_trajectories.py 的 seed=0 运行（确定性，两次运行结果一致）。

输出：result/r2_1_vertical_profile.xlsx + 控制台表
"""

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")

import test as t
import viz_procedures as vp

TAN3 = math.tan(math.radians(3.0))     # 3° 连续下降
IF_HEIGHT = 900.0                       # 与 test.py 的 IF_HEIGHT 一致

# D 格：run_wn_trajectories.py (seed=0) 三权重结果
D_RESULTS = {
    0.0: dict(N=3193883, L=762.6),
    0.5: dict(N=1815732, L=791.2),
    1.0: dict(N=1670874, L=860.0),
}


def _grid_km(name_a, name_b):
    """两个航路点间的网格距离 [km]（1 格 = 1 km）。"""
    xa, ya = vp.latlon_to_grid(vp._WP[name_a][0], vp._WP[name_a][1])
    xb, yb = vp.latlon_to_grid(vp._WP[name_b][0], vp._WP[name_b][1])
    return math.hypot(xa - xb, ya - yb)


def _published_alt_map():
    """A 格高度：与 viz_procedures 的并集口径完全一致（按排序后的唯一航路点插值）。"""
    names = set()
    for route in vp._FULL_ROUTE_NAMES.values():
        names.update(route)
    items = [(n, vp._WP[n][0], vp._WP[n][1], vp._WP[n][2]) for n in sorted(names)]
    return {n: h for (n, _, _, h) in vp.interpolate_heights(items)}


def _cdo_alt_map():
    """B 格高度：以 IF(900m) 为根，沿已发布航路向外按 3° 爬升，封顶于子树最低进场点高度；
    进场点强制取已发布进场高度（与 test.py 的 assign_heights / override_entry_heights 同规则）。"""
    parent = {}
    for route in vp._FULL_ROUTE_NAMES.values():
        for a, b in zip(route, route[1:]):
            parent[a] = b                     # a 朝跑道方向的父节点是 b
    child = {}
    for c, p in parent.items():
        child.setdefault(p, []).append(c)

    def subtree_cap(node):
        """子树内进场点的最低已发布高度（到达该高度即平飞）。"""
        kids = child.get(node)
        if not kids:
            return vp._WP[node][2]
        return min(subtree_cap(k) for k in kids)

    alt = {"IF": IF_HEIGHT, "PUD": 0.0}
    stack = ["IF"]
    while stack:
        n = stack.pop()
        for c in child.get(n, []):
            if c == "PUD":
                continue
            alt[c] = min(alt[n] + _grid_km(n, c) * TAN3 * 1000.0, subtree_cap(c))
            stack.append(c)

    # 进场点强制取已发布进场高度（与 test.py 一致）
    for n in list(alt):
        if n != "IF" and n != "PUD" and n not in child:
            alt[n] = float(vp._WP[n][2])
    return alt


def _union_segments(alt_map):
    """并集航段（航路点对去重）——与 viz_procedures 同一口径，共享段只计一次。"""
    pairs = set()
    for route in vp._FULL_ROUTE_NAMES.values():
        for a, b in zip(route, route[1:]):
            pairs.add((a, b))
    segs = []
    for n1, n2 in sorted(pairs):
        x1, y1 = vp.latlon_to_grid(vp._WP[n1][0], vp._WP[n1][1])
        x2, y2 = vp.latlon_to_grid(vp._WP[n2][0], vp._WP[n2][1])
        segs.append((x1, y1, alt_map[n1], x2, y2, alt_map[n2]))
    return segs


def main():
    pop, _ = t.load_population_grid("population_data_square.xlsx")
    nm = t.SELNoiseModel(pop)

    alt_pub = _published_alt_map()
    alt_cdo = _cdo_alt_map()

    n_a, _, _ = nm.exposed_population(_union_segments(alt_pub))
    n_b, _, _ = nm.exposed_population(_union_segments(alt_cdo))

    print("=" * 78)
    print("R2-1 垂直剖面单因子对比：A（已发布水平/已发布剖面） vs B（已发布水平/3°CDO）")
    print("=" * 78)
    print(f"  A  已发布水平 + 已发布阶梯剖面 : N = {n_a:,.0f}")
    print(f"  B  已发布水平 + 3° CDO 剖面    : N = {n_b:,.0f}")
    print(f"  垂直剖面贡献 B−A               : {n_b - n_a:+,.0f} "
          f"({(n_b - n_a) / n_a * 100:+.2f}%)")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("三格汇总（A / B / D），D 为 run_wn_trajectories.py seed=0")
    print("=" * 78)
    hdr = f"{'配置':<26}{'水平':<12}{'垂直':<18}{'N 暴露人口':>14}{'R_N vs A':>12}"
    print(hdr)
    print("-" * 78)
    print(f"{'A ':<24}{'已发布':<12}{'已发布阶梯':<18}{n_a:>14,.0f}{'—':>12}")
    r_b = (n_a - n_b) / n_a * 100.0
    print(f"{'B  ':>2}{'':<22}{'已发布':<12}{'3° CDO':<18}{n_b:>14,.0f}{r_b:>+11.2f}%")
    for w in (0.0, 0.5, 1.0):
        nd = D_RESULTS[w]["N"]
        r_d = (n_a - nd) / n_a * 100.0
        print(f"{'D (w_N=%.1f)' % w:<24}{'优化':<12}{'3° CDO':<18}{nd:>14,.0f}{r_d:>+11.2f}%")
    print("-" * 78)

    print("\n" + "=" * 78)
    print("效应分解（单因子，另一因子固定；绝对增量可加）")
    print("=" * 78)
    print(f"{'':<10}{'垂直 ΔN(A→B)':>16}{'水平 ΔN(B→D)':>16}{'总 ΔN(A→D)':>16}"
          f"{'垂直占比':>10}{'水平占比':>10}")
    print("-" * 78)
    dn_vert = n_a - n_b
    for w in (0.0, 0.5, 1.0):
        nd = D_RESULTS[w]["N"]
        dn_horiz = n_b - nd
        dn_total = n_a - nd
        print(f"{'w_N=%.1f' % w:<10}{dn_vert:>16,.0f}{dn_horiz:>16,.0f}"
              f"{dn_total:>16,.0f}{dn_vert / dn_total * 100:>9.1f}%"
              f"{dn_horiz / dn_total * 100:>9.1f}%")
    print("-" * 78)
    print("注：ΔN > 0 = 暴露人口减少（改善）。垂直效应对三档权重相同（1,820,715 人）。")
    print("    相对改善率 R_N：A→B +29.51%；A→D 见上表（分母不同，故不可直接相加）。")

    import openpyxl
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    wb = openpyxl.Workbook()
    hf = Font(bold=True, size=11)
    fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    bd = Border(*[Side(style='thin')] * 4)

    ws = wb.active
    ws.title = "ABC-D"
    rows = [
        ["配置", "水平航路", "垂直剖面", "N 暴露人口", "R_N vs A (%)"],
        ["A", "已发布", "已发布阶梯", round(n_a), None],
        ["B", "已发布", "3° CDO", round(n_b), round(r_b, 2)],
    ]
    for w in (0.0, 0.5, 1.0):
        nd = D_RESULTS[w]["N"]
        rows.append([f"D (w_N={w:.1f})", "优化", "3° CDO", nd,
                     round((n_a - nd) / n_a * 100.0, 2)])
    for ri, row in enumerate(rows, 1):
        for ci, v in enumerate(row, 1):
            c = ws.cell(row=ri, column=ci, value=v)
            c.border = bd
            if ri == 1:
                c.font = hf; c.fill = fill
                c.alignment = Alignment(horizontal='center')
        ws.cell(row=ri, column=4).number_format = '#,##0'
    for ci, wd in enumerate([16, 12, 16, 16, 14], 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = wd

    ws2 = wb.create_sheet("分解")
    ws2.append(["w_N",
                "垂直 ΔN (A→B)", "水平 ΔN (B→D)", "总 ΔN (A→D)",
                "垂直占比 (%)", "水平占比 (%)",
                "R_N 总改善 A→D (%)"])
    for c in ws2[1]:
        c.font = hf; c.fill = fill; c.border = bd
    for w in (0.0, 0.5, 1.0):
        nd = D_RESULTS[w]["N"]
        dn_vert = n_a - n_b
        dn_horiz = n_b - nd
        dn_total = n_a - nd
        ws2.append([w, dn_vert, dn_horiz, dn_total,
                    round(dn_vert / dn_total * 100, 1),
                    round(dn_horiz / dn_total * 100, 1),
                    round(dn_total / n_a * 100, 2)])
    for row in ws2.iter_rows(min_row=2):
        for c in row:
            c.border = bd
            if c.column in (2, 3, 4):
                c.number_format = '#,##0'
    for ci, wd in enumerate([10, 16, 16, 16, 14, 14, 20], 1):
        ws2.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = wd

    out = "result/r2_1_vertical_profile.xlsx"
    wb.save(out)
    print(f"\n结果: {out}")


if __name__ == "__main__":
    main()

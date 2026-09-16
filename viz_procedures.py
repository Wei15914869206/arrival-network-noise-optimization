# -*- coding: utf-8 -*-
"""
viz_procedures.py — 上海浦东机场进场程序航路点可视化（5条进场 + 1条进近）

所有航路点已录入（仅标注，暂不连线）：
  SASAN:  SASAN(6000) → EKIMU(插值) → JIUTING(3900) → IAF1(2400)
  ANDONG: ANDONG(6000) → IAF2(2700)
  MATNU:  MATNU(5100) → PD228(插值) → BEKOK(插值) → IAF3(1500)
  LISHE:  LISHE(6000) → SAMKI(5700) → BAVIK(插值) → IAF2(2700)
  DUMET:  DUMET(4800) → PD1(2400) → PD2(插值) → IAF3(1500)
  进近:   IAF1(2400)→APP1(1800)→APP2(1200)→IF(900)
          IAF2(2700)→IF(900)
          IAF3(1500)→APP3(1200)→APP4(1200, 由HSH推算)

输出：
  - 2D 平面图（含人口密度底图、PUD◆、FAF◆、航路点标注）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import openpyxl


# ============================================================
# 全局配置（与 test.py 一致）
# ============================================================

MAX_TURN_ANGLE = 60
MIN_TURN_DISTANCE = 5
GRID_SIZE_X = 296
GRID_SIZE_Y = 296
GRID_UNIT_METERS = 1000
FAF_HEIGHT = 500

POP_LON_WEST = 120.320834
POP_LAT_SOUTH = 29.520833
GRID_STEP_DEG = 1.0 / 120.0


def latlon_to_grid(lat, lon):
    x = (lon - POP_LON_WEST) / GRID_STEP_DEG
    y = (lat - POP_LAT_SOUTH) / GRID_STEP_DEG
    return (x, y)


def dms(d, m, s=0.0):
    return d + m / 60.0 + s / 3600.0


def calculate_distance(p1, p2):
    """网格欧氏距离——仅用于插值参数计算（非物理距离）。"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def grid_to_latlon(x, y):
    """逆映射：网格坐标 → (lat, lon)，用于球面距离计算。"""
    lat = y * GRID_STEP_DEG + POP_LAT_SOUTH
    lon = x * GRID_STEP_DEG + POP_LON_WEST
    return (lat, lon)


def _haversine_km(lat1, lon1, lat2, lon2):
    """球面距离 [km]（WGS84 平均半径 R=6371 km）。"""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _grid_path_length_km(path):
    """折线路径的球面总长度 [km]，共享航段按出现次数重复计入。

    path: [(x,y,...), ...] 网格坐标（仅用前两维）。
    """
    total = 0.0
    for i in range(len(path) - 1):
        lat1, lon1 = grid_to_latlon(path[i][0], path[i][1])
        lat2, lon2 = grid_to_latlon(path[i + 1][0], path[i + 1][1])
        total += _haversine_km(lat1, lon1, lat2, lon2)
    return total


# ============================================================
# 人口加载与 SEL 噪声模型（直接复用 test.py 的 SELNoiseModel）
# ============================================================

import test as _test_module  # 复用现有模块避免重复代码


def load_population_grid(path):
    return _test_module.load_population_grid(path)


SELNoiseModel = _test_module.SELNoiseModel


# PUD 与 FAF 坐标（见下方，使用 _great_circle_destination 计算）


# ============================================================
# 工具函数
# ============================================================

def _great_circle_destination(lat_deg, lon_deg, bearing_deg, dist_km):
    """球面几何：从 (lat, lon) 沿 bearing 走 dist_km 后的位置。"""
    R = 6371.0
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    dR = dist_km / R
    lat2 = math.asin(math.sin(lat1) * math.cos(dR)
                     + math.cos(lat1) * math.sin(dR) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(dR) * math.cos(lat1),
                             math.cos(dR) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def _great_circle_destination_hdg_to_pt(lat_a, lon_a, pt_b, dist_km):
    """从 A 出发，沿大圆路径走向 B，走 dist_km 后的位置。"""
    # 计算 A→B 的初始方位
    p1, p2 = math.radians(lat_a), math.radians(pt_b[0])
    dl = math.radians(pt_b[1] - lon_a)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = math.degrees(math.atan2(y, x))
    return _great_circle_destination(lat_a, lon_a, bearing, dist_km)


def _gc_intersect_radial(origin, bearing_deg, target, target_dist_nm):
    """沿 origin 的 bearing 方向搜索，找到距 target 为 target_dist_nm 的点。"""
    lo, hi = 0.0, 300.0  # km
    for _ in range(50):
        mid = (lo + hi) / 2
        pt = _great_circle_destination(*origin, bearing_deg, mid)
        d = _haversine(*pt, *target)
        if d < target_dist_nm * 1.852:
            lo = mid
        else:
            hi = mid
    return _great_circle_destination(*origin, bearing_deg, (lo + hi) / 2)


def _haversine(lat1, lon1, lat2, lon2):
    """球面距离 [km]——用于 _gc_intersect_radial 的目标距离判断。"""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# 进场程序航路点定义（5条进场 + 1条进近）
# ============================================================

# 格式: list of (名字, lat_deg, lon_deg, height_m)

# 已解算的参考点
_XSY   = (dms(30, 55, 54), dms(121, 52, 24))   # XSY = IF
_HSH   = (dms(31, 22,  6), dms(121, 50, 48))   # HSH ≠ XSY，用于 APP4 推算
_ANDONG_LL = (dms(30, 15, 24), dms(121, 13, 18))
_DUMET_LL  = (dms(31, 21, 28), dms(122, 46, 30))
_PUD_LL    = (dms(31, 10, 18), dms(121, 47,  0))
_IAF1_LL   = (dms(31,  7, 48), dms(121, 40, 18))

_IAF2_LL   = _great_circle_destination(*_ANDONG_LL,  62.0, 32.6 * 1.852)
_IAF3_LL   = _great_circle_destination(*_XSY,        50.0, 15.4 * 1.852)
_APP1_LL   = _great_circle_destination(*_IAF1_LL,   168.0,  9.5 * 1.852)
_APP2_LL   = _great_circle_destination(*_IAF1_LL,   168.0, 18.4 * 1.852)
_APP3_LL   = _great_circle_destination(*_XSY,        50.0,  8.6 * 1.852)
_APP4_LL   = _great_circle_destination(*_HSH,       168.0, 28.3 * 1.852)
_PD1_LL    = _gc_intersect_radial(_DUMET_LL, 262.0, _PUD_LL, 31.3)
_PD2_LL    = _gc_intersect_radial(_PUD_LL,   82.0, _XSY,    27.1)

# PUD 网格坐标与 FAF（FAF 在 PUD→IF 大圆路径上，距 PUD 11.8 NM）
P = np.array(latlon_to_grid(*_PUD_LL))
_FAF_LL = _great_circle_destination_hdg_to_pt(*_PUD_LL, _XSY, 11.8 * 1.852)
FAF = np.array(latlon_to_grid(*_FAF_LL))

# --- 航路点主字典 (name -> (lat, lon, height)) ---

_WP = {
    # SASAN 线
    "SASAN":   (dms(31, 35, 22), dms(120, 19, 10), 6000),
    "EKIMU":   (dms(31, 21,  6), dms(121,  6, 36), None),
    "PD1":     (*_PD1_LL,   2400),
    "JIUTING": (dms(31,  7, 24), dms(121, 20, 30), 3900),
    "IAF1":    (*_IAF1_LL,  2400),
    "APP1":    (*_APP1_LL,  1800),
    "APP2":    (*_APP2_LL,  1200),
    "IF":      (*_XSY,        900),
    # ANDONG 线
    "ANDONG":  (*_ANDONG_LL, 6000),
    "SAMKI":   (dms(30, 15, 12), dms(121, 33, 30), 5700),
    "BAVIK":   (dms(30, 22,  0), dms(121, 37, 54), None),
    "IAF2":    (*_IAF2_LL,   2700),
    # LISHE
    "LISHE":   (dms(29, 53, 42), dms(121, 20,  0), 6000),
    # MATNU 线
    "MATNU":   (dms(31, 39, 36), dms(122, 38,  0), 5100),
    "PD228":   (dms(31, 32, 54), dms(122, 41, 12), None),
    "BEKOK":   (dms(31, 27,  0), dms(122, 27,  0), None),
    "PD2":     (*_PD2_LL,   None),
    "IAF3":    (*_IAF3_LL,  1500),
    "APP3":    (*_APP3_LL,  1200),
    "APP4":    (*_APP4_LL,  1200),
    # DUMET
    "DUMET":   (*_DUMET_LL, 4800),
    # HSH (参考点标记)
    "HSH":     (*_HSH,       None),
    "PUD":     (*_PUD_LL,       0),
}

# --- 进场程序连线定义 ---

_PROCEDURE_NAMES = [
    ("SASAN Arrival",  ["SASAN","EKIMU","PD1","JIUTING","IAF1","APP1","APP2","IF"],  "#1f77b4"),
    ("ANDONG Arrival", ["ANDONG","SAMKI","BAVIK","IAF2","IF"],                        "#d62728"),
    ("LISHE Arrival",  ["LISHE","SAMKI"],                                             "#2ca02c"),
    ("MATNU Arrival",  ["MATNU","PD228","BEKOK","PD2","IAF3","APP3","APP4","IF"],    "#ff7f0e"),
    ("DUMET Arrival",  ["DUMET","BEKOK"],                                             "#9467bd"),
]

# 每条程序的完整航线（含合并后的共享段 + 最终进近），用于 L_total 计算。
# LISHE 在 SAMKI 汇入 ANDONG，DUMET 在 BEKOK 汇入 MATNU。
_FULL_ROUTE_NAMES = {
    "SASAN Arrival":  ["SASAN","EKIMU","PD1","JIUTING","IAF1","APP1","APP2","IF","PUD"],
    "ANDONG Arrival": ["ANDONG","SAMKI","BAVIK","IAF2","IF","PUD"],
    "LISHE Arrival":  ["LISHE","SAMKI","BAVIK","IAF2","IF","PUD"],
    "MATNU Arrival":  ["MATNU","PD228","BEKOK","PD2","IAF3","APP3","APP4","IF","PUD"],
    "DUMET Arrival":  ["DUMET","BEKOK","PD2","IAF3","APP3","APP4","IF","PUD"],
}

def _full_route_length_km(proc_name):
    """从 _FULL_ROUTE_NAMES 查完整航路点 →球面距离 [km]，共享航段按出现次数重复计入。"""
    names = _FULL_ROUTE_NAMES.get(proc_name)
    if names is None:
        return 0.0
    pts = []
    for n in names:
        lat, lon = _WP[n][0], _WP[n][1]
        x, y = latlon_to_grid(lat, lon)
        pts.append((x, y))
    return _grid_path_length_km(pts)


def _build_full_route_segments(proc_name):
    """从 _FULL_ROUTE_NAMES 构建完整航线的噪声段 [(x1,y1,h1,x2,y2,h2), ...]。

    含共享下游段 + 最终进近（IF→PUD），高度按已知航路点线性插值填 None 孔。
    返回 segments 列表，供噪声评估（每条程序独立评估）。
    """
    names = _FULL_ROUTE_NAMES.get(proc_name)
    if names is None:
        return []
    items = [(n, _WP[n][0], _WP[n][1], _WP[n][2]) for n in names]
    proc_h = interpolate_heights(items)
    segments, _, _ = build_procedure_paths(proc_h)
    return segments

# 构建 (名字, list_of_points, 颜色) 列表 + 独立标注点列表
ALL_PROCEDURES = []
EXTRA_WAYPOINTS = []
_seen_wp = set()
for _pname, _names, _color in _PROCEDURE_NAMES:
    _pts = []
    for _n in _names:
        _lat, _lon, _h = _WP[_n]
        _pts.append((_n, _lat, _lon, _h))
        if _n not in _seen_wp:
            EXTRA_WAYPOINTS.append((_n, _lat, _lon, _h))
            _seen_wp.add(_n)
    ALL_PROCEDURES.append((_pname, _pts, _color))
# 只标注但不连线的点
for _n in ["HSH"]:
    if _n not in _seen_wp:
        EXTRA_WAYPOINTS.append((_n, *_WP[_n]))


# ============================================================
# 高度插值
# ============================================================

def interpolate_heights(procedure):
    """对高度为 None 的航路点做线性插值（基于相邻已知高度点间的 grid 距离）。"""
    pts_grid = [latlon_to_grid(lat, lon) for _, lat, lon, _ in procedure]
    heights = [h for _, _, _, h in procedure]

    for i in range(len(heights)):
        if heights[i] is not None:
            continue
        # 向前找已知高度
        j = i - 1
        while j >= 0 and heights[j] is None:
            j -= 1
        # 向后找已知高度
        k = i + 1
        while k < len(heights) and heights[k] is None:
            k += 1
        if j >= 0 and k < len(heights):
            d_total = calculate_distance(pts_grid[j], pts_grid[k])
            d_to_j = calculate_distance(pts_grid[j], pts_grid[i])
            frac = d_to_j / max(d_total, 0.01)
            heights[i] = heights[j] + frac * (heights[k] - heights[j])
        elif j >= 0:
            heights[i] = heights[j]
        elif k < len(heights):
            heights[i] = heights[k]

    return [(name, lat, lon, h) for (name, lat, lon, _), h in zip(procedure, heights)]


# ============================================================
# 路径生成
# ============================================================

def build_procedure_paths(procedure_with_heights):
    """将一条程序的航路点转为 (x, y, z) 三段式路径段列表 + 完整路径点。

    返回:
        segments: [(x1, y1, h1, x2, y2, h2), ...]  供噪声评估
        path_points: [(x, y, z), ...]              供可视化
        waypoint_labels: [(x, y, name), ...]        航路点标注
    """
    pts = [latlon_to_grid(lat, lon) for _, lat, lon, _ in procedure_with_heights]
    heights = [h for _, _, _, h in procedure_with_heights]
    names = [name for name, _, _, _ in procedure_with_heights]

    segments = []
    path_points = []
    waypoint_labels = []

    for i in range(len(pts)):
        x, y = pts[i]
        z = heights[i]
        path_points.append((x, y, z))
        waypoint_labels.append((x, y, names[i]))
        if i > 0:
            px, py, pz = path_points[i - 1]
            segments.append((px, py, pz, x, y, z))

    return segments, path_points, waypoint_labels


# ============================================================
# 评估与可视化
# ============================================================

def run():
    print("=" * 70)
    print("viz_procedures.py — 上海浦东现有进场程序可视化与噪声评估")
    print("=" * 70)

    # ── 加载人口 ──
    try:
        pop, N_pop = load_population_grid('population_data_square.xlsx')
        print(f"\n人口: shape={pop.shape} N={N_pop:,.0f}")
        nm = SELNoiseModel(pop)
        print(f"  alpha_atm={nm._alpha_atm*1000:.3f} dB/km "
              f"r_eff_table[500m]={nm._r_eff_table[5]:.1f}km "
              f"r_eff_table[4000m]={nm._r_eff_table[40]:.1f}km")
    except Exception as e:
        print(f"\n无人口数据: {e}")
        return

    print(f"\nPUD: ({P[0]:.0f},{P[1]:.0f})  "
          f"FAF: ({FAF[0]:.0f},{FAF[1]:.0f})  "
          f"FAF 在 PUD→IF 连线上")
    for name, proc, _ in ALL_PROCEDURES:
        pts = [latlon_to_grid(lat, lon) for _, lat, lon, _ in proc]
        print(f"  {name}: {' → '.join(n for n,_,_,_ in proc)}  "
              f"{len(pts)} 个航路点")

    # ── 评估每条程序 ──
    # 每条程序用完整航线（含下游共享段+IF→PUD）评估噪声，与 test.py 的5条独立航线语义一致。
    _union_pairs = set()   # 收集并集所需的唯一航段（waypoint pair，供去重）
    proc_results = []

    for proc_name, procedure, color in ALL_PROCEDURES:
        if len(procedure) == 0:
            continue
        print("\n" + "─" * 60)
        print("  %s" % proc_name)
        print("─" * 60)

        proc_h = interpolate_heights(procedure)
        _, path_pts, wp_labels = build_procedure_paths(proc_h)

        # 航线总长度（完整航线含共享段+进近，球面距离，共享段按出现次数重复计入）
        total_len = _full_route_length_km(proc_name)

        # 噪声评估：完整航线（含下游共享段+IF→PUD）
        full_segments = _build_full_route_segments(proc_name)
        n_exposed, _, _ = nm.exposed_population(full_segments)

        # 收集唯一航段（waypoint 名称对去重），用于并集噪声评估
        full_names = _FULL_ROUTE_NAMES.get(proc_name, [])
        for j in range(len(full_names) - 1):
            _union_pairs.add((full_names[j], full_names[j + 1]))

        proc_results.append({
            'name': proc_name, 'color': color,
            'segments': full_segments, 'path_pts': path_pts,
            'wp_labels': wp_labels,
            'total_len': total_len, 'n_exposed': n_exposed,
        })

        print("    航线长度: %.1f km" % total_len)
        print("    N_>65dB_SEL: {:,}".format(int(n_exposed)))
        for name, lat, lon, h in proc_h:
            gx, gy = latlon_to_grid(lat, lon)
            print("      %-10s (%.0f,%.0f)  %.0fm" % (name, gx, gy, h))

    # ── 并集噪声（所有唯一物理航段各出现一次）──
    # ── 并集噪声（所有唯一物理航段各出现一次）──
    # 先对全部出现的航路点做高度插值，再按去重 pair 逐段生成 segments。
    _all_wp_names = set()
    for _names in _FULL_ROUTE_NAMES.values():
        _all_wp_names.update(_names)
    _all_wp_list = [(_n, _WP[_n][0], _WP[_n][1], _WP[_n][2]) for _n in sorted(_all_wp_names)]
    _all_wp_h = interpolate_heights(_all_wp_list)
    _wp_h_map = {_name: _h for (_name, _, _, _h) in _all_wp_h}

    union_segments = []
    for _n1, _n2 in sorted(_union_pairs):
        _x1, _y1 = latlon_to_grid(_WP[_n1][0], _WP[_n1][1])
        _x2, _y2 = latlon_to_grid(_WP[_n2][0], _WP[_n2][1])
        _h1 = _wp_h_map[_n1]
        _h2 = _wp_h_map[_n2]
        union_segments.append((_x1, _y1, _h1, _x2, _y2, _h2))
    if len(union_segments) > 0:
        n_union, _, mask_union = nm.exposed_population(union_segments)
    else:
        n_union, mask_union = 0.0, None
    total_all = sum(r['total_len'] for r in proc_results)
    print("\n" + "=" * 60)
    print("  合计噪声（并集掩膜）")
    print("    总航线长度: %.1f km" % total_all)
    print("    N_>65dB_SEL: {:,}".format(int(n_union)))

    # ── 2D 可视化 ──
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

    fig2, ax2 = plt.subplots(figsize=(14, 14))
    ax2.set_xlim(0, GRID_SIZE_X)
    ax2.set_ylim(0, GRID_SIZE_Y)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('Shanghai Pudong Arrival Procedures [SEL Noise Map]', fontweight='bold')

    # 人口底图
    if pop is not None:
        ny, nx = pop.shape
        im = ax2.imshow(pop + 1, origin='lower', cmap='YlOrBr',
                        norm=LogNorm(vmin=1, vmax=max(pop.max(), 2)),
                        extent=[0, nx, 0, ny], alpha=0.75)
        cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Population (log scale)', fontsize=11)

    # 画每条程序
    for idx, r in enumerate(proc_results):
        arr = np.array(r['path_pts'])
        ax2.plot(arr[:, 0], arr[:, 1], color='black', linestyle='--',
                 linewidth=1.5, alpha=0.7,)
        # 航路点标记
        for x, y, name in r['wp_labels']:
            if name in ('FAF', 'Airport'):
                continue
            ax2.scatter(x, y, c=colors[idx % len(colors)], s=40,
                        marker='o', edgecolor='black', linewidth=0.5, zorder=5)

    # FAF 和 PUD 特殊标记
    ax2.scatter(FAF[0], FAF[1], c='gold', s=150, marker='D',
                edgecolor='black', linewidth=2, zorder=10)
    ax2.text(FAF[0] + 4, FAF[1] + 3, 'FAF', fontsize=12, fontweight='bold', color='gold')
    ax2.scatter(P[0], P[1], c='red', s=200, marker='s',
                edgecolor='black', linewidth=2, zorder=10)
    ax2.text(P[0] + 4, P[1] + 3, 'PUD', fontsize=12, fontweight='bold', color='red')

    # 独立参考点（不在进场程序中的点）
    for name, lat, lon, _ in EXTRA_WAYPOINTS:
        ex, ey = latlon_to_grid(lat, lon)
        ax2.scatter(ex, ey, c='grey', s=50, marker='s',
                    edgecolor='black', linewidth=1, zorder=6)

    # FAF→PUD 延长线（虚线）
    _ext_faf = np.array([FAF[0] + 3 * (FAF[0] - P[0]),
                         FAF[1] + 3 * (FAF[1] - P[1])])
    ax2.plot([P[0], _ext_faf[0]], [P[1], _ext_faf[1]],
             'grey', linestyle='--', linewidth=1, alpha=0.4)

    plt.tight_layout()
    plt.savefig('viz_procedures_2d.png', dpi=300, bbox_inches='tight')
    print("\n2D 图已保存: viz_procedures_2d.png")

    # ── 3D 可视化 ──
    fig3, ax3 = plt.subplots(figsize=(16, 12), subplot_kw={'projection': '3d'})
    ax3.set_xlim(0, GRID_SIZE_X)
    ax3.set_ylim(0, GRID_SIZE_Y)
    ax3.set_zlim(0, 6500)
    ax3.set_box_aspect([1, 1, 0.35])
    ax3.set_xlabel('X (km)')
    ax3.set_ylabel('Y (km)')
    ax3.set_zlabel('Height (m)')
    ax3.set_title('Shanghai Pudong Arrival Procedures [3D View]', fontweight='bold')

    for idx, r in enumerate(proc_results):
        arr = np.array(r['path_pts'])
        ax3.plot(arr[:, 0], arr[:, 1], arr[:, 2],
                 color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        for x, y, name in r['wp_labels']:
            z = 0
            for px, py, pz in r['path_pts']:
                if abs(px - x) < 0.5 and abs(py - y) < 0.5:
                    z = pz; break
            if name in ('FAF', 'Airport'):
                continue
            ax3.scatter(x, y, z, c=colors[idx % len(colors)], s=40,
                        marker='o', edgecolor='black', linewidth=0.5)

    # FAF, PUD
    ax3.scatter(FAF[0], FAF[1], FAF_HEIGHT, c='gold', s=150, marker='D',
                edgecolor='black', linewidth=2, zorder=10)
    ax3.text(FAF[0] + 4, FAF[1] + 3, FAF_HEIGHT + 200, 'FAF',
             fontsize=12, fontweight='bold', color='gold')
    ax3.scatter(P[0], P[1], 0, c='red', s=200, marker='s',
                edgecolor='black', linewidth=2, zorder=10)
    ax3.text(P[0] + 4, P[1] + 3, 200, 'PUD',
             fontsize=12, fontweight='bold', color='red')

    ax3.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig('viz_procedures_3d.png', dpi=300, bbox_inches='tight')
    print("3D 图已保存: viz_procedures_3d.png")

    # ── 汇总 ──
    print(f"\n{'=' * 70}")
    print(f"汇总")
    print(f"{'=' * 70}")
    print(f"  {'程序':<18} {'航线长度':>10} {'N_>65dB_SEL':>14} {'N_>65/N_pop':>12}")
    print(f"  {'─' * 54}")
    for r in proc_results:
        jn = r['n_exposed'] / nm.n_pop if nm.n_pop > 0 else 0
        print(f"  {r['name']:<18} {r['total_len']:>8.1f} km  "
              f"{r['n_exposed']:>14,.0f}  {jn:>11.4f}")
    jn_union = n_union / nm.n_pop if nm.n_pop > 0 else 0
    print(f"  {'─' * 54}")
    print(f"  {'合计（并集）':<18} {total_all:>8.1f} km  "
          f"{n_union:>14,.0f}  {jn_union:>11.4f}")
    print(f"{'=' * 70}")
    print("\nDone.")


if __name__ == "__main__":
    run()

# -*- coding: utf-8 -*-
"""导出论文引用到的两张参考表（派生数据，不含任何 AIP 原文）。

运行（在仓库根目录）：
    python export_reference_tables.py

输出：
    data/entry_points_if_faf.csv              进场点 / IF / FAF / 跑道入口
    data/waypoints_published_procedures.csv   已公布程序的航路点坐标与高度
    data/procedure_chains.csv                 各程序的航路点顺序链

坐标说明：lat/lon 为 WGS-84 十进制度；grid_x/grid_y 为模型笛卡尔坐标（km），
由 test.latlon_to_grid 换算（plate carrée，120 km/°，原点见 PROVENANCE.md）。
"""

import csv
import os

import test as t
import viz_procedures as v

OUT_DIR = "data"


def ll_of(wp_name):
    """航路点经纬度（十进制度）。"""
    lat, lon, _ = v._WP[wp_name]
    return float(lat), float(lon)


def grid_of(lat, lon):
    g = t.latlon_to_grid(lat, lon)
    return float(g[0]), float(g[1])


# E 编号 -> 进场点名称。test.py 在导入时按编号规则（相对 P->F 的顺时针角）
# 生成 numbered_points，即 E1..E5 的来源；已用坐标反查核对：
# E1=MATNU, E2=DUMET, E3=LISHE, E4=ANDONG, E5=SASAN。
E_ORDER = [
    (1, "MATNU"),
    (2, "DUMET"),
    (3, "LISHE"),
    (4, "ANDONG"),
    (5, "SASAN"),
]

# 注意：ENTRY_ALTS 与“原始顺序”对应，而不是与 E 编号对应。
_RAW_ORDER = ["SASAN", "ANDONG", "LISHE", "MATNU", "DUMET"]
_ALT_OF = dict(zip(_RAW_ORDER, t.ENTRY_ALTS))


def write_entry_points():
    path = os.path.join(OUT_DIR, "entry_points_if_faf.csv")
    rows = []
    for idx, name in E_ORDER:
        lat, lon = ll_of(name)
        gx, gy = grid_of(lat, lon)
        rows.append(dict(
            role="entry_point (E%d)" % idx, name=name,
            latitude_deg="%.6f" % lat, longitude_deg="%.6f" % lon,
            altitude_m=_ALT_OF[name],
            grid_x_km="%.3f" % gx, grid_y_km="%.3f" % gy))
    # 跑道入口 P、最终进近定位点 FAF、中间定位点 IF
    for role, name, pt in (("runway_threshold", "PUD/P", t.P),
                           ("final_approach_fix", "FAF", t.FAF),
                           ("intermediate_fix", "IF", t.IF_PT)):
        rows.append(dict(role=role, name=name, latitude_deg="", longitude_deg="",
                         altitude_m="", grid_x_km="%.3f" % float(pt[0]),
                         grid_y_km="%.3f" % float(pt[1])))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("->", path, "(%d rows)" % len(rows))


def write_waypoints():
    used = {}
    for proc, chain in v._FULL_ROUTE_NAMES.items():
        for wp in chain:
            used.setdefault(wp, set()).add(proc)

    path = os.path.join(OUT_DIR, "waypoints_published_procedures.csv")
    rows = []
    for name in v._WP:
        lat, lon, alt = v._WP[name]
        if lat is None:
            continue
        in_procs = sorted(used.get(name, []))
        rows.append(dict(
            waypoint=name,
            latitude_deg="%.6f" % float(lat), longitude_deg="%.6f" % float(lon),
            altitude_m="" if alt is None else "%d" % int(alt),
            used_by_procedures="; ".join(in_procs)))
    rows.sort(key=lambda r: (not r["used_by_procedures"], r["waypoint"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("->", path, "(%d rows)" % len(rows))


def write_chains():
    path = os.path.join(OUT_DIR, "procedure_chains.csv")
    rows = []
    for proc in sorted(v._FULL_ROUTE_NAMES):
        for order, wp in enumerate(v._FULL_ROUTE_NAMES[proc], 1):
            rows.append(dict(procedure=proc, stop_order=order, waypoint=wp))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("->", path, "(%d rows)" % len(rows))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    write_entry_points()
    write_waypoints()
    write_chains()

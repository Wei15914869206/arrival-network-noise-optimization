# -*- coding: utf-8 -*-
"""
plot_population.py — 独立生成浦东区域人口密度热力图

数据来源：population_data_square.xlsx（与 test.py 共用）
输出：population_heatmap.png（300 dpi）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import map_base

# ============================================================
# 人口数据加载（与 test.py load_population_grid 一致）
# ============================================================

POP_LON_WEST = 120.320834
POP_LAT_SOUTH = 29.520833
GRID_STEP_DEG = 1.0 / 120.0


def dms(d, m, s=0.0):
    return d + m / 60.0 + s / 3600.0


def load_population_grid(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    data = []
    for r in rows[1:]:
        vals = [float(v) if v is not None else 0.0 for v in r[1:]]
        data.append(vals)
    pop_north_to_south = np.array(data, dtype=float)
    pop = np.flipud(pop_north_to_south)
    return pop, float(pop.sum())


def latlon_to_grid(lat, lon):
    x = (lon - POP_LON_WEST) / GRID_STEP_DEG
    y = (lat - POP_LAT_SOUTH) / GRID_STEP_DEG
    return (x, y)


# ============================================================
# 关键地标坐标
# ============================================================

_airport_ll = (dms(31, 10, 18), dms(121, 47, 0))   # PUD
PUD = np.array(latlon_to_grid(*_airport_ll))

# FAF 在大圆路径上 PUD→XSY，距 PUD 11.8 NM（与 test.py 一致）
import math as _m
_IF_LL = (dms(30, 55, 54), dms(121, 52, 24))


def _great_circle_destination_to_pt(lat_a, lon_a, lat_b, lon_b, dist_km):
    R = 6371.0
    p1, p2 = _m.radians(lat_a), _m.radians(lat_b)
    dl = _m.radians(lon_b - lon_a)
    y = _m.sin(dl) * _m.cos(p2)
    x = _m.cos(p1) * _m.sin(p2) - _m.sin(p1) * _m.cos(p2) * _m.cos(dl)
    bearing = _m.degrees(_m.atan2(y, x))
    lat1 = _m.radians(lat_a); lon1 = _m.radians(lon_a)
    brng = _m.radians(bearing); dR = dist_km / R
    lat2 = _m.asin(_m.sin(lat1) * _m.cos(dR) + _m.cos(lat1) * _m.sin(dR) * _m.cos(brng))
    lon2 = lon1 + _m.atan2(_m.sin(brng) * _m.sin(dR) * _m.cos(lat1),
                             _m.cos(dR) - _m.sin(lat1) * _m.sin(lat2))
    return _m.degrees(lat2), _m.degrees(lon2)


_FAF_LL = _great_circle_destination_to_pt(*_airport_ll, *_IF_LL, 11.8 * 1.852)
FAF = np.array(latlon_to_grid(*_FAF_LL))

# 进场点
_entry_list = [
    ("SASAN",  dms(31, 35, 22), dms(120, 19, 10)),
    ("ANDONG", dms(30, 15, 24), dms(121, 13, 18)),
    ("LISHE",  dms(29, 53, 42), dms(121, 20,  0)),
    ("MATNU",  dms(31, 39, 36), dms(122, 38,  0)),
    ("DUMET",  dms(31, 21, 28), dms(122, 46, 30)),
]

# ============================================================
# 加载数据
# ============================================================

pop, N_pop = load_population_grid("population_data_square.xlsx")
ny, nx = pop.shape
print(f"人口栅格: {nx}×{ny}, 总人口: {N_pop:,.0f}")

# ============================================================
# 绘图
# ============================================================

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_aspect('equal')
ax.tick_params(labelsize=11)

# 人口热力图（零值格显白）
pop_masked = np.where(pop > 0, pop, np.nan)
im = ax.imshow(pop_masked + 1, origin='lower', cmap='YlOrBr',
               norm=LogNorm(vmin=2, vmax=max(pop.max(), 2)),
               extent=[0, nx, 0, ny], alpha=0.92)

cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.05, shrink=0.90)
cbar.set_label("Population density (persons/km²)", fontsize=18)
cbar.ax.tick_params(labelsize=15)

# 进场点
for ename, elat, elon in _entry_list:
    x, y = latlon_to_grid(elat, elon)
    ax.scatter(x, y, c='#2E7D32', s=200, marker='^', edgecolor='white',
               linewidth=1.2, zorder=10)
    ax.text(x, y - 10, ename, fontsize=16, fontweight='bold', color='#1B5E20',
            ha='center', va='top')

# 经纬度刻度（与轨迹图共用同一底图约定）
xticks_km = [0, 50, 100, 150, 200, 250]
yticks_km = [0, 50, 100, 150, 200, 250]
map_base.apply_latlon_ticks(ax, xticks_km, yticks_km, fontsize=16)

map_base.add_scale_bar(ax, 20)
map_base.add_north_arrow(ax)

fig.tight_layout()
fig.savefig("population_heatmap.png", dpi=300, bbox_inches='tight')
print("保存: population_heatmap.png")
fig.savefig("Figure3.pdf", bbox_inches='tight')
print("保存: Figure3.pdf")
plt.show()

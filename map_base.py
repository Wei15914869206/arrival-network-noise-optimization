# -*- coding: utf-8 -*-
"""
map_base.py — 地图底图共用绘制：经纬度刻度 + 比例尺 + 指北针。

test.py（轨迹图）与 plot_population.py（热力图）共用，保证两张图的
坐标系统与地图要素一致。网格 1 格 = 1 km。

原点/步长与 test.py、plot_population.py 中的定义一致（冻结的地理常量）。
"""

POP_LON_WEST = 120.320834
POP_LAT_SOUTH = 29.520833
GRID_STEP_DEG = 1.0 / 120.0


def grid_to_lon(x):
    return POP_LON_WEST + x * GRID_STEP_DEG


def grid_to_lat(y):
    return POP_LAT_SOUTH + y * GRID_STEP_DEG


def apply_latlon_ticks(ax, x_ticks, y_ticks, fontsize=16):
    """把网格坐标刻度替换为经纬度刻度（°E / °N）。"""
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{grid_to_lon(v):.1f}°E" for v in x_ticks],
                       fontsize=fontsize)
    ax.set_yticklabels([f"{grid_to_lat(v):.1f}°N" for v in y_ticks],
                       fontsize=fontsize)


def add_scale_bar(ax, length_km=20, fontsize=14, color='black'):
    """在右下角画一个 length_km 的直线比例尺（网格 1 格 = 1 km）。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0
    x_end = x1 - 0.06 * dx
    x_start = x_end - length_km
    y = y0 + 0.055 * dy
    ax.plot([x_start, x_end], [y, y], color=color, lw=2.5,
            solid_capstyle='butt', zorder=30)
    for xx in (x_start, x_end):
        ax.plot([xx, xx], [y - 0.012 * dy, y + 0.012 * dy],
                color=color, lw=2.0, zorder=30)
    ax.text((x_start + x_end) / 2, y + 0.025 * dy, f"{length_km} km",
            ha='center', va='bottom', fontsize=fontsize, color=color, zorder=30)


def add_north_arrow(ax, x=0.85, y=0.88, height=0.08, fontsize=16, color='black'):
    """在右上区域画一个朝上的指北针（数据北 = +y）。"""
    ax.annotate('', xy=(x, y + height), xytext=(x, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2.2),
                xycoords='axes fraction', textcoords='axes fraction', zorder=30)
    ax.text(x, y + height + 0.005, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold',
            color=color, zorder=30)

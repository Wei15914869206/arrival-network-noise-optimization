# -*- coding: utf-8 -*-
"""
test.py — 浦东多进场航线降噪设计（自包含）

求解框架：单链 SA 邻域搜索，决策变量 (T,Θ)——T=汇聚拓扑（树旋转邻域），
  Θ=各汇聚点扇环参数 (u,v)。
路径解码：反推 2D A* 最小噪声路径——从 IF 向进场点方向逐段外推，
  垂直剖面为"先平飞、后全程 3° 连续下降"的最高剖面，
  段内转弯次数受 MAX_TURNS_PER_SEGMENT 硬约束。
目标 F = w_N·N/N₀ + (1−w_N)·L/L₀（加权和，最小化），
  输出 R_N, R_L（相对基线的改善率，正数=改善）。

A* 边代价（解码器代理，广义航程）:
  c = Δs·(w_L + w_N·ñ),  ñ = α(z)·(ρ_strip/ρ̄),
  α(z)=E(z)/E(z_ref), ρ_strip 为航向条带（半宽 r_eff）平均人口密度；
H=w_L×2D欧氏距离，绕行率硬约束 ρ≤RHO_ASTAR_MAX；
最终评估用精确 SEL 模型（大气吸收 + 横向衰减 + 并集掩膜暴露人口）。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import heapq
import copy
import time
import map_base

# ============================================================
# 全局配置参数
# ============================================================

MAX_TURN_ANGLE = 60
MIN_TURN_DISTANCE = 5
MAX_TURNS_PER_SEGMENT = 3   # 段内最大转弯次数（None=不限制）
FAF_NO_CROSS = False        # True: 关闭 FAF→P 禁越线约束（诊断用）
INIT_THETA_MIDPOINT = True  # 初始解从扇环中点(u=v=0.5)出发（标定后默认，消除前期不可行）
NS_MAX_ITER = 150            # SA 链迭代代数
TARGET_TOPOLOGY = None      # 固定汇聚拓扑（设为 None 则 SA 搜索拓扑）
SECTOR_R_MIN_FRAC = 0.5     # 扇环半径下界 = R_c × 该比例（越小汇聚点越可逼近 FAF）
GRID_SIZE_X = 296
GRID_SIZE_Y = 296
GRID_UNIT_METERS = 1000
FAF_HEIGHT = 500
IF_HEIGHT = 900           # IF (XSY) 进场汇聚点高度 [m]

# ============================================================
# 坐标投影
# ============================================================

POP_LON_WEST = 120.320834
POP_LAT_SOUTH = 29.520833
GRID_STEP_DEG = 1.0 / 120.0


def latlon_to_grid(lat, lon):
    x = (lon - POP_LON_WEST) / GRID_STEP_DEG
    y = (lat - POP_LAT_SOUTH) / GRID_STEP_DEG
    return (x, y)


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
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _grid_path_length_km(path):
    """折线路径的球面总长度 [km]。

    path: [(x,y,...), ...] 网格坐标（仅用前两维 x,y），
    每小段经 grid_to_latlon → haversine 累加。
    """
    total = 0.0
    for i in range(len(path) - 1):
        lat1, lon1 = grid_to_latlon(path[i][0], path[i][1])
        lat2, lon2 = grid_to_latlon(path[i + 1][0], path[i + 1][1])
        total += _haversine_km(lat1, lon1, lat2, lon2)
    return total


def dms(d, m, s=0.0):
    return d + m / 60.0 + s / 3600.0


# ============================================================
# 配置类
# ============================================================

class PathfinderConfig:
    """路径规划器配置（存储网格/高度参数）。"""
    def __init__(self, grid_size_x=GRID_SIZE_X, grid_size_y=GRID_SIZE_Y,
                 grid_unit_meters=GRID_UNIT_METERS, faf_height=FAF_HEIGHT):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.grid_unit_meters = grid_unit_meters
        self.faf_height = faf_height


# ============================================================
# 人口栅格加载
# ============================================================

def load_population_grid(path):
    """读取 population_data_square.xlsx，返回 (pop, N_pop)。

    返回 pop[iy, ix]：iy=0 为最南行（与网格 y=0 底部对齐），ix=0 为最西列。
    """
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


# ============================================================
# 噪声评估模型（层级2：SEL框架 + 大气吸收 + 横向衰减）
# 新增: r_eff预计算表, SAT人口积分图, min_pop_accum势场（供A*使用）
# ============================================================

class SELNoiseModel:
    """基于 SEL（声暴露级）的航空噪声评估模型。

    声传播：L_A(R,β) = L_A,ref - 20·log₁₀(R/D_ref) - α_atm·(R-D_ref) - Λ(β)
    SEL积分：SEL = 10·log₁₀( Δt · Σ 10^(L_A,i/10) )

    预计算表（供A*快速代理）：
    - r_eff[z_bucket]: 高度z处SEL≥65dB的等效地面半宽 [km]
    - SAT:             人口的积分图 [O(1)矩形求和]
    - min_pop_accum:   从每格到目标的最小累积人口（反向Dijkstra）
    """

    def __init__(self, pop,
                 L_A_ref=140.0, D_ref=1.0, f_ref=500.0,
                 T=20.0, RH=70.0, p_ratio=1.0,
                 v_approach=80.0, SEL_threshold=65.0,
                 grid_unit_meters=GRID_UNIT_METERS, z_bucket=100, z_max=6000):
        self.pop = pop
        self.n_pop = float(pop.sum()) if pop.sum() > 0 else 1.0
        self.ny, self.nx = pop.shape
        self.L_A_ref = L_A_ref
        self.D_ref = D_ref
        self.f_ref = f_ref
        self.T_C = T
        self.RH = RH
        self.p_ratio = p_ratio
        self.v_approach = v_approach
        self.SEL_threshold = SEL_threshold
        self.grid_unit_meters = grid_unit_meters
        self.z_bucket = z_bucket
        self.z_max = z_max
        self.n_buckets = z_max // z_bucket + 1
        self._alpha_atm = self._atmospheric_absorption_db_per_m(f_ref, T, RH, p_ratio)
        # 预计算表
        self._r_eff_table = self._build_r_eff_table()
        self._SAT = self._build_SAT()
        # min_pop_accum 在 A* 启动时按目标点动态构建（调用 build_min_pop_accum）

    # ── SAE ARP 866A 大气吸收 ──

    @staticmethod
    def _atmospheric_absorption_db_per_m(f, T_C=20.0, RH=70.0, p_ratio=1.0):
        T_K = T_C + 273.15
        T_ref = 293.15
        T_ratio = T_K / T_ref
        p_sat = 101.325 * 10 ** (-6.8346 * (273.16 / T_K) ** 1.261 + 4.6151)
        h_w = RH * p_sat / (101.325 * p_ratio)
        f_rO = p_ratio * (24.0 + 4.04e4 * h_w * (0.02 + h_w) / (0.391 + h_w))
        f_rN = p_ratio * (T_ratio ** (-0.5)) * (
            9.0 + 280.0 * h_w * math.exp(-4.17 * (T_ratio ** (-1.0/3.0) - 1.0)))
        alpha_classic = 1.84e-11 * (p_ratio ** -1.0) * (T_ratio ** 0.5) * f * f
        alpha_O2 = (0.01275 * math.exp(-2239.1 / T_K) * (T_ratio ** (-2.5))
                    * f * f / (f_rO + f * f / f_rO))
        alpha_N2 = (0.1068 * math.exp(-3352.0 / T_K) * (T_ratio ** (-2.5))
                    * f * f / (f_rN + f * f / f_rN))
        return 8.686 * (alpha_classic + alpha_O2 + alpha_N2)

    # ── ECAC Doc 29 横向衰减 ──

    @staticmethod
    def _lateral_attenuation_db(elevation_deg):
        beta = np.asarray(elevation_deg, dtype=float)
        atten = np.where(beta < 60.0,
                         3.96 - 0.066 * beta + 9.9 * np.exp(-0.13 * beta), 0.0)
        atten = np.maximum(atten, 0.0)
        return float(atten) if atten.ndim == 0 else atten

    # ── A 计权声级 ──

    def _L_A(self, slant_dist_m, elevation_deg):
        R = np.maximum(slant_dist_m, 0.1)
        return (self.L_A_ref - 20.0 * np.log10(R / self.D_ref)
                - self._alpha_atm * (R - self.D_ref)
                - self._lateral_attenuation_db(elevation_deg))

    # ── 单航段 SEL 计算 ──

    def SEL_segment_map(self, x1, y1, h1, x2, y2, h2, n_samples=30):
        """单航段 SEL 地图——各采样点高度线性插值，不再用整段平均高度。

        航段格式: (x1,y1,h1,x2,y2,h2)，坐标单位=网格(km)，高度单位=m。
        """
        seg_len_km = math.hypot(x2 - x1, y2 - y1)
        if seg_len_km < 0.01:
            return None, None
        seg_len_m = seg_len_km * self.grid_unit_meters
        # bbox 用 (h1+h2)/2 估算（+5km 余量覆盖高低端差异）
        h_avg = (h1 + h2) / 2.0
        max_range_km = self._max_horizontal_range_km(h_avg, seg_len_km)
        xmin = max(0, int(math.floor(min(x1, x2) - max_range_km)))
        xmax = min(self.nx - 1, int(math.ceil(max(x1, x2) + max_range_km)))
        ymin = max(0, int(math.floor(min(y1, y2) - max_range_km)))
        ymax = min(self.ny - 1, int(math.ceil(max(y1, y2) + max_range_km)))
        if xmin > xmax or ymin > ymax:
            return None, None
        t_arr = np.linspace(0.0, 1.0, n_samples)
        xs = x1 + t_arr * (x2 - x1)
        ys = y1 + t_arr * (y2 - y1)
        hs = h1 + t_arr * (h2 - h1)   # 线性插值高度
        gx_grid = np.arange(xmin, xmax + 1, dtype=float) * self.grid_unit_meters
        gy_grid = np.arange(ymin, ymax + 1, dtype=float) * self.grid_unit_meters
        gx, gy = np.meshgrid(gx_grid, gy_grid)
        dt = seg_len_m / (self.v_approach * n_samples)
        energy_sum = np.zeros_like(gx, dtype=float)
        for i in range(n_samples):
            xi_m = xs[i] * self.grid_unit_meters
            yi_m = ys[i] * self.grid_unit_meters
            hi_m = float(hs[i])
            dx, dy = gx - xi_m, gy - yi_m
            R = np.sqrt(dx * dx + dy * dy + hi_m * hi_m)
            elevation = np.degrees(np.arctan2(hi_m, np.sqrt(dx * dx + dy * dy)))
            L_A_i = self._L_A(R, elevation)
            energy_sum += 10.0 ** (L_A_i / 10.0)
        SEL_2d = 10.0 * np.log10(dt * energy_sum)
        return SEL_2d, (xmin, xmax, ymin, ymax)

    def _max_horizontal_range_km(self, h_m, seg_len_km):
        tau_eff = seg_len_km * self.grid_unit_meters / self.v_approach
        SEL_bonus = 10.0 * math.log10(max(tau_eff, 1.0))
        lo, hi = 0.0, 50.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            R = math.sqrt((mid * 1000.0) ** 2 + h_m * h_m) + 0.01
            L_A = (self.L_A_ref - 20.0 * math.log10(R / self.D_ref)
                   - self._alpha_atm * (R - self.D_ref))
            SEL = L_A + SEL_bonus
            if SEL > self.SEL_threshold:
                lo = mid
            else:
                hi = mid
        return lo + 5.0

    def _add_segment_energy(self, energy_grid, x1, y1, h1, x2, y2, h2, n_samples=30):
        """将单个航段的声能量累加到全局能量网格（各采样点声能求和，不转 dB）。

        航段格式: (x1,y1,h1,x2,y2,h2)。各采样点高度线性插值，不再用整段平均高度。
        SEL = 10·log₁₀( (1/t₀) · Σ_segments Σ_i Δt_i · 10^(L_A,ij/10) ), t₀=1s
        """
        seg_len_km = math.hypot(x2 - x1, y2 - y1)
        if seg_len_km < 0.01:
            return
        seg_len_m = seg_len_km * self.grid_unit_meters
        # bbox 用 (h1+h2)/2 估算（+5km 余量覆盖高低端差异）
        h_avg = (h1 + h2) / 2.0
        max_range_km = self._max_horizontal_range_km(h_avg, seg_len_km)
        xmin = max(0, int(math.floor(min(x1, x2) - max_range_km)))
        xmax = min(self.nx - 1, int(math.ceil(max(x1, x2) + max_range_km)))
        ymin = max(0, int(math.floor(min(y1, y2) - max_range_km)))
        ymax = min(self.ny - 1, int(math.ceil(max(y1, y2) + max_range_km)))
        if xmin > xmax or ymin > ymax:
            return
        t_arr = np.linspace(0.0, 1.0, n_samples)
        xs = x1 + t_arr * (x2 - x1)
        ys = y1 + t_arr * (y2 - y1)
        hs = h1 + t_arr * (h2 - h1)   # 线性插值高度
        gx_grid = np.arange(xmin, xmax + 1, dtype=float) * self.grid_unit_meters
        gy_grid = np.arange(ymin, ymax + 1, dtype=float) * self.grid_unit_meters
        gx, gy = np.meshgrid(gx_grid, gy_grid)
        dt = seg_len_m / (self.v_approach * n_samples)
        for i in range(n_samples):
            xi_m = xs[i] * self.grid_unit_meters
            yi_m = ys[i] * self.grid_unit_meters
            hi_m = float(hs[i])
            dx, dy = gx - xi_m, gy - yi_m
            R = np.sqrt(dx * dx + dy * dy + hi_m * hi_m)
            elevation = np.degrees(np.arctan2(hi_m, np.sqrt(dx * dx + dy * dy)))
            L_A_i = self._L_A(R, elevation)
            energy_grid[ymin:ymax + 1, xmin:xmax + 1] += dt * (10.0 ** (L_A_i / 10.0))

    def exposed_population(self, segments, n_samples=30):
        """对所有航段计算 SEL 暴露人口——先全局能量累积，再统一转 SEL。

        修正：同一架飞机完整航迹的所有采样点声能量先求和，再转为 SEL，
        而非各航段独立算 SEL 后逐格取 max。后者在段交界处系统性地低估 1-3 dB。
        SEL_j = 10·log₁₀( (1/t₀) · Σ_segments Σ_i Δt_i · 10^(L_A,ij/10) ), t₀=1s
        """
        energy = np.zeros((self.ny, self.nx), dtype=float)
        for x1, y1, h1, x2, y2, h2 in segments:
            self._add_segment_energy(energy, x1, y1, h1, x2, y2, h2, n_samples)
        SEL_global = np.full((self.ny, self.nx), -np.inf, dtype=float)
        nonzero = energy > 0
        SEL_global[nonzero] = 10.0 * np.log10(energy[nonzero])
        mask = SEL_global >= self.SEL_threshold
        n_exposed = float(self.pop[mask].sum())
        return n_exposed, SEL_global, mask

    # ══════════════════════════════════════════════
    # A* 代理预计算表
    # ══════════════════════════════════════════════

    def _bucket_idx(self, z_m):
        return max(0, min(self.n_buckets - 1,
                         int(round(z_m / self.z_bucket))))

    def _bucket_z(self, b):
        return b * self.z_bucket + self.z_bucket / 2.0

    def _build_r_eff_table(self):
        """对每个高度桶二分求解 SEL≥65dB 的等效地面半宽 [km]。

        简化为: 假定飞机正下方（β=90°, Λ=0）单点飞越，
        等效暴露时间用 2*r/v 近似。
        物理上: 栅格距离越远→斜距越大→L_A越小→SEL越低。
        """
        table = np.zeros(self.n_buckets)
        for b in range(self.n_buckets):
            h_m = self._bucket_z(b)
            lo, hi = 0.0, 50.0  # km
            for _ in range(30):
                mid = (lo + hi) / 2.0
                r_m = mid * 1000.0
                R = math.sqrt(r_m * r_m + h_m * h_m) + 0.01
                L_A = self._L_A(R, 90.0)  # 正下方, 无横向衰减
                tau = 2.0 * r_m / self.v_approach
                SEL = L_A + 10.0 * math.log10(max(tau, 0.01))
                if SEL > self.SEL_threshold:
                    lo = mid
                else:
                    hi = mid
            table[b] = lo
        return table

    def r_eff_for_z(self, z_m):
        """查询高度z处的等效地面半宽 [km]。"""
        b = self._bucket_idx(z_m)
        return float(self._r_eff_table[b])

    def energy_factor_for_z(self, z_m):
        """正下方声能因子 10^((L_A(z)-SEL_threshold)/10)（不含 r_eff）。

        仅表征高度方向的相对声学严重度；足迹半宽由 r_eff_for_z 单独提供，
        避免在 A* 代价中对半宽重复计权。
        """
        h_m = max(z_m, 1.0)
        L_A = self._L_A(h_m, 90.0)
        return 10.0 ** ((L_A - self.SEL_threshold) / 10.0)

    def atten_for_z(self, z_m):
        """兼容旧接口: r_eff(z) * energy_factor(z)。

        物理含义: 高度 z 处单位飞越的"等效影响半宽 × 声能因子"。
        A* 边权请优先用 energy_factor_for_z + r_eff_for_z 分离形式。
        """
        return self.r_eff_for_z(z_m) * self.energy_factor_for_z(z_m)

    def _build_SAT(self):
        """构建人口的积分图（Summed-Area Table）[ny+1, nx+1]。

        SAT[y+1, x+1] = Σ_{iy≤y, ix≤x} pop[iy, ix]
        区域查询 O(1)。
        """
        sat = np.zeros((self.ny + 1, self.nx + 1), dtype=float)
        sat[1:, 1:] = self.pop
        sat = np.cumsum(sat, axis=0)
        sat = np.cumsum(sat, axis=1)
        return sat

    def pop_in_rect(self, x1, x2, y1, y2):
        """矩形区域人口和 [x1,x2]×[y1,y2]，边界钳入网格。 O(1)。"""
        ix1 = max(0, min(self.nx - 1, int(x1)))
        ix2 = max(0, min(self.nx - 1, int(x2)))
        iy1 = max(0, min(self.ny - 1, int(y1)))
        iy2 = max(0, min(self.ny - 1, int(y2)))
        if ix1 > ix2:
            ix1, ix2 = ix2, ix1
        if iy1 > iy2:
            iy1, iy2 = iy2, iy1
        s = self._SAT
        return float(s[iy2 + 1, ix2 + 1] - s[iy1, ix2 + 1]
                     - s[iy2 + 1, ix1] + s[iy1, ix1])

    def pop_in_strip(self, mid_x, mid_y, length_km, r_eff_km, angle_deg):
        """以 (mid_x, mid_y) 为中心、长 length_km、宽 2×r_eff_km 的带状区域人口。

        angle_deg 为航向角，控制旋转方向。
        矩形近似带状区域（轴向对齐于航向）。
        """
        rad = math.radians(angle_deg)
        half_l = length_km / 2.0
        r = max(r_eff_km, 0.5)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        # 矩形四角（粗略），用轴对齐包围盒取人口
        corners_x = [half_l, half_l, -half_l, -half_l]
        corners_y = [r, -r, r, -r]
        rx = [mid_x + cx * cos_a - cy * sin_a for cx, cy in zip(corners_x, corners_y)]
        ry = [mid_y + cx * sin_a + cy * cos_a for cx, cy in zip(corners_x, corners_y)]
        return self.pop_in_rect(min(rx), max(rx), min(ry), max(ry))

    def build_min_pop_accum(self, goal_x, goal_y):
        """反向Dijkstra: 从目标点出发，边权=pop(x,y)，求每格到目标的最小累积人口。

        返回 (min_pop_accum, parents_dict)。
        min_pop_accum[ny, nx] 形状同 pop。
        """
        gx, gy = int(round(goal_x)), int(round(goal_y))
        gx = max(0, min(self.nx - 1, gx))
        gy = max(0, min(self.ny - 1, gy))
        # 8-邻接
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, -1), (1, -1), (-1, 1)]
        dist = np.full((self.ny, self.nx), np.inf, dtype=float)
        dist[gy, gx] = 0.0
        heap = [(0.0, gx, gy)]
        while heap:
            d, x, y = heapq.heappop(heap)
            if d > dist[y, x]:
                continue
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.nx and 0 <= ny < self.ny:
                    nd = d + float(self.pop[ny, nx])
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        heapq.heappush(heap, (nd, nx, ny))
        self._min_pop_accum = dist
        return dist

    def min_pop_accum_val(self, x, y):
        """查询 (x,y) 处到目标的最小累积人口。"""
        ix = max(0, min(self.nx - 1, int(round(x))))
        iy = max(0, min(self.ny - 1, int(round(y))))
        return float(self._min_pop_accum[iy, ix])


# ============================================================
# 2D A* 最小噪声路径规划器（反推规划：机场侧 → 进场点侧）
#
# 角色：外层 SA 用精确 SEL 评价整树航迹；A* 仅作单段航迹解码器，
#       在运行约束下最小化与 SEL 同源的可加代理（广义航程）。
#
# 垂直剖面硬编码（规划方向 = 反推方向，起点低、终点高）：
#   z(x,y) = start_z + d_start × tan(3°)   以最大 3° "爬升"（航线尽可能高）
#          = goal_z                        达到进场点高度后平飞
# 对应实际飞行方向：先平飞，后全程 3° 连续下降（最高剖面）。
# A* 仅搜索 (x,y)，48 方向邻居，转弯约束。
#
# 边代价（广义航程，量纲 km）：
#   c = Δs · (w_L + w_N · ñ)
#   ñ = α(z) · (ρ_strip / ρ̄)
#   α(z) = E(z)/E(z_ref)   正下方声能因子相对参考高度归一化
#   ρ_strip = P_strip / (Δs · 2 r_eff(z))   航向条带内平均人口密度
#   r_eff(z) 只决定条带几何（SEL≥65 dB 等效半宽），不与 α 再乘半宽
#   w_N = 参考强度 ñ=1 时“绕行 1 km 与噪声项”的交换率
#
# H 代价：w_L × 2D 欧氏距离（噪声项 ≥0，不破坏航程项可采纳下界）。
# 绕行率硬约束：g_L + H > RHO_ASTAR_MAX × 弦长 → 剪枝（运行可接受上界）。
# ============================================================

MAX_ASTAR_ITERS = 300000
ASTAR_W_L = 1.0
ASTAR_W_N = 0.0          # A* 噪声权重（ASTAR_WN_AUTO_SYNC=True 时自动跟随 WN）
# 消融实验开关
USE_ASTAR_DECODER = True       # False → plan_hybrid_paths 全部用直线段
CHECK_WAYPOINT_TURNS = True    # False → 跳过汇聚点转弯角度后验（M1 消融用）
ASTAR_WN_AUTO_SYNC = True  # True → evaluate_X 自动 ASTAR_W_N ← WN；False → 手动控制
ASTAR_Z_REF = 1000.0     # α(z) 归一化参考高度 [m]（中间进近量级，使 α(z_ref)=1）
RHO_ASTAR_MAX = 1.25     # 段内绕行率硬上界：路径长 ≤ ρ×弦长（民航进场绕行容忍量级）
_TAN3 = math.tan(math.radians(3.0))          # ≈ 0.0524
_DESCENT_RATE = _TAN3 * 1000.0               # ≈ 52.4 m 下降 / km 水平距离


class AStar2DNoise:
    """2D A*：固定垂直剖面 + 航向条带人口代理（广义航程）。"""

    def __init__(self, noise_model):
        self.nm = noise_model
        self.nx = noise_model.nx
        self.ny = noise_model.ny
        self._nbrs = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)
                      if dx != 0 or dy != 0]
        self._cache = {}
        # 全域平均人口密度 [人/km²]：相对密度 ρ_strip/ρ̄ 的分母
        self._pop_mean = noise_model.n_pop / max(noise_model.nx * noise_model.ny, 1)
        # α(z)=E(z)/E(z_ref)；z_ref 处 α=1，低空 >1、高空 <1
        e_ref = noise_model.energy_factor_for_z(ASTAR_Z_REF)
        self._energy_ref = e_ref if e_ref > 1e-30 else 1e-30

    # ── 固定垂直剖面 ──

    def _z_at(self, x, y):
        """当前位置 (x,y) 的固定高度 [m]（规划方向：起点低、终点高）。

        策略：从起点（FAF侧）以最大 3° "爬升"使航线尽可能高，
        到达终点（进场点侧）高度 goal_z 后平飞。
        对应实际飞行方向：先平飞，后 3° 连续下降。
        """
        d = math.hypot(x - self._sx, y - self._sy)          # 距起点 2D 距离 [km]
        z_3deg = self._sz + d * _DESCENT_RATE               # 3° 爬升线高度
        return min(self._gz, z_3deg)                        # 达到进场点高度后平飞

    # ── 航向工具 ──

    @staticmethod
    def get_angle(fr, to):
        return math.degrees(math.atan2(to[1] - fr[1], to[0] - fr[0]))

    @staticmethod
    def angle_diff(a1, a2):
        d = abs(a1 - a2) % 360
        return min(d, 360 - d)

    # ── G: 广义航程边代价 ──

    def _G_step(self, cx, cy, nx, ny, parent_xy):
        """单步边代价 c = Δs · (w_L + w_N · ñ)，ñ = α(z) · ρ_strip/ρ̄。

        - α(z)：正下方声能因子相对 z_ref 归一化（高度越低越贵）
        - ρ_strip：沿航向、宽 2·r_eff(z) 的条带平均人口密度
        - r_eff 只定条带几何，与精确评价共用 SEL≥65 dB 足迹族
        返回 (edge_cost, step_km)。
        """
        step_km = math.hypot(nx - cx, ny - cy)
        if step_km < 1e-6:
            return 0.0, 0.0

        mid_x, mid_y = (cx + nx) / 2.0, (cy + ny) / 2.0
        z_mid = self._z_at(mid_x, mid_y)
        r_eff = max(self.nm.r_eff_for_z(z_mid), 0.5)  # km；下限避免退化条带
        alpha = self.nm.energy_factor_for_z(z_mid) / self._energy_ref

        # 航向条带人口（长 Δs、半宽 r_eff）；内部以旋转矩形 AABB + SAT 查询
        hdg = self.get_angle((cx, cy), (nx, ny))
        pop_strip = self.nm.pop_in_strip(mid_x, mid_y, step_km, r_eff, hdg)
        area = max(step_km * (2.0 * r_eff), 0.01)  # km²
        rel_density = (pop_strip / area) / max(self._pop_mean, 1e-6)
        n_tilde = alpha * rel_density  # 无量纲相对噪声强度

        edge_cost = step_km * (ASTAR_W_L + ASTAR_W_N * n_tilde)
        return edge_cost, step_km

    # ── H: 航程项可采纳启发（忽略非负噪声项）──

    def _H(self, x, y):
        """h = d₂ → 对应 g 中 w_L·L 的下界；f = g + w_L·h。"""
        return math.hypot(x - self._gx, y - self._gy)

    # ── 邻域扩展（2D，含转弯约束 + FAF-P 禁越线约束）──

    def _expand(self, cx, cy, parent_xy, last_turn_xy):
        results = []
        for dx, dy in self._nbrs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < self.nx and 0 <= ny < self.ny):
                continue
            # FAF-P 禁越线：该侧子树的所有航段不得越过 FAF→P 延长线。
            # FAF 周边 5km 内豁免——规划从 FAF 出发时，initial_heading 引导
            # 的第一步可能短暂"跨线"，但会立即回到正确侧；下游航段不受影响。
            if self._side_allowed is not None:
                dist_to_faf = math.hypot(nx - self._side_ox, ny - self._side_oy)
                if dist_to_faf > 5.0:
                    cross = (self._side_vx * (ny - self._side_oy)
                             - self._side_vy * (nx - self._side_ox))
                    if self._side_allowed == 1 and cross < -1e-6:
                        continue
                    if self._side_allowed == -1 and cross > 1e-6:
                        continue
            if parent_xy is not None:
                ch = self.get_angle(parent_xy, (cx, cy))
                nh = self.get_angle((cx, cy), (nx, ny))
                ta = self.angle_diff(ch, nh)
                if ta > MAX_TURN_ANGLE:
                    continue
                if ta > 0.1 and last_turn_xy is not None:
                    d = math.hypot(cx - last_turn_xy[0], cy - last_turn_xy[1])
                    if d < MIN_TURN_DISTANCE:
                        continue
            results.append((nx, ny))
        return results

    # ── A* 主循环 ──

    def astar_2d(self, start, goal, initial_heading=None, goal_heading=None,
                 max_turns=None, faf_line_side=None):
        """2D A* 最小噪声路径搜索（反推方向：start=机场侧低，goal=进场点侧高）。

        代价 = 广义航程 Σ Δs·(w_L + w_N·ñ)，ñ=α(z)·ρ_strip/ρ̄；
        f = g + w_L·H；路径长受绕行率硬约束 ρ ≤ RHO_ASTAR_MAX。

        start/goal: (x, y, z_m) 三元组。
        initial_heading: 约束路径第一步的航向（离开起点的方向）[°]
        goal_heading:    约束路径最后一步的航向（进入终点的方向）[°]
        max_turns:       段内最大转弯次数硬约束（None=不限制）。
        faf_line_side:   (faf, airport, allowed_sign) 或 None。
                         FAF→P 延长线禁越约束：allowed_sign=+1 只允许线左侧、
                         −1 只允许线右侧。None 不限制。

        返回: path [(x,y,z), ...] 三元组或 []。
        """
        sx, sy = int(start[0]), int(start[1])
        self._sx, self._sy = sx, sy
        self._sz = float(start[2])
        gx, gy = int(goal[0]), int(goal[1])
        self._gx, self._gy, self._gz = gx, gy, float(goal[2])

        # FAF-P 禁越线状态（供 _expand 使用）
        if faf_line_side is not None:
            fo, fa, self._side_allowed = faf_line_side
            self._side_ox, self._side_oy = fo[0], fo[1]
            self._side_vx = fa[0] - fo[0]
            self._side_vy = fa[1] - fo[1]
        else:
            self._side_allowed = None

        # 缓存：含端点高度 + FAF 禁越侧 + A* 权重（不同权重产生不同路径）
        _side_tag = faf_line_side[2] if faf_line_side is not None else None
        ck = (sx, sy, gx, gy, int(round(self._sz)), int(round(self._gz)),
              initial_heading, goal_heading, max_turns, _side_tag,
              ASTAR_W_L, ASTAR_W_N)
        if ck in self._cache:
            return self._cache[ck]

        track_turns = max_turns is not None   # 不限制时 k 恒为 0，状态空间同纯 2D
        skey = (sx, sy, 0)

        # 绕行率硬约束的路径长预算：ρ×弦长（弦长设下限，避免极短段被剪死）
        chord = math.hypot(gx - sx, gy - sy)
        len_budget = RHO_ASTAR_MAX * max(chord, MIN_TURN_DISTANCE)

        open_set = []
        heapq.heappush(open_set, (ASTAR_W_L * self._H(sx, sy), sx, sy, 0))
        came_from = {}
        g = {skey: 0.0}       # 组合代价（广义航程）
        g_l = {skey: 0.0}     # 纯航程（供绕行率剪枝）
        closed = set()
        last_turn = {skey: None}

        vp = None  # virtual parent for heading
        if initial_heading is not None:
            rad = math.radians(initial_heading)
            vp = (sx - math.cos(rad), sy - math.sin(rad))
            came_from[skey] = vp

        iters = 0
        while open_set and iters < MAX_ASTAR_ITERS:
            iters += 1
            _, cx, cy, k = heapq.heappop(open_set)
            cst = (cx, cy, k)
            if cst in closed:
                continue
            closed.add(cst)

            # 目标检查（任意转弯数均可到达）
            if (cx, cy) == (gx, gy):
                # goal_heading 检查
                if goal_heading is not None:
                    parent = came_from.get(cst)
                    if parent is not None and parent != vp:
                        arr_h = self.get_angle(parent[:2], (cx, cy))
                        if self.angle_diff(arr_h, goal_heading) > MAX_TURN_ANGLE:
                            closed.discard(cst)
                            continue
                path_xy = [(cx, cy)]
                nk = cst
                while nk in came_from:
                    p = came_from[nk]
                    if vp is not None and p == vp:
                        break
                    if p not in g_l:
                        break
                    path_xy.append(p[:2])
                    nk = p
                path_xy = path_xy[::-1]
                # 赋予高度
                result = [(px, py, self._z_at(px, py)) for px, py in path_xy]
                # 修正起点和终点高度为精确值
                result[0] = (result[0][0], result[0][1], self._sz)
                result[-1] = (result[-1][0], result[-1][1], self._gz)
                self._cache[ck] = result
                return result

            parent_st = came_from.get(cst, None)
            parent_xy = parent_st[:2] if parent_st is not None else None
            lt = last_turn.get(cst, None)

            for nx, ny in self._expand(cx, cy, parent_xy, lt):
                # 本步是否转弯（与上一步航向变化 > 0.1°）
                is_turn = False
                if parent_xy is not None:
                    ch = self.get_angle(parent_xy, (cx, cy))
                    nh = self.get_angle((cx, cy), (nx, ny))
                    is_turn = self.angle_diff(ch, nh) > 0.1
                # 起点处相对 initial_heading 的转弯是节点转弯，不计入段内次数
                nk_turns = k
                if track_turns and is_turn and not (vp is not None and parent_xy == vp):
                    nk_turns = k + 1
                    if nk_turns > max_turns:
                        continue
                nkey = (nx, ny, nk_turns)
                if nkey in closed:
                    continue

                ec, step_km = self._G_step(cx, cy, nx, ny, parent_xy)
                tent_g_l = g_l[cst] + step_km
                # 绕行率剪枝：经该点的最短可能总长下界 > ρ×弦长 → 剪
                if tent_g_l + self._H(nx, ny) > len_budget:
                    continue
                tent_g = g[cst] + ec

                if nkey not in g or tent_g < g[nkey]:
                    came_from[nkey] = cst
                    g_l[nkey] = tent_g_l
                    g[nkey] = tent_g
                    f = tent_g + ASTAR_W_L * self._H(nx, ny)
                    last_turn[nkey] = ((cx, cy) if is_turn else lt)
                    heapq.heappush(open_set, (f, nx, ny, nk_turns))

        self._cache[ck] = []
        return []


# ============================================================
# 二叉树结构
# ============================================================

class TreeNode:
    def __init__(self, id, left=None, right=None, is_entry=False):
        self.id = id
        self.left = left
        self.right = right
        self.is_entry = is_entry
        self.depth = 0
        self.min_leaf = None
        self.x = 0.0
        self.y = 0.0
        self.u = 0.0
        self.v = 0.0

    def __repr__(self):
        return f"Node({self.id}, x={self.x:.2f}, y={self.y:.2f})"


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_sector_rings(A, B, center):
    """以 center 为极点的扇环（角区间 + 半径上下界），A, B 为两个子节点坐标。

    radius_min 由 SECTOR_R_MIN_FRAC 控制——越小汇聚点越可逼近 center。
    """
    FA = A - center; FB = B - center
    dist_FA = np.linalg.norm(FA); dist_FB = np.linalg.norm(FB)
    R_c = min(dist_FA, dist_FB)
    radius_min, radius_max = R_c * SECTOR_R_MIN_FRAC, R_c
    angle_A = np.arctan2(FA[1], FA[0]); angle_B = np.arctan2(FB[1], FB[0])
    angle_diff_ccw = angle_B - angle_A
    if angle_diff_ccw < 0: angle_diff_ccw += 2 * np.pi
    angle_start, angle_end = (angle_A, angle_B) if angle_diff_ccw <= np.pi else (angle_B, angle_A)
    if angle_end < angle_start: angle_end += 2 * np.pi
    return (angle_start, angle_end, radius_min, radius_max)


def decode_params(u, v, sector_params, center):
    angle_start, angle_end, radius_min, radius_max = sector_params
    angle_range = angle_end - angle_start
    actual_angle = angle_start + v * angle_range
    actual_radius = radius_min + u * (radius_max - radius_min)
    return (center[0] + actual_radius * np.cos(actual_angle),
            center[1] + actual_radius * np.sin(actual_angle))


def generate_binary_trees(n):
    def generate(n):
        if n == 1: return [[]]
        trees = []
        for i in range(1, n):
            for left in generate(i):
                for right in generate(n - i):
                    trees.append([left, right])
        return trees
    raw_trees = generate(n)
    leaf_ids = [1, 2, 3, 4, 5]; leaf_index = [0]
    def tree_to_node(tree):
        if len(tree) == 0:
            node = TreeNode(leaf_ids[leaf_index[0]], is_entry=True)
            leaf_index[0] += 1; return node
        return TreeNode(0, left=tree_to_node(tree[0]), right=tree_to_node(tree[1]), is_entry=False)
    all_trees = []
    for tree in raw_trees:
        leaf_index[0] = 0; all_trees.append(tree_to_node(tree))
    return all_trees


def _e1_e2_are_siblings(tree):
    """检查 E1(1) 和 E2(2) 是否为直接兄弟节点（第一层汇聚约束）。"""
    if tree.is_entry:
        return False
    left, right = tree.left, tree.right
    if left.is_entry and right.is_entry:
        return {left.id, right.id} == {1, 2}
    return _e1_e2_are_siblings(left) or _e1_e2_are_siblings(right)


CONSTRAIN_E1E2_FIRST = True   # True: E1/E2 必须在第一层汇聚（含约束版本，14 种）
                                # False: 取消 E1/E2 约束（42 种，加泰兰数 C₄=14 的全排列）

def generate_convergence_structures(constrain_e1e2=None):
    if constrain_e1e2 is None:
        constrain_e1e2 = CONSTRAIN_E1E2_FIRST
    trees = generate_binary_trees(5)
    # E1 只能和 E2 在第一层汇聚 → 过滤拓扑（可关闭）
    if constrain_e1e2:
        trees = [t for t in trees if _e1_e2_are_siblings(t)]
    for tree in trees:
        internal_nodes = []
        def collect_and_calc(node):
            if node is None: return
            collect_and_calc(node.left); collect_and_calc(node.right)
            if not node.is_entry:
                ll = node.left.depth if not node.left.is_entry else 0
                rl = node.right.depth if not node.right.is_entry else 0
                node.depth = 1 if (node.left.is_entry and node.right.is_entry) else max(ll, rl) + 1
                internal_nodes.append(node)
        collect_and_calc(tree)
        def calc_min(node):
            if node.is_entry: return node.id
            node.min_leaf = min(calc_min(node.left), calc_min(node.right)); return node.min_leaf
        calc_min(tree)
        internal_nodes.sort(key=lambda n: (n.depth, n.min_leaf))
        for i, node in enumerate(internal_nodes): node.id = 6 + i
    return trees


def tree_to_bracket(node):
    if node is None: return ""
    if node.is_entry: return str(node.id)
    return f"({tree_to_bracket(node.left)},{tree_to_bracket(node.right)})"


def decode_tree(tree, params, numbered_points, root_pt):
    """以 root_pt（IF）为汇聚根和扇环极点，解码 Θ → 各汇聚点坐标。"""
    entry_coords = {id: point for id, point in numbered_points}
    def _decode_node(node, pi):
        if node.is_entry:
            node.x, node.y = entry_coords[node.id]; return pi
        pi = _decode_node(node.left, pi); pi = _decode_node(node.right, pi)
        if node is tree:
            node.x, node.y = root_pt[0], root_pt[1]; node.u = node.v = 0.0; return pi
        node.u, node.v = params[pi], params[pi + 1]; pi += 2
        lp = np.array([node.left.x, node.left.y]); rp = np.array([node.right.x, node.right.y])
        sector = calculate_sector_rings(lp, rp, root_pt)
        node.x, node.y = decode_params(node.u, node.v, sector, root_pt)
        return pi
    _decode_node(tree, 0)
    return tree


# ============================================================
# 路径规划
# ============================================================

def plan_hybrid_paths(decoded_tree, pathfinder, FAF, P,
                      astar=None, astar_segment_id=None, verbose=False,
                      max_turns=None):
    """混合路径规划: 指定航段用 A* 噪声优化，其余用直线。

    规划顺序为反推：从机场/FAF 向进场点方向逐段外推（A* 从父节点搜向
    子节点，规划完反转回飞行方向），航向经 node_incoming_headings 传播。

    astar_segment_id:
        None  → 全部用直线
        int   → 仅 child_id==该值的航段用 A*
        True  → 所有非 FAF→机场的航段用 A*
    max_turns:
        None  → 段内转弯次数不限
        int   → 所有 A* 航段统一的段内最大转弯次数
        dict  → 按 child_id 规定，如 {5: 2, 3: 1}（未列出的不限）
    """
    if astar_segment_id is not None and astar is None:
        raise ValueError("astar_segment_id specified but astar is None")
    astar_all = (astar_segment_id is True)

    def _straight_path(sp, ep, zs, ze):
        return [(int(sp[0]), int(sp[1]), zs), (int(ep[0]), int(ep[1]), ze)]

    all_paths = []; is_valid = True

    segments = []
    def collect_segments(node):
        if node.is_entry: return
        for child in (node.left, node.right):
            start = (child.x, child.y); end = (node.x, node.y)
            if math.hypot(start[0] - end[0], start[1] - end[1]) > 0.5:
                segments.append({'start': start, 'end': end,
                                 'start_node': child, 'end_node': node,
                                 'child_id': child.id})
        collect_segments(node.left); collect_segments(node.right)
    collect_segments(decoded_tree)
    segments.append({'start': (FAF[0], FAF[1]), 'end': (P[0], P[1]),
                     'start_node': decoded_tree, 'end_node': None, 'child_id': None})

    faf_height = pathfinder.faf_height
    _tan_climb = math.tan(math.radians(3.0))

    _entry_name_to_alt = {}
    for i, pt in enumerate(entry_points):
        _entry_name_to_alt[(pt[0], pt[1])] = ENTRY_ALTS[i]

    def subtree_alt_cap(node):
        """节点高度上限 = 子树内进场点高度的最小值（到达该高度即平飞）。"""
        if node.is_entry:
            return _entry_name_to_alt.get((node.x, node.y), float('inf'))
        return min(subtree_alt_cap(node.left), subtree_alt_cap(node.right))

    node_heights = {}
    node_heights[(decoded_tree.x, decoded_tree.y)] = float(IF_HEIGHT)   # IF 汇聚点高度 900m
    node_heights[(P[0], P[1])] = 0.0

    def assign_heights(node, pz):
        nk = (node.x, node.y)
        if nk not in node_heights:
            if hasattr(node, "parent_node") and node.parent_node is not None:
                pp = node.parent_node
                dm = math.hypot(node.x - pp.x, node.y - pp.y) * pathfinder.grid_unit_meters
                # 反推方向以最大 3° 爬升（航线尽可能高），不超过子树进场点最低高度
                node_heights[nk] = min(pz + dm * _tan_climb, subtree_alt_cap(node))
            else:
                node_heights[nk] = pz
        nz = node_heights[nk]
        if not node.is_entry:
            node.left.parent_node = node; node.right.parent_node = node
            assign_heights(node.left, nz); assign_heights(node.right, nz)
    decoded_tree.parent_node = None
    assign_heights(decoded_tree, float(IF_HEIGHT))

    def override_entry_heights(node):
        if node.is_entry:
            key = (node.x, node.y); alt = _entry_name_to_alt.get(key)
            if alt is not None: node_heights[key] = float(alt)
        else:
            override_entry_heights(node.left); override_entry_heights(node.right)
    override_entry_heights(decoded_tree)

    # ── 反推规划（参考 integrated_pathfinder）：从机场/FAF 向进场点方向逐段外推 ──
    def set_depths(node):
        if node.is_entry:
            node.depth = 0; return 0
        dl = set_depths(node.left); dr = set_depths(node.right)
        node.depth = max(dl, dr) + 1; return node.depth
    set_depths(decoded_tree)

    # 先 FAF→P，再按父节点（end_node）深度降序：FAF 侧航段先规划、进场点侧最后
    segments.sort(key=lambda s: s['end_node'].depth if s['end_node'] is not None else 10**9,
                  reverse=True)

    # ── 禁越线：IF（汇聚根）的两个子树分居 IF→PUD 延长线两侧 ──
    def _faf_cross(px, py):
        # _faf_cross 计算点 (px,py) 在 FAF→PUD 线的哪一侧；
        # 因 FAF 在 IF→PUD 大圆上，IF→PUD 航向与 FAF→PUD 一致——判侧结果相同。
        return (P[0] - FAF[0]) * (py - FAF[1]) - (P[1] - FAF[1]) * (px - FAF[0])

    def _subtree_side(subtree_root):
        """子树内进场点在 IF→PUD 延长线的哪一侧（+1 左 / −1 右），用于约束 A* 禁越界。"""
        sides = []
        def _side_collect(n):
            if n.is_entry:
                sides.append(_faf_cross(n.x, n.y))
            else:
                _side_collect(n.left); _side_collect(n.right)
        _side_collect(subtree_root)
        pos = sum(1 for s in sides if s > 0)
        return 1 if pos >= len(sides) - pos else -1

    if astar is not None:
        left_side = _subtree_side(decoded_tree.left)
        right_side = _subtree_side(decoded_tree.right)
        # 某些拓扑下两子树在 IF→PUD 线同侧（3/2 进场点分布所致），
        # 此时禁越线约束无意义，设为 None 跳过
        if left_side == right_side:
            left_side = right_side = None

    def _seg_subtree_side(seg):
        """返回航段所属 FAF 子树的允许侧（+1 或 −1），非 A* 段返回 None。"""
        if FAF_NO_CROSS:
            return None
        if astar is None:
            return None
        sn = seg.get('start_node')
        if sn is None:
            return None

        def _in_subtree(node, root):
            if node is root:
                return True
            if root.is_entry:
                return False
            return _in_subtree(node, root.left) or _in_subtree(node, root.right)

        if _in_subtree(sn, decoded_tree.left):
            return left_side
        if _in_subtree(sn, decoded_tree.right):
            return right_side
        return None  # FAF itself or not found

    # ================================================================

    # 每个节点的"入射航向"（规划方向 = 反推方向，由更内侧航段传播而来）。
    # FAF 预置 P→FAF 航向：规划从 FAF 出发的第一步朝此方向，确保路径能在
    # FAF 处对齐最终进近航道（禁越线不约束 FAF→W 段，冲突已解除）。
    faf_key = (FAF[0], FAF[1])
    node_incoming_headings = {faf_key: AStar2DNoise.get_angle((P[0], P[1]), (FAF[0], FAF[1]))}

    for i, seg in enumerate(segments):
        sp, ep = seg['start'], seg['end']   # sp=子节点(进场点侧,高)  ep=父节点(FAF侧,低)
        z_start = node_heights.get(sp, 0.0); z_end = node_heights.get(ep, 0.0)
        chord_km = math.hypot(sp[0] - ep[0], sp[1] - ep[1])
        cid = seg.get('child_id'); sn = seg.get('start_node'); en = seg.get('end_node')
        sl = f"E{sn.id}" if sn and sn.is_entry else f"W{sn.id}" if sn else "IF"
        el = "AP" if en is None else "IF" if en is decoded_tree else f"W{en.id}"
        sp_key = (sp[0], sp[1])
        ep_key = (ep[0], ep[1]) if ep else None

        use_astar = (astar is not None and seg["end_node"] is not None and
                     (astar_all or
                      (astar_segment_id is not None and cid is not None
                       and cid == astar_segment_id)))

        # 反推规划的 initial_heading：进入父节点的规划航向（由更内侧航段传来）
        inc_h = node_incoming_headings.get(ep_key) if ep_key is not None else None

        # 段内最大转弯次数（int 统一；dict 按 child_id 规定）
        mt = max_turns.get(cid) if isinstance(max_turns, dict) else max_turns

        if use_astar:
            # 反推：A* 从父节点（低）搜向子节点（高），完成后反转为飞行方向
            astart = (int(ep[0]), int(ep[1]), z_end)
            aend = (int(sp[0]), int(sp[1]), z_start)
            # FAF-P 禁越线约束
            seg_side = _seg_subtree_side(seg)
            fls = (FAF, P, seg_side) if seg_side is not None else None
            path = astar.astar_2d(astart, aend, initial_heading=inc_h,
                                  max_turns=mt, faf_line_side=fls)
            tag = "A*"; method = "A*"
            if path:
                # 先记录规划方向到达子节点的航向（作为下一段的 initial_heading），再反转
                if len(path) >= 2:
                    node_incoming_headings[sp_key] = AStar2DNoise.get_angle(
                        path[-2][:2], path[-1][:2])
                path = path[::-1]   # 反转为实际飞行方向（子→父）
            else:
                path = _straight_path(sp, ep, z_start, z_end)
                tag = "A*FAIL"; method = "直线(回退)"; is_valid = False
                node_incoming_headings[sp_key] = AStar2DNoise.get_angle(ep, sp)
        else:
            path = _straight_path(sp, ep, z_start, z_end)
            tag = "直线"; method = "直线"
            if seg["end_node"] is not None:
                node_incoming_headings[sp_key] = AStar2DNoise.get_angle(ep, sp)

        seg_actual_len = sum(math.hypot(path[j + 1][0] - path[j][0],
                                        path[j + 1][1] - path[j][1])
                             for j in range(len(path) - 1)) if len(path) > 1 else 0.0

        if verbose:
            base = (f"  {tag:8s} [{i + 1}/{len(segments)}] "
                    f"({sl},{sp[0]:.0f},{sp[1]:.0f})→({el},{ep[0]:.0f},{ep[1]:.0f}) "
                    f"Δz={z_start - z_end:.0f}m chord={chord_km:.1f}km")
            if use_astar:
                # 段内实际转弯次数（相邻两步航向变化 > 0.1°）
                n_turns = sum(
                    1 for j in range(1, len(path) - 1)
                    if AStar2DNoise.angle_diff(
                        AStar2DNoise.get_angle(path[j - 1][:2], path[j][:2]),
                        AStar2DNoise.get_angle(path[j][:2], path[j + 1][:2])) > 0.1)
                base += f" len={seg_actual_len:.1f}km turns={n_turns}"
                if mt is not None:
                    base += f"(≤{mt})"
                if inc_h is not None and len(path) >= 2:
                    # 规划方向离开父节点的航向 vs 内侧航段传来的要求
                    dep_plan = AStar2DNoise.get_angle(path[-1][:2], path[-2][:2])
                    ta = AStar2DNoise.angle_diff(dep_plan, inc_h)
                    base += f" dep={dep_plan:.0f}°(req~{inc_h:.0f}°Δ{ta:.0f}°)"
            print(base)

        all_paths.append({"path": path, "start_node": sn, "end_node": en,
                          "method": method, "child_id": cid})

    # ── 航路点转弯后验 ──
    # 对每个内部汇聚点，取到达航段的末端航向与离开航段的起始航向，
    # 夹角须 ≤ MAX_TURN_ANGLE=60°，否则 is_valid=False（SA 大罚淘汰）。
    # CHECK_WAYPOINT_TURNS=False 时可跳过（M1 消融用）。
    if CHECK_WAYPOINT_TURNS:
        node_arr = {}   # end_node → [arrival headings]
        node_dep = {}   # start_node → departure heading
        for pi in all_paths:
            path = pi['path']
            if len(path) < 2:
                continue
            en = pi.get('end_node')
            sn = pi.get('start_node')
            if en is not None:
                node_arr.setdefault(en, []).append(
                    AStar2DNoise.get_angle(path[-2][:2], path[-1][:2]))
            if sn is not None:
                node_dep[sn] = AStar2DNoise.get_angle(path[0][:2], path[1][:2])
        for node, arr_list in node_arr.items():
            if node not in node_dep:
                continue
            dep = node_dep[node]
            if any(AStar2DNoise.angle_diff(arr, dep) > MAX_TURN_ANGLE for arr in arr_list):
                is_valid = False
                break

    node_height_ranges = {k: (z, z) for k, z in node_heights.items()}
    return all_paths, is_valid, node_height_ranges


def calculate_actual_route_length(all_paths, decoded_tree):
    entry_nodes = []
    def collect_entries(node):
        if node.is_entry: entry_nodes.append(node)
        else: collect_entries(node.left); collect_entries(node.right)
    collect_entries(decoded_tree)
    pbs = {id(pi['start_node']): pi for pi in all_paths if pi.get('start_node')}
    total = 0.0
    for en in entry_nodes:
        rl, cur = 0.0, en
        while cur is not None:
            if id(cur) in pbs:
                pi = pbs[id(cur)]; p = pi['path']
                rl += _grid_path_length_km(p)
                cur = pi.get('end_node')
            else: break
        total += rl
    return total


def paths_to_noise_segments(all_paths):
    """将 all_paths 拆分为噪声评估用小线段 [(x1,y1,h1,x2,y2,h2), ...]。

    每段携带起止点高度，SEL 评估时各采样点高度线性插值。
    """
    segs = []
    for pi in all_paths:
        p = pi['path']
        for j in range(len(p) - 1):
            x1, y1, h1 = p[j][0], p[j][1], p[j][2]
            x2, y2, h2 = p[j + 1][0], p[j + 1][1], p[j + 1][2]
            segs.append((x1, y1, h1, x2, y2, h2))
    return segs


# ============================================================
# 邻域搜索 + SA 求解框架（自包含；拷自 sa.py / tree.py 并适配本文件）
#
# 决策变量 X=(T, Θ)：T=汇聚拓扑（树旋转邻域），Θ=各汇聚点扇环参数 (u,v)。
# 路径解码 = plan_hybrid_paths（本文件的反推 2D A* + max_turns，不变），
# 噪声评估 = SELNoiseModel.exposed_population（不变）。
# 目标 F = w_N·N/N₀ + (1−w_N)·L/L₀（加权和，最小化）。
# ============================================================

BIG_PENALTY = 1.0e6       # A* 失败时的大罚值，远大于正常 F(0~2)
RHO_MAX = 1.5            # 弧弦比阈值：ρ>1.5 判定绕飞，触发定向拓扑重构
SA_T0 = 0.5  # SA 初始温度（标定后：相对目标 F≈0.6，接受率约50-60%）
SA_BETA = 0.90  # 温度衰减系数 T←β·T（T₀=100→T_end≈0.1 @ 150代）
P_TOPO_MOVE = 0.25       # 每步以此概率做拓扑移动，否则做 Θ 扰动
WN = 1              # =0 → 纯距离优化（F = L/L₀）
N_REF = 6169767          # 基线噪声暴露人口（现有进场程序并集）
L_REF = 829.2            # 基线总航线长度 [km]（5条程序含共享段+进近求和）


# --- 拓扑邻域算子：树旋转（叶中序保持不变；拷自 tree.py）---
# 用嵌套元组表示树（叶=进场点id，内部=(left,right)），旋转在元组上做，可证保序。

def _normalize_internal_ids(tree):
    """变换树结构后刷新内部节点 id/depth/min_leaf（与 generate_convergence_structures 同规则）。"""
    internal_nodes = []
    def collect_and_calc(node):
        if node is None: return
        collect_and_calc(node.left); collect_and_calc(node.right)
        if not node.is_entry:
            ll = node.left.depth if not node.left.is_entry else 0
            rl = node.right.depth if not node.right.is_entry else 0
            node.depth = 1 if (node.left.is_entry and node.right.is_entry) else max(ll, rl) + 1
            internal_nodes.append(node)
    collect_and_calc(tree)
    def calc_min(node):
        if node.is_entry: return node.id
        node.min_leaf = min(calc_min(node.left), calc_min(node.right)); return node.min_leaf
    calc_min(tree)
    internal_nodes.sort(key=lambda n: (n.depth, n.min_leaf))
    for i, node in enumerate(internal_nodes): node.id = 6 + i
    return tree


def _tree_to_nested(node):
    if node.is_entry:
        return node.id
    return (_tree_to_nested(node.left), _tree_to_nested(node.right))


def _nested_to_tree(t):
    if isinstance(t, tuple):
        return TreeNode(0, left=_nested_to_tree(t[0]),
                        right=_nested_to_tree(t[1]), is_entry=False)
    return TreeNode(t, is_entry=True)


def _nested_rotations(t):
    """单步旋转邻居（嵌套元组）。右旋 ((a,b),c)→(a,(b,c))，左旋 (a,(b,c))→((a,b),c)，均保序。"""
    if not isinstance(t, tuple):
        return []
    x, y = t
    results = []
    if isinstance(x, tuple):                 # 右旋：x=(a,b)
        a, b = x
        results.append((a, (b, y)))
    if isinstance(y, tuple):                 # 左旋：y=(b,c)
        b, c = y
        results.append(((x, b), c))
    for rx in _nested_rotations(x):          # 左子树内部旋转
        results.append((rx, y))
    for ry in _nested_rotations(y):          # 右子树内部旋转
        results.append((x, ry))
    return results


def tree_rotations(tree):
    """返回 tree 单步树旋转可达的所有合法邻居树（去重，叶中序不变）。"""
    nested = _tree_to_nested(tree)
    seen = set()
    neighbors = []
    for rt in _nested_rotations(nested):
        node = _normalize_internal_ids(_nested_to_tree(rt))
        bracket = tree_to_bracket(node)
        if bracket not in seen:
            seen.add(bracket)
            neighbors.append(node)
    return neighbors


def _leaf_depth_by_id(node, depth=0, out=None):
    if out is None:
        out = {}
    if node.is_entry:
        out[node.id] = depth
    else:
        _leaf_depth_by_id(node.left, depth + 1, out)
        _leaf_depth_by_id(node.right, depth + 1, out)
    return out


def _leaves_under(node, out=None):
    if out is None:
        out = set()
    if node.is_entry:
        out.add(node.id)
    else:
        _leaves_under(node.left, out)
        _leaves_under(node.right, out)
    return out


def _inherit_theta(old_tree, old_theta, new_tree, rng):
    """拓扑旋转后 Θ 继承：以叶集为指纹匹配新旧内部节点。

    旧树已通过 decode_tree 消费 old_theta，遍历序已知（后序：左→右→自身）。
    新树按同一后序遍历每个非根内部节点，查 fingerprint → 命中则继承 (u,v)，
    未命中则随机初始化（新指纹罕有，通常仅被旋转直接影响的 1-2 个节点）。
    """
    old_fp = {}
    idx = [0]

    def harvest(node):
        if node.is_entry:
            return
        harvest(node.left)
        harvest(node.right)
        if node is not old_tree:
            fp = frozenset(_leaves_under(node))
            old_fp[fp] = (old_theta[idx[0]], old_theta[idx[0] + 1])
            idx[0] += 2

    harvest(old_tree)

    theta_new = []

    def walk(node):
        if node.is_entry:
            return
        walk(node.left)
        walk(node.right)
        if node is not new_tree:
            fp = frozenset(_leaves_under(node))
            u, v = old_fp.get(fp, (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)))
            theta_new.extend([u, v])

    walk(new_tree)
    return np.array(theta_new)


def targeted_rotations(tree, high_ratios):
    """优先返回涉及高弧弦比航段的旋转。

    按候选旋转"改变了深度的叶集"与高弧弦比航段覆盖叶集的重叠度排序，重叠多的排前。
    """
    target_leaves = set()
    for r in high_ratios:
        target_leaves |= _leaves_under(r['start_node'])

    cands = tree_rotations(tree)
    base = _leaf_depth_by_id(tree)
    scored = []
    for c in cands:
        cm = _leaf_depth_by_id(c)
        changed = {lid for lid in base if base.get(lid) != cm.get(lid)}
        scored.append((len(changed & target_leaves), c))
    scored.sort(key=lambda s: -s[0])
    return [c for _, c in scored]


def count_internal_params(tree):
    """统计一棵拓扑的 Θ 维度：内部非根节点数 × 2（每个 (u,v)）。与 decode_tree 顺序一致。"""
    internal = []
    def collect(node, is_root=False):
        if node is None or node.is_entry:
            return
        if not is_root:
            internal.append(node)
        collect(node.left, False)
        collect(node.right, False)
    collect(tree, is_root=True)
    return len(internal) * 2


# --- 评价 X=(T,Θ) ---

def compute_arc_chord_ratios(all_paths):
    """每条结构边的弧弦比 ρ=L_A*/|a−b|（跳过 FAF→机场段）。"""
    ratios = []
    for path_info in all_paths:
        if path_info.get('end_node') is None:
            continue
        path = path_info['path']
        arc = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                  for i in range(len(path) - 1))
        chord = calculate_distance(path[0][:2], path[-1][:2])
        if chord < 1e-9:
            continue
        ratios.append({'start_node': path_info['start_node'],
                       'end_node': path_info['end_node'],
                       'rho': arc / chord})
    return ratios


def compute_F_from_paths(all_paths, decoded_tree, is_valid, noise_model, w_n=None):
    """加权目标 F = w_N·N(X)/N₀ + (1−w_N)·L(X)/L₀，最小化。

    A* 失败(is_valid=False) → 返回 (BIG_PENALTY, None, None, None, None)。
    返回 (F, R_N, R_L, N_exposed, L_total)。
    """
    if w_n is None:
        w_n = WN
    if not is_valid:
        return BIG_PENALTY, None, None, None, None
    segs = paths_to_noise_segments(all_paths)
    n_exposed, _, _ = noise_model.exposed_population(segs)
    L_total = calculate_actual_route_length(all_paths, decoded_tree)
    F = w_n * (n_exposed / N_REF) + (1.0 - w_n) * (L_total / L_REF)
    R_N = (N_REF - n_exposed) / N_REF * 100.0   # 正数=改善
    R_L = (L_REF - L_total) / L_REF * 100.0     # 正数=改善
    return F, R_N, R_L, n_exposed, L_total


def evaluate_X(theta, tree, numbered_points, root_pt, P, pathfinder, noise_model, astar):
    """评价候选解 X=(T,Θ)：反推 A* 解码 + SEL 精确暴露人口。

    root_pt: 汇聚根节点（IF/XSY），扇环极点。
    若 ASTAR_WN_AUTO_SYNC=True，自动 ASTAR_W_N ← WN（内外层权重一致）。
    返回 (F, info)。
    """
    if ASTAR_WN_AUTO_SYNC:
        global ASTAR_W_N
        ASTAR_W_N = WN
    tree_copy = copy.deepcopy(tree)
    decoded = decode_tree(tree_copy, list(theta), numbered_points, root_pt)

    # ── 汇聚点间距约束（汇聚点即转弯点，相邻汇聚点/子节点间距 ≥ MIN_TURN_DISTANCE）──
    def _check_node_distances(node):
        if node is None or node.is_entry:
            return True
        for child in (node.left, node.right):
            if child is None:
                continue
            d = math.hypot(node.x - child.x, node.y - child.y)
            if d < MIN_TURN_DISTANCE:
                return False
        return _check_node_distances(node.left) and _check_node_distances(node.right)

    if not _check_node_distances(decoded):
        info = {'decoded_tree': decoded, 'all_paths': [], 'node_height_ranges': {},
                'is_valid': False, 'R_N': None, 'R_L': None,
                'N_exposed': None, 'L_total': None, 'ratios': []}
        return BIG_PENALTY, info

    _astar_mode = True if USE_ASTAR_DECODER else None
    all_paths, is_valid, nhr = plan_hybrid_paths(
        decoded, pathfinder, root_pt, P, astar=astar, astar_segment_id=_astar_mode,
        max_turns=MAX_TURNS_PER_SEGMENT)
    F, R_N, R_L, n_exposed, L_total = compute_F_from_paths(
        all_paths, decoded, is_valid, noise_model)
    ratios = compute_arc_chord_ratios(all_paths) if is_valid else []
    info = {'decoded_tree': decoded, 'all_paths': all_paths,
            'node_height_ranges': nhr, 'is_valid': is_valid,
            'R_N': R_N, 'R_L': R_L,
            'N_exposed': n_exposed, 'L_total': L_total, 'ratios': ratios}
    return F, info


# --- SA 邻域搜索 ---

def neighbor_theta(theta, rng, step=0.15):
    """生成一个邻域解 Θ'：每步只随机扰动一个汇聚点的 (u,v)，其余不变。

    避免全分量同时扰动导致好解被毁灭——后期的精细改进需要小范围局部调整。
    step 是单分量扰动标准差；扇环参数 (u,v)∈[0,1]。
    """
    theta = np.asarray(theta, dtype=float).copy()
    n = len(theta)
    if n < 2:
        return theta
    # 随机选一个汇聚点，扰动它的 u,v 两个分量
    i = rng.randint(0, n // 2) * 2
    theta[i]     = min(max(theta[i]     + rng.normal(0.0, step), 0.0), 1.0)
    theta[i + 1] = min(max(theta[i + 1] + rng.normal(0.0, step), 0.0), 1.0)
    return theta


def sa_accept(dF, T, rng):
    """模拟退火接受准则：更优直接接受；更差以 exp(-dF/T) 概率接受。"""
    if dF < 0:
        return True
    if T <= 1e-12:
        return False
    return rng.random_sample() < math.exp(-dF / T)


def neighborhood_search_run(numbered_points, root_pt, P, pathfinder, noise_model, astar,
                            init_tree=None, max_iter=200, step=0.15, seed=0,
                            T0=SA_T0, beta=SA_BETA, p_topo=P_TOPO_MOVE,
                            log_every=0, log_prefix=""):
    """单条 SA 搜索链：X=(T,Θ) 一起搜。

    每步邻域移动二选一：
      - 位置移动（prob 1-p_topo）：neighbor_theta 扰动 Θ，拓扑不变。
      - 拓扑移动（prob p_topo）：树旋转换 T；有高弧弦比航段(ρ>RHO_MAX)时用
        targeted_rotations 偏向绕飞处；换 T 后 Θ 按叶集指纹继承。
    SA 接受，温度 T←β·T。返回该链最优 (F*, tree*, theta*, info*, history)。

    log_every>0 时每 log_every 代打印一行（调 SA 参数看这个：
    接受率前期应偏高、随降温下降；ΔF 量级用于标定 T0）。
    """
    rng = np.random.RandomState(seed)
    enum_trees = generate_convergence_structures()

    tree = copy.deepcopy(init_tree) if init_tree is not None \
        else enum_trees[rng.randint(0, len(enum_trees))]
    dim = count_internal_params(tree)
    theta = rng.uniform(0.0, 1.0, size=dim)
    F, info = evaluate_X(theta, tree, numbered_points, root_pt, P,
                         pathfinder, noise_model, astar)

    # 初始解无效（A* 失败）时重采样，避免 SA 困在 BIG_PENALTY
    _resample = 0
    while (not info.get("is_valid")) and _resample < 20:
        _resample += 1
        tree = copy.deepcopy(init_tree) if init_tree is not None \
            else enum_trees[rng.randint(0, len(enum_trees))]
        dim = count_internal_params(tree)
        theta = np.full(dim, 0.5) if INIT_THETA_MIDPOINT else rng.uniform(0.0, 1.0, size=dim)
        F, info = evaluate_X(theta, tree, numbered_points, root_pt, P,
                             pathfinder, noise_model, astar)
    if not info.get("is_valid") and log_every > 0:
        print(f"{log_prefix}  WARNING: no valid initial solution after {_resample} attempts")

    best_F, best_tree, best_theta, best_info = F, copy.deepcopy(tree), theta.copy(), info
    history = [F]
    T_cur = T0
    n_accepted = 0
    n_worse_accepted = 0

    if log_every > 0:
        print(f"{log_prefix}  {'代':>4} {'温度T':>10} {'当前F':>10} {'最优F':>10} "
              f"{'移动':>4} {'ΔF':>10} {'接受':>4} {'累计接受率':>9}")
        print(f"{log_prefix}  {0:4d} {T_cur:10.4f} {F:10.4f} {best_F:10.4f} "
              f"{'init':>4} {'-':>10} {'-':>4} {'-':>9}", flush=True)

    for gen in range(1, max_iter + 1):
        do_topo = (rng.random_sample() < p_topo)
        if do_topo:
            high = [r for r in info.get('ratios', []) if r['rho'] > RHO_MAX]
            cands = targeted_rotations(tree, high) if high else tree_rotations(tree)
            if not cands:
                do_topo = False
            else:
                tree_new = cands[rng.randint(0, len(cands))]
                theta_new = _inherit_theta(tree, theta, tree_new, rng)
        if not do_topo:
            tree_new = tree
            theta_new = neighbor_theta(theta, rng, step)

        F_new, info_new = evaluate_X(theta_new, tree_new, numbered_points, root_pt, P,
                                     pathfinder, noise_model, astar)

        dF = F_new - F
        accepted = sa_accept(dF, T_cur, rng)
        if accepted:
            tree, theta, F, info = tree_new, theta_new, F_new, info_new
            n_accepted += 1
            if dF >= 0:
                n_worse_accepted += 1
            if F < best_F:
                best_F, best_tree, best_theta, best_info = \
                    F, copy.deepcopy(tree), theta.copy(), info

        history.append(F)

        if log_every > 0 and (gen % log_every == 0 or gen == max_iter):
            move = 'topo' if do_topo else 'pos'
            print(f"{log_prefix}  {gen:4d} {T_cur:10.4f} {F:10.4f} {best_F:10.4f} "
                  f"{move:>4} {dF:+10.4f} {'是' if accepted else '否':>4} "
                  f"{n_accepted / gen * 100:8.1f}%", flush=True)

        T_cur *= beta

    if log_every > 0:
        print(f"{log_prefix}  链结束：接受 {n_accepted}/{max_iter} "
              f"({n_accepted / max_iter * 100:.1f}%)，其中更差仍接受 {n_worse_accepted} 次；"
              f"最优 F*={best_F:.4f}")

    return best_F, best_tree, best_theta, best_info, history


# ============================================================
# 可视化
# ============================================================

# 优化方案线条样式
_OPT_LINE = dict(color='black', linewidth=2.0, alpha=0.92, zorder=4)


def _build_baseline_paths_2d():
    """构建现有进场程序的可视化数据（直接复用 viz_procedures 航路点）。

    返回: [(proc_name, path_xy, color, wp_xy_list), ...]
      path_xy: 折线点 [(x,y), ...]（含所有航段）
      color:   航路点颜色
      wp_xy:   [(x, y, name), ...] 航路点标注位置
    """
    import viz_procedures as _vp

    results = []
    for _pname, _pts, _color in _vp.ALL_PROCEDURES:
        xy = [_vp.latlon_to_grid(lat, lon) for _, lat, lon, _ in _pts]
        wp_xy = [(x, y, name) for (name, _, _, _), (x, y) in zip(_pts, xy)]
        results.append((_pname, xy, _color, wp_xy))
    return results


def visualize_result(decoded_tree, all_paths, pathfinder, FAF, P,
                     node_height_ranges=None, title="", pop=None,
                     fname_prefix="test_result", show_baseline=True):
    """顶刊风格可视化：2D 平面图（3D 仅作补充）。

    2D 图含优化航线（彩实线）、基线进场程序（灰虚线）和人口密度底图。
    """
    baseline_data = _build_baseline_paths_2d() if show_baseline else []

    # ================================================================
    # 3D 视图（快速预览，论文以 2D 图为主）
    # ================================================================
    fig_3d = plt.figure(figsize=(14, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.set_xlim(0, pathfinder.grid_size_x)
    ax_3d.set_ylim(0, pathfinder.grid_size_y)
    ax_3d.set_zlim(0, 6500)
    ax_3d.set_box_aspect([1, 1, 0.32])
    ax_3d.set_xlabel('X (km)', fontsize=10)
    ax_3d.set_ylabel('Y (km)', fontsize=10)
    ax_3d.set_zlabel('Altitude (m)', fontsize=10)
    ax_3d.set_title(title, fontsize=13, fontweight='bold', pad=12)

    if show_baseline:
        for _bname, _bpts, _color, _wps in baseline_data:
            bx, by = [b[0] for b in _bpts], [b[1] for b in _bpts]
            ax_3d.plot(bx, by, [0]*len(_bpts),
                       color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    for idx, pi in enumerate(all_paths):
        p = pi['path']
        if p:
            arr = np.array(p)
            ax_3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], **_OPT_LINE)

    def _get_node_z(node):
        for pi_ in all_paths:
            if pi_.get('end_node') is node and pi_['path']:
                return pi_['path'][-1][2]
        k = (node.x, node.y)
        if node_height_ranges and k in node_height_ranges:
            return (node_height_ranges[k][0] + node_height_ranges[k][1]) / 2
        return pathfinder.faf_height if pathfinder.faf_height > 0 else IF_HEIGHT

    def _plot_3d_nodes(node, is_root=False):
        if node.is_entry:
            z = 0
            for pi_ in all_paths:
                if pi_.get('start_node') is node and pi_['path']:
                    z = pi_['path'][0][2]; break
            ax_3d.scatter(node.x, node.y, z, c='#2E7D32', s=100, marker='^',
                          edgecolor='white', linewidth=1.0, zorder=10)
            ax_3d.text(node.x + 2, node.y + 2, z + 120,
                       f'E{node.id}', fontsize=8, fontweight='bold')
        else:
            if is_root:
                ax_3d.scatter(node.x, node.y, _get_node_z(node),
                              c='#F9A825', s=80, marker='o',
                              edgecolor='white', linewidth=1.0, zorder=10)
                ax_3d.text(node.x + 2, node.y + 2, _get_node_z(node) + 120,
                           'IF', fontsize=9, fontweight='bold')
            _plot_3d_nodes(node.left)
            _plot_3d_nodes(node.right)

    _plot_3d_nodes(decoded_tree, is_root=True)
    ax_3d.scatter(FAF[0], FAF[1], FAF_HEIGHT, c='#E65100', s=70, marker='s',
                  edgecolor='white', linewidth=1.0, zorder=10)
    ax_3d.text(FAF[0] + 2, FAF[1] + 2, FAF_HEIGHT + 120,
               'FAF', fontsize=9, fontweight='bold')
    ax_3d.scatter(P[0], P[1], 0, c='#B71C1C', s=160, marker='*',
                  edgecolor='white', linewidth=1.0, zorder=10)
    ax_3d.view_init(elev=25, azim=45)
    fig_3d.tight_layout()
    fig_3d.savefig(f'{fname_prefix}_3d.png', dpi=300, bbox_inches='tight')
    print(f"3D saved: {fname_prefix}_3d.png")
    plt.close(fig_3d)

    # ================================================================
    # 2D 主图（论文用图）
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(13, 13))
    ax2.set_xlim(0, pathfinder.grid_size_x)
    ax2.set_ylim(0, pathfinder.grid_size_y)
    ax2.set_aspect('equal')
    _ticks = [0, 50, 100, 150, 200, 250]
    map_base.apply_latlon_ticks(ax2, _ticks, _ticks, fontsize=16)

    # 人口底图
    if pop is not None:
        ny, nx = pop.shape
        im = ax2.imshow(pop + 1, origin='lower', cmap='YlOrBr',
                        norm=LogNorm(vmin=1, vmax=max(pop.max(), 2)),
                        extent=[0, nx, 0, ny], alpha=0.72, zorder=0)
        cbar = fig2.colorbar(im, ax=ax2, fraction=0.045, pad=0.03, shrink=0.92)
        cbar.set_label('Population density (persons/km²)', fontsize=18)
        cbar.ax.tick_params(labelsize=15)

    # 基线程序（viz_procedures 风格：黑色虚线 + 彩色航路点）
    if show_baseline:
        _baseline_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        for idx, (_bname, _bpts, _color, _wps) in enumerate(baseline_data):
            bx, by = [b[0] for b in _bpts], [b[1] for b in _bpts]
            ax2.plot(bx, by, color='black', linestyle='--',
                     linewidth=1.5, alpha=0.7)
            for wx, wy, wname in _wps:
                ax2.scatter(wx, wy,
                            c=_baseline_colors[idx % len(_baseline_colors)],
                            s=40, marker='o', edgecolor='black',
                            linewidth=0.5, zorder=5)

    # 优化方案（黑色实线）
    for _idx, pi in enumerate(all_paths):
        p = pi['path']
        if p:
            arr = np.array(p)
            ax2.plot(arr[:, 0], arr[:, 1], **_OPT_LINE)

    # 节点标注
    def _plot_2d_nodes(node, is_root=False):
        if node.is_entry:
            ax2.scatter(node.x, node.y, c='#2E7D32', s=110, marker='^',
                        edgecolor='white', linewidth=1.2, zorder=10)
            ax2.text(node.x + 4, node.y + 4, f'E{node.id}',
                     fontsize=16, fontweight='bold', color='#1B5E20')
        else:
            if is_root:
                ax2.scatter(node.x, node.y, c='#F9A825', s=90, marker='o',
                            edgecolor='white', linewidth=1.2, zorder=10)
                ax2.text(node.x + 4, node.y + 4, 'IF',
                         fontsize=16, fontweight='bold', color='#E65100')
            _plot_2d_nodes(node.left)
            _plot_2d_nodes(node.right)

    _plot_2d_nodes(decoded_tree, is_root=True)

    # FAF
    ax2.scatter(FAF[0], FAF[1], c='#E65100', s=70, marker='s',
                edgecolor='white', linewidth=1.2, zorder=10)
    ax2.text(FAF[0] + 4, FAF[1] + 3, 'FAF',
             fontsize=16, fontweight='bold', color='#BF360C')


    # 图例
    from matplotlib.lines import Line2D
    legend_elements = []
    if show_baseline:
        legend_elements.append(
            Line2D([0], [0], color='black', linestyle='--',
                   linewidth=1.5, label='Baseline procedures'))
    legend_elements.append(
        Line2D([0], [0], color='black', linewidth=2.5,
               label='Optimized procedures'))
    ax2.legend(handles=legend_elements, loc='upper left',
               fontsize=13, framealpha=0.9, edgecolor='#CCCCCC')

    map_base.add_scale_bar(ax2, 20)
    map_base.add_north_arrow(ax2)

    fig2.tight_layout()
    fig2.savefig(f'{fname_prefix}_2d.png', dpi=300, bbox_inches='tight')
    print(f"2D saved: {fname_prefix}_2d.png")
    plt.close(fig2)


# ============================================================
# 坐标 + 主程序
# ============================================================

_airport_ll = (dms(31, 10, 18), dms(121, 47, 0))        # PUD—浦东机场跑道
P = np.array(latlon_to_grid(*_airport_ll))

_IF_LL = (dms(30, 55, 54), dms(121, 52, 24))             # IF (XSY), 900m, 汇聚根节点
IF_PT = np.array(latlon_to_grid(*_IF_LL))

# FAF 在 PUD→IF 大圆上，距 PUD 11.8 NM（与 viz_procedures 一致）
def _great_circle_destination_to_pt(lat_a, lon_a, pt_b, dist_km):
    """从 A 沿大圆路径走向 B，走 dist_km 后的 (lat, lon)。"""
    p1, p2 = math.radians(lat_a), math.radians(pt_b[0])
    dl = math.radians(pt_b[1] - lon_a)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = math.degrees(math.atan2(y, x))
    R = 6371.0
    lat1 = math.radians(lat_a); lon1 = math.radians(lon_a)
    brng = math.radians(bearing)
    dR = dist_km / R
    lat2 = math.asin(math.sin(lat1) * math.cos(dR)
                     + math.cos(lat1) * math.sin(dR) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(dR) * math.cos(lat1),
                              math.cos(dR) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

_FAF_LL = _great_circle_destination_to_pt(*_airport_ll, _IF_LL, 11.8 * 1.852)
FAF = np.array(latlon_to_grid(*_FAF_LL))
# FAF 在 PUD→IF 线上；IF→PUD 航向与 FAF→PUD 航向一致（共线）。

_entry_ll = [
    (dms(31, 35, 22), dms(120, 19, 10)), (dms(30, 15.4), dms(121, 13.3)),
    (dms(29, 53.7), dms(121, 20.0)), (dms(31, 39, 36), dms(122, 38, 0)),
    (dms(31, 21, 39), dms(122, 46, 30)),
]
ENTRY_ALTS = [6000, 6000, 6000, 5100, 4800]

def _clamp(x, y):
    return (min(max(x, 0.0), GRID_SIZE_X - 1.0), min(max(y, 0.0), GRID_SIZE_Y - 1.0))
entry_points = np.array([_clamp(*latlon_to_grid(lat, lon)) for lat, lon in _entry_ll])

PF = IF_PT - P
def clockwise_angle(v1, v2):
    def cw(v):
        a = np.arctan2(v[1], v[0]); return 2 * np.pi - a if a < 0 else 2 * np.pi - a
    a1, a2 = cw(v1), cw(v2); return a2 - a1 if a2 >= a1 else 2 * np.pi - (a1 - a2)
angles = []
for i, pt in enumerate(entry_points):
    angles.append((clockwise_angle(PF, P - pt), i, pt))
angles.sort(key=lambda x: x[0])
numbered_points = [(i + 1, angles[i][2]) for i in range(len(angles))]


if __name__ == "__main__":
    print("=" * 70)
    print("test.py — SA 邻域搜索 (T,Θ) + 反推A*最小噪声路径解码")
    print("=" * 70)

    _names = ['SASAN', 'ANDONG', 'LISHE', 'MATNU', 'DUMET']
    print(f"\n坐标:"); print(f"  PUD=({P[0]:.0f},{P[1]:.0f})  FAF=({FAF[0]:.0f},{FAF[1]:.0f})  IF=({IF_PT[0]:.0f},{IF_PT[1]:.0f})")
    for i, (nm, pt) in enumerate(zip(_names, entry_points)):
        print(f"  E{i+1} {nm:7s}=({pt[0]:.0f},{pt[1]:.0f}) alt={ENTRY_ALTS[i]}m")

    # 加载人口
    pop = None; noise_model = None
    try:
        pop, N_pop = load_population_grid('population_data_square.xlsx')
        print(f"\n人口: shape={pop.shape} N={N_pop:.0f}")
        noise_model = SELNoiseModel(pop)
        print(f"  α_atm={noise_model._alpha_atm*1000:.3f} dB/km "
              f"r_eff_table[500m]={noise_model._r_eff_table[5]:.1f}km "
              f"r_eff_table[4000m]={noise_model._r_eff_table[40]:.1f}km")
    except Exception as e:
        print(f"\n无人口数据: {e}")

    if noise_model is None:
        print("\n目标 F=w_N·N/N₀+(1−w_N)·L/L₀ 依赖人口数据，无法求解，退出。")
    else:
        pathfinder = PathfinderConfig()
        astar_sa = AStar2DNoise(noise_model)

        # 固定拓扑（TARGET_TOPOLOGY 非 None 时 p_topo=0，只搜 Θ）
        if TARGET_TOPOLOGY is not None:
            all_trees = generate_convergence_structures()
            target_tree = next(t for t in all_trees
                               if tree_to_bracket(t) == TARGET_TOPOLOGY)
            print(f"\n固定拓扑: {TARGET_TOPOLOGY}")
        else:
            target_tree = None

        print(f"\n{'─' * 50}")
        p_topo_actual = 0.0 if target_tree is not None else P_TOPO_MOVE
        print(f"SA 邻域搜索 (max_iter={NS_MAX_ITER}, "
              f"T0={SA_T0}, beta={SA_BETA}, p_topo={p_topo_actual})")
        print(f"{'─' * 50}")

        # 单次评价耗时基准 → 总耗时估计
        _probe_tree = target_tree if target_tree is not None \
            else generate_convergence_structures()[0]
        _probe_theta = np.random.RandomState(0).uniform(
            0.0, 1.0, count_internal_params(_probe_tree))
        _t0 = time.time()
        F_probe, _ = evaluate_X(_probe_theta, _probe_tree, numbered_points, IF_PT, P,
                                pathfinder, noise_model, astar_sa)
        _dt = time.time() - _t0
        _n_evals = NS_MAX_ITER + 1
        print(f"  单次评价 {_dt:.1f}s (F={F_probe:.4f})，共约 {_n_evals} 次评价 "
              f"≈ {_dt * _n_evals / 60:.0f} min（上界，A* 缓存命中会更快）", flush=True)

        F_best, tree_best, theta_best, info_best, history = neighborhood_search_run(
            numbered_points, IF_PT, P, pathfinder, noise_model, astar_sa,
            init_tree=target_tree, max_iter=NS_MAX_ITER, seed=0,
            log_every=1, p_topo=p_topo_actual)

        bracket = tree_to_bracket(tree_best)
        N_best = info_best.get('N_exposed')
        L_best = info_best.get('L_total') or 0.0
        R_N = info_best.get('R_N')
        R_L = info_best.get('R_L')
        print(f"\n  最优解: 拓扑 {bracket}")
        print(f"    F = w_N·N/N₀ + (1−w_N)·L/L₀ = {F_best:.4f}  (w_N={WN})")
        print(f"    N_exposed = {N_best:,.0f}   R_N = {R_N:+.2f}%")
        print(f"    L_total   = {L_best:.1f} km   R_L = {R_L:+.2f}%")
        print(f"    (基线: N₀={N_REF:,}, L₀={L_REF} km)")

        visualize_result(info_best['decoded_tree'], info_best['all_paths'],
                         pathfinder, FAF, P,
                         node_height_ranges=info_best['node_height_ranges'],
                         pop=pop,
                         title=f"SA best {bracket}  F={F_best:.4f}  "
                               f"N={N_best:,.0f} (R_N={R_N:+.1f}%)  "
                               f"L={L_best:.1f}km (R_L={R_L:+.1f}%)")

    print("\nDone.")

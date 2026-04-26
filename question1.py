"""

此程序解决了问题一。
主要思路：
1. 数据预处理：将客户需求拆分成多个虚拟节点，确保每个节点的重量和体积都在安全阈值内，便于后续调度。
2. 成本评估：在计算路径成本时，模拟行驶过程，根据当前时间段动态调整速度，计算行驶时间和能耗
3. 优化算法：采用两阶段混合优化
    - 第一阶段：ALNS大邻域搜索，随机破坏和重建路径
    - 第二阶段：MA模因精修，种群进化和局部搜索
4. 多次随机种子尝试，寻找全局最优解
5. 输出单车单趟最佳调度方案
6. 基于最佳方案，生成一车多趟的排班方案，确保每辆实体车的回仓和周转时间满足要求，同时最大化节省启动费。
7. 生成详细的排班方案文本文件，并在控制台输出统计结果。


最终生成的报告文件在当前目录下：
    - Q1_单车单趟明细排班方案.txt:    包含每辆车的调度详情和统计结果
    - Q1一车多趟的最终排班.txt:      包含基于单车单趟结果生成的一车多趟排班方案和统计结果
日志文件：
    - logs/q1.log: 包含算法执行过程中的详细日志


"""

import copy
import logging
import math
import random
import sys
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import re
import os
from datetime import datetime

# 日志与基础配置
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/q1.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
font_names = ["SimHei", "Microsoft YaHei", "STHeiti", "SimSun"]
for name in font_names:
    if name in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams["font.sans-serif"] = [name]
        break

#  全局参数与交通规则配置
SERVICE_TIME = 20.0
TURNAROUND_TIME = 30.0
FIXED_COST = 400.0

# 交通拥堵时段及对应平均时速 (开始时间, 结束时间, 平均速度)
TRAFFIC_INTERVALS = [
    (0, 480, 35.4),
    (480, 540, 9.8),
    (540, 600, 55.3),
    (600, 690, 35.4),
    (690, 780, 9.8),
    (780, 900, 55.3),
    (900, 1020, 35.4),
    (1020, 99999, 35.4),
]


class TDVRP_Ultimate_Solver:
    def __init__(self):
        self.c_fixed = 400.0
        self.c_wait, self.c_late = 20.0 / 60.0, 50.0 / 60.0
        self.c_fuel, self.c_ev = 7.61, 1.64
        self.c_carbon = 0.65
        self.eta_fuel, self.gamma_ev = 2.547, 0.501
        self.service_time = 20.0

        self.traffic_intervals = [
            (0, 480, 35.4, 5.2),
            (480, 540, 9.8, 4.7),
            (540, 600, 55.3, 0.1),
            (600, 690, 35.4, 5.2),
            (690, 780, 9.8, 4.7),
            (780, 900, 55.3, 0.1),
            (900, 1020, 35.4, 5.2),
            (1020, 99999, 35.4, 5.2),
        ]
        self.departure_options = [float(t) for t in range(480, 961, 30)]

        raw_configs = [
            {"name": "EV-1", "type": 1, "cap_kg": 3000, "cap_m3": 15.0, "count": 10},
            {"name": "EV-2", "type": 1, "cap_kg": 1250, "cap_m3": 8.5, "count": 15},
            {"name": "Fuel-1", "type": 0, "cap_kg": 3000, "cap_m3": 13.5, "count": 60},
            {"name": "Fuel-2", "type": 0, "cap_kg": 1500, "cap_m3": 10.8, "count": 50},
            {"name": "Fuel-3", "type": 0, "cap_kg": 1250, "cap_m3": 6.5, "count": 50},
        ]
        self.fleet_pool = []
        for cfg in raw_configs:
            for _ in range(int(cfg["count"])):
                self.fleet_pool.append(cfg.copy())

        self.customers, self.dist_matrix = {}, {}

        self.alns_iters = 250
        self.ma_pop_size = 200
        self.ma_generations = 150
        self.T_start, self.T_end, self.cooling_rate = 1000.0, 0.01, 0.97
        self.q_min, self.q_max = 5, 20
        self.early_stop_patience = 40

    def _read_csv_auto(self, file_path):
        tried = []
        for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(file_path, encoding=enc)
            except UnicodeDecodeError as e:
                tried.append(f"{enc}: {e}")
        raise ValueError(f"无法读取文件 {file_path}，已尝试编码: {', '.join(tried)}")

    def load_data(self, cust_file, dist_file):
        df_cust = self._read_csv_auto(cust_file)
        df_dist = self._read_csv_auto(dist_file)

        raw_dist = {}
        for _, row in df_dist.iterrows():
            s, e = int(row["起点ID"]), int(row["终点ID"])
            if s not in raw_dist:
                raw_dist[s] = {}
            raw_dist[s][e] = float(row["路网距离"])

        self.customers[0] = {
            "weight": 0,
            "vol": 0,
            "ready": 0,
            "due": 99999,
            "orig_id": 0,
        }

        vid_counter = 1000
        orig_to_vids = {0: [0]}

        SAFE_W = 1200.0
        SAFE_V = 6.0

        for _, row in df_cust.iterrows():
            cid = int(row["目标客户编号"])
            w, v = float(row["重量"]), float(row["体积"])
            rdy, du = float(row["开始时间"]), float(row["结束时间"])
            orig_to_vids[cid] = []

            while w > SAFE_W or v > SAFE_V:
                ratio = min(SAFE_W / w if w > 0 else 1.0, SAFE_V / v if v > 0 else 1.0)
                chunk_w = w * ratio
                chunk_v = v * ratio

                node_id = cid if len(orig_to_vids[cid]) == 0 else vid_counter
                if node_id == vid_counter:
                    vid_counter += 1

                self.customers[node_id] = {
                    "weight": chunk_w,
                    "vol": chunk_v,
                    "ready": rdy,
                    "due": du,
                    "orig_id": cid,
                }
                orig_to_vids[cid].append(node_id)

                w -= chunk_w
                v -= chunk_v

            if w > 0.001 or v > 0.001:
                node_id = cid if len(orig_to_vids[cid]) == 0 else vid_counter
                if node_id == vid_counter:
                    vid_counter += 1
                self.customers[node_id] = {
                    "weight": w,
                    "vol": v,
                    "ready": rdy,
                    "due": du,
                    "orig_id": cid,
                }
                orig_to_vids[cid].append(node_id)

        for cid1, vids1 in orig_to_vids.items():
            for cid2, vids2 in orig_to_vids.items():
                dist = 0.0 if cid1 == cid2 else raw_dist.get(cid1, {}).get(cid2, 999.0)
                for v1 in vids1:
                    if v1 not in self.dist_matrix:
                        self.dist_matrix[v1] = {}
                    for v2 in vids2:
                        self.dist_matrix[v1][v2] = dist

    def simulate_segment(self, dist, start_t, current_load, veh, speed_mode="mean"):
        t, d_left, en_sum = start_t, dist, 0.0
        load_ratio = min(current_load / veh["cap_kg"], 1.0)
        while d_left > 1e-4:
            v_avg, std_dev, t_end = 35.4, 5.2, 99999
            for s_i, e_i, v_i, std_i in self.traffic_intervals:
                if s_i <= t < e_i:
                    v_avg, std_dev, t_end = v_i, std_i, e_i
                    break
            v_actual = (
                v_avg
                if speed_mode == "mean"
                else max(5.0, min(np.random.normal(v_avg, std_dev), 80.0))
            )
            v_min = v_actual / 60.0
            d_run = min(d_left, (t_end - t) * v_min)
            t += d_run / v_min
            d_left -= d_run
            base = (
                (31.75 - 0.2554 * v_actual + 0.0025 * (v_actual**2))
                if veh["type"] == 0
                else (36.19 - 0.12 * v_actual + 0.0014 * (v_actual**2))
            )
            en_sum += (
                (base / 100.0)
                * d_run
                * (1 + (0.4 if veh["type"] == 0 else 0.35) * load_ratio)
            )
        return t, en_sum

    def evaluate_route_total_cost(self, path, veh, load, speed_mode="mean"):
        best_sub, best_ts, best_carbon = float("inf"), 480.0, 0.0
        for ts in self.departure_options:
            sub, ct, cw, route_carbon = self.c_fixed, ts, load, 0.0
            for i in range(len(path) - 1):
                nxt = path[i + 1]
                d = self.dist_matrix.get(path[i], {}).get(nxt, 999.0)
                arr_t, en = self.simulate_segment(d, ct, cw, veh, speed_mode=speed_mode)
                step_carbon = en * (
                    self.eta_fuel if veh["type"] == 0 else self.gamma_ev
                )
                sub += (
                    en * (self.c_fuel if veh["type"] == 0 else self.c_ev)
                    + step_carbon * self.c_carbon
                )
                route_carbon += step_carbon
                if nxt != 0:
                    tw_s, tw_e = (
                        self.customers[nxt]["ready"],
                        self.customers[nxt]["due"],
                    )
                    sub += (
                        max(0, tw_s - arr_t) * self.c_wait
                        + max(0, arr_t - tw_e) * self.c_late
                    )
                    ct, cw = (
                        max(arr_t, tw_s) + self.service_time,
                        cw - self.customers[nxt]["weight"],
                    )
                else:
                    ct = arr_t
            if sub < best_sub:
                best_sub, best_ts, best_carbon = sub, ts, route_carbon
        return best_sub, best_ts, best_carbon

    def apply_advanced_local_search(self, path, veh, load, speed_mode="mean"):
        best_p = copy.copy(path)
        best_c, _, _ = self.evaluate_route_total_cost(
            best_p, veh, load, speed_mode=speed_mode
        )

        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_p) - 2):
                for j in range(i + 1, len(best_p) - 1):
                    new_p = best_p[:i] + best_p[i : j + 1][::-1] + best_p[j + 1 :]
                    new_c, _, _ = self.evaluate_route_total_cost(
                        new_p, veh, load, speed_mode=speed_mode
                    )
                    if new_c < best_c - 1e-4:
                        best_c, best_p, improved = new_c, new_p, True

            if not improved:
                for i in range(1, len(best_p) - 1):
                    for j in range(i + 1, len(best_p) - 1):
                        new_p = copy.copy(best_p)
                        new_p[i], new_p[j] = new_p[j], new_p[i]
                        new_c, _, _ = self.evaluate_route_total_cost(
                            new_p, veh, load, speed_mode=speed_mode
                        )
                        if new_c < best_c - 1e-4:
                            best_c, best_p, improved = new_c, new_p, True
        return best_p

    def compute_cost_and_optimize(
        self, sequence, return_details=False, speed_mode="mean"
    ):
        routes, curr_p, w_s, v_s, v_idx = [], [], 0.0, 0.0, 0
        for cid in sequence:
            c = self.customers[cid]
            veh = self.fleet_pool[v_idx]

            if w_s + c["weight"] <= veh["cap_kg"] and v_s + c["vol"] <= veh["cap_m3"]:
                curr_p.append(cid)
                w_s += c["weight"]
                v_s += c["vol"]
            else:
                if curr_p:
                    routes.append({"veh": veh, "path": [0] + curr_p + [0], "load": w_s})
                    v_idx += 1
                    if v_idx >= len(self.fleet_pool):
                        return (
                            (1e12, [], sequence, 0)
                            if return_details
                            else (1e12, sequence)
                        )
                    veh = self.fleet_pool[v_idx]

                while c["weight"] > veh["cap_kg"] or c["vol"] > veh["cap_m3"]:
                    v_idx += 1
                    if v_idx >= len(self.fleet_pool):
                        return (
                            (1e12, [], sequence, 0)
                            if return_details
                            else (1e12, sequence)
                        )
                    veh = self.fleet_pool[v_idx]

                curr_p, w_s, v_s = [cid], c["weight"], c["vol"]

        if curr_p:
            routes.append(
                {"veh": self.fleet_pool[v_idx], "path": [0] + curr_p + [0], "load": w_s}
            )

        total_cost, total_carbon, optimized_seq = 0.0, 0.0, []
        for r in routes:
            optimized_seq.extend(r["path"][1:-1])
            sub_c, ts_opt, r_carb = self.evaluate_route_total_cost(
                r["path"], r["veh"], r["load"], speed_mode=speed_mode
            )
            r["best_start_time"], r["sub_cost"], r["carbon"] = ts_opt, sub_c, r_carb
            total_cost += sub_c
            total_carbon += r_carb

        if return_details:
            return total_cost, routes, optimized_seq, total_carbon
        return total_cost, optimized_seq

    def run_hybrid_optimization(self):
        c_ids = [cid for cid in self.customers.keys() if cid != 0]
        curr_seq = random.sample(c_ids, len(c_ids))
        curr_c, curr_seq = self.compute_cost_and_optimize(curr_seq, speed_mode="mean")
        best_seq, best_c, history = copy.deepcopy(curr_seq), curr_c, [curr_c]
        T = self.T_start

        logging.info(">> 启动第一阶段: ALNS 大邻域压榨")
        for it in range(self.alns_iters):
            q = random.randint(self.q_min, self.q_max)
            if random.random() < 0.5:
                removed = random.sample(curr_seq, q)
                destroyed = [x for x in curr_seq if x not in removed]
            else:
                start = random.randint(0, len(curr_seq) - q)
                removed = curr_seq[start : start + q]
                destroyed = curr_seq[:start] + curr_seq[start + q :]

            new_s = list(destroyed)
            for r in removed:
                bc, bs = float("inf"), None
                for idx in random.sample(range(len(new_s) + 1), min(6, len(new_s) + 1)):
                    ts = new_s[:idx] + [r] + new_s[idx:]
                    cost, _ = self.compute_cost_and_optimize(ts)
                    if cost < bc:
                        bc, bs = cost, ts
                new_s = bs

            nc, ns = self.compute_cost_and_optimize(new_s)

            if nc < curr_c or random.random() < math.exp((curr_c - nc) / max(T, 1e-9)):
                curr_seq, curr_c = ns, nc
                if nc < best_c:
                    best_seq, best_c = copy.deepcopy(ns), nc

            T = max(self.T_end, T * self.cooling_rate)
            history.append(best_c)
            if it % 20 == 0:
                logging.info(
                    f"   ALNS Iter {it:3d} | 最优成本: {best_c:.2f} | 此时温度: {T:.1f}"
                )

        logging.info(f">> 启动第二阶段: MA 模因精修 (起始极值: {best_c:.2f})")
        pop = [copy.deepcopy(best_seq) for _ in range(20)] + [
            random.sample(c_ids, len(c_ids)) for _ in range(self.ma_pop_size - 20)
        ]
        stag = 0

        for g in range(self.ma_generations):
            fits, n_pop = [], []
            for ind in pop:
                c, opt = self.compute_cost_and_optimize(ind, speed_mode="mean")
                fits.append(c)
                n_pop.append(opt)
            pop, midx = n_pop, int(np.argmin(fits))

            if fits[midx] < best_c:
                best_c, best_seq, stag = fits[midx], copy.deepcopy(pop[midx]), 0
            else:
                stag += 1

            history.append(best_c)
            if g % 10 == 0:
                logging.info(f"   MA Gen {g:3d} | 极限压榨成本: {best_c:.2f}")
            if stag >= self.early_stop_patience:
                logging.info(f"   算法在第 {g} 代收敛，已触及当前种子的极限解。")
                break

            np_new = [copy.deepcopy(best_seq)] * 2
            while len(np_new) < self.ma_pop_size:
                p1 = pop[min(random.sample(range(len(pop)), 3), key=lambda i: fits[i])]
                p2 = pop[min(random.sample(range(len(pop)), 3), key=lambda i: fits[i])]
                a, b = sorted(random.sample(range(len(p1)), 2))
                c1 = [-1] * len(p1)
                c1[a : b + 1] = p1[a : b + 1]
                ptr = (b + 1) % len(p1)
                for x in p2:
                    if x not in c1:
                        c1[ptr] = x
                        ptr = (ptr + 1) % len(p1)

                mut_rate = 0.3 if stag > 10 else 0.1
                if random.random() < mut_rate:
                    i, j = random.sample(range(len(c1)), 2)
                    c1[i], c1[j] = c1[j], c1[i]
                np_new.append(c1)
            pop = np_new[: self.ma_pop_size]

        return best_seq, history


def execute_universe(seed):
    random.seed(seed)
    np.random.seed(seed)
    solver = TDVRP_Ultimate_Solver()
    solver.load_data("总表_5.csv", "路网距离对照表.csv")
    logging.info(f"\n{'=' * 40}\n>>> 开启终极搜索宇宙 (Seed: {seed})\n{'=' * 40}")
    t0 = time.time()
    best_seq, history = solver.run_hybrid_optimization()
    fc, fr, _, fcarb = solver.compute_cost_and_optimize(
        best_seq, return_details=True, speed_mode="mean"
    )
    t1 = time.time()
    logging.info(f"宇宙 {seed} 探索完毕。耗时: {t1 - t0:.1f}秒 | 斩获成本: {fc:.2f}")
    return fc, fr, history, fcarb, seed


def load_geodata(file_path="总表_5.csv"):
    """
    模拟加载坐标
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        coords = {
            int(r["目标客户编号"]): (r["X (km)"], r["Y (km)"]) for _, r in df.iterrows()
        }
    else:
        coords = {}
    coords[0] = (20, 20)
    return coords


def time_to_min(t_str):
    h, m = map(int, t_str.split(":"))
    return h * 60 + m


def min_to_time(minutes):
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"


def get_dynamic_travel_time(dist, start_t):
    """
    基于给定的距离和当前发车时间，使用真实的、分时段变化的平均限速计算到达时间。
    完全复刻自 TDVRP_Ultimate_Solver 的限速分段模拟逻辑。
    """
    t = start_t
    d_left = dist
    while d_left > 1e-4:
        v_avg, t_end = 35.4, 99999
        for s_i, e_i, v_i in TRAFFIC_INTERVALS:
            if s_i <= t < e_i:
                v_avg, t_end = v_i, e_i
                break

        # 将速度转为 km/min 进行计算
        v_min = v_avg / 60.0

        # 在当前交通时段内，最多能跑的距离
        d_run = min(d_left, (t_end - t) * v_min)

        t += d_run / v_min
        d_left -= d_run

    return t


def generate_mtvrp_schedule(input_file, coords, output_file="一车多趟排班方案.txt"):
    if not os.path.exists(input_file):
        return f"错误：找不到输入文件 {input_file}"

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 提取：原始车辆编号、车型、发车时间、成本、路径
    pattern = r"▶ 车辆编号: (\d+) \|.*?车型: \[(.*?)\]\s+- 发车时间: (\d{2}:\d{2})\s+- 本趟成本: ([\d.]+) 元\s+- 行驶路径: ([\d -> ]+)"
    trips_raw = re.findall(pattern, content, re.DOTALL)

    all_trips = []
    for raw_id, v_type, start_str, cost, path_str in trips_raw:
        path = [int(x) for x in path_str.split(" -> ")]
        start_min = time_to_min(start_str)

        # 计算回仓时间 (使用动态、随时间变化的限速逻辑)
        curr_t = start_min
        for i in range(len(path) - 1):
            p1 = coords.get(path[i], (20, 20))  # 默认坐标容错
            p2 = coords.get(path[i + 1], (20, 20))
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

            # 【此处替换原恒定速度逻辑】: 获取随时间变动的到达时间
            curr_t = get_dynamic_travel_time(dist, curr_t)

            if path[i + 1] != 0:
                curr_t += SERVICE_TIME

        all_trips.append(
            {
                "orig_id": raw_id,
                "type": v_type,
                "start_min": start_min,
                "start_str": start_str,
                "end_min": curr_t,
                "cost": float(cost),
                "path": path_str.strip(),
            }
        )

    # 按发车时间排序
    all_trips.sort(key=lambda x: x["start_min"])

    # 物理车辆分配
    fleet = {}  # {车型: [实体车1, 实体车2...]}

    for trip in all_trips:
        vt = trip["type"]
        if vt not in fleet:
            fleet[vt] = []

        assigned = False
        for idx, veh in enumerate(fleet[vt]):
            # 核心逻辑：回仓+周转 <= 下一趟出发
            if veh["available_at"] + TURNAROUND_TIME <= trip["start_min"]:
                veh["trips"].append(trip)
                veh["available_at"] = trip["end_min"]
                assigned = True
                break

        if not assigned:
            # 创建新实体车
            fleet[vt].append({"available_at": trip["end_min"], "trips": [trip]})

    total_cost = 0
    physical_veh_count = 0

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"{'=' * 30} 一车多趟最终排班方案 {'=' * 30}\n")
        f_out.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_out.write(f"输入文件: {input_file}\n")
        # 修改输出文件的说明以反映使用了真实的时变限速
        f_out.write(
            f"运行参数: SPEED=Dynamic(根据路况动态变化), SERVICE={SERVICE_TIME}min, TURNAROUND={TURNAROUND_TIME}min\n\n"
        )
        f_out.write(
            f"{'物理车ID':<10} | {'车型':<8} | {'趟次':<4} | {'发车时间':<8} | {'本趟成本':<10} | {'行驶路径'}\n"
        )
        f_out.write("-" * 100 + "\n")

        for vt in sorted(fleet.keys()):
            for i, veh in enumerate(fleet[vt]):
                physical_veh_count += 1
                # 实体车编号规则：车型简写+序号 (如 EV1-01)
                veh_label = f"{vt.replace('-', '')}-{i + 1:02d}"

                # 扣除被节省的启动费：每辆实体车只收一次400元
                # 算法：本车总成本 = Σ(每趟原始成本) - (总趟数-1)*400
                trip_costs = [t["cost"] for t in veh["trips"]]
                actual_veh_total = (
                    sum(trip_costs) - (len(veh["trips"]) - 1) * FIXED_COST
                )
                total_cost += actual_veh_total

                for t_idx, t in enumerate(veh["trips"]):
                    f_out.write(
                        f"{veh_label:<10} | {vt:<8} | {t_idx + 1:<4} | {t['start_str']:<8} | {t['cost']:<10.2f} | {t['path']}\n"
                    )
                f_out.write("-" * 100 + "\n")

        f_out.write(f"\n[统计结果]\n")
        f_out.write(f"1. 原始任务总数: {len(all_trips)} 趟 \n")
        f_out.write(
            f"2. 最终物理车辆: {physical_veh_count} 辆 (成功削减 {len(all_trips) - physical_veh_count} 辆) \n"
        )
        f_out.write(f"3. 最终优化总成本: {total_cost:,.2f} 元 \n")
        f_out.write(
            f"4. 节省启动费: {(len(all_trips) - physical_veh_count) * FIXED_COST:,.2f} 元 \n"
        )

    # 同时在控制台打印统计结果
    print(f"排班方案已生成: {output_file}")
    print(f"原始任务总数: {len(all_trips)} 趟")
    print(
        f"最终物理车辆: {physical_veh_count} 辆 (成功削减 {len(all_trips) - physical_veh_count} 辆)"
    )
    print(f"最终优化总成本: {total_cost:,.2f} 元")
    print(f"节省启动费: {(len(all_trips) - physical_veh_count) * FIXED_COST:,.2f} 元")


if __name__ == "__main__":
    # seeds = [1523876194]
    # 在我们的测试中，问题一的最佳种子是1523876194
    # 此处保留随机生成以供多次验证，可以直接指定该种子以复现最佳结果
    seeds = random.SystemRandom().sample(range(0, 2**32), 40)
    gb_cost, gb_routes, gb_hist, gb_seed, gb_carb = float("inf"), [], [], None, 0.0

    for s in seeds:
        cost, routes, hist, carbon, sid = execute_universe(s)
        if cost < gb_cost:
            gb_cost, gb_routes, gb_hist, gb_seed, gb_carb = (
                cost,
                routes,
                hist,
                sid,
                carbon,
            )

    mapper = TDVRP_Ultimate_Solver()
    mapper.load_data("总表_5.csv", "路网距离对照表.csv")

    ev_cnt = sum(1 for r in gb_routes if r["veh"]["type"] == 1)
    fuel_cnt = sum(1 for r in gb_routes if r["veh"]["type"] == 0)

    final_str = (
        f"\n{'=' * 60}\n【问题 1 纯净版单趟优化统计】\n"
        f"最佳破纪录种子: {gb_seed}\n"
        f"实际出车总数: {len(gb_routes)} 辆 (EV: {ev_cnt} | Fuel: {fuel_cnt})\n"
        f"总调度成本: {gb_cost:.2f} 元\n"
        f"碳排放总量: {gb_carb:.2f} kg\n"
        f"{'=' * 60}"
    )
    logging.info(final_str)

    report_filename = "Q1_单车单趟明细排班方案.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("=== 问题1 单车单趟明细排班方案 ===\n")
        f.write(f"总成本: {gb_cost:.2f} 元 | 碳排: {gb_carb:.2f} kg\n")
        f.write(
            f"车辆总数: {len(gb_routes)} 辆 (新能源 {ev_cnt} 辆, 燃油车 {fuel_cnt} 辆)\n\n"
        )

        for i, r in enumerate(gb_routes):
            st = r["best_start_time"]
            h, m = int(st // 60), int(st % 60)

            real_path = [mapper.customers[node]["orig_id"] for node in r["path"]]

            clean_path = []
            for node in real_path:
                if not clean_path or clean_path[-1] != node:
                    clean_path.append(node)

            path_str = " -> ".join(map(str, clean_path))

            detail_line = (
                f"▶ 车辆编号: {i + 1:02d} | 车型: [{r['veh']['name']}]\n"
                f"   - 发车时间: {h:02d}:{m:02d}\n"
                f"   - 本趟成本: {r['sub_cost']:.2f} 元\n"
                f"   - 行驶路径: {path_str}\n"
            )
            f.write(detail_line + "\n")

            if i < 5:
                logging.info(detail_line)

        if len(gb_routes) > 5:
            logging.info(f"... 更多车辆明细已保存至 {report_filename} ...\n")

    geo = load_geodata("总表_5.csv")
    generate_mtvrp_schedule(
        "Q1_单车单趟明细排班方案.txt", geo, "Q1一车多趟的最终排班.txt"
    )

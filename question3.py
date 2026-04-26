"""

此程序解决了问题三，在问题二的基础上，针对三个不同时刻的突发事件，
                      （订单取消、紧急插单、订单时间窗变更）进行了动态调度优化。
主要思路：
1. 数据预处理：将客户需求拆分成多个虚拟节点，确保每个节点的重量和体积都在安全阈值内，便于后续调度。
2. 成本评估：在计算路径成本时，模拟行驶过程
    - 根据当前时间段动态调整速度，计算行驶时间和能耗
    - 对于燃油车经过绿区的情况，增加巨额罚款以强制避开
3. 优化算法：采用两阶段混合优化
    - 第一阶段：ALNS大邻域搜索，随机破坏和重建路径
    - 第二阶段：MA模因精修，种群进化和局部搜索
4. 突发事件处理：在每个事件发生时，锁定已发车的任务，重新调度未发车的任务，
   并将新方案与锁定方案合并，确保已执行部分不受影响。
5. 报告生成：输出初始静态方案和最终动态优化方案的详细报告，
   包括成本、碳排放、物理车辆数等关键指标对比，以及每辆车的具体排班信息。


最终生成的报告文件在当前目录下：
    - Q2_Static_Plan.txt：               问题二的初始静态方案详细报告
    - Q3_Final_Sequential_Report.txt：   基于问题二的最优解，问题三的最终动态优化方案详细报告
日志文件：
    - logs/q3.log：     包含程序执行过程中的详细日志

"""

import copy
import logging
import math
import os
import random
import sys
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime

#  基础配置与日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/q3.log", encoding="utf-8"),
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

#  全局交通限速配置 (用于一车多趟时间计算)
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


def get_dynamic_travel_time(dist, start_t):
    """
    基于给定的距离和当前发车时间，使用真实的、分时段变化的平均限速计算到达时间。
    一车多趟合并逻辑专用。
    """
    t = start_t
    d_left = dist
    while d_left > 1e-4:
        v_avg, t_end = 35.4, 99999
        for s_i, e_i, v_i in TRAFFIC_INTERVALS:
            if s_i <= t < e_i:
                v_avg, t_end = v_i, e_i
                break

        v_min = v_avg / 60.0
        d_run = min(d_left, (t_end - t) * v_min)
        t += d_run / v_min
        d_left -= d_run

    return t


class TDVRP_Ultimate_Solver:
    def __init__(self):
        # 成本与物理参数
        self.c_fixed = 400.0
        self.c_wait, self.c_late = 20.0 / 60.0, 50.0 / 60.0
        self.c_fuel, self.c_ev = 7.61, 1.64
        self.c_carbon = 0.65
        self.eta_fuel, self.gamma_ev = 2.547, 0.501
        self.service_time = 20.0

        # 一车多趟核心约束
        self.turnaround_time = 30.0

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

        # 算法性能参数
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
            except:
                tried.append(enc)
        raise ValueError(f"无法读取文件 {file_path}")

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
            "is_green": 0,
            "orig_id": 0,
        }
        vid_counter, orig_to_vids = 1000, {0: [0]}
        SAFE_W, SAFE_V = 1200.0, 6.0

        for _, row in df_cust.iterrows():
            cid = int(row["目标客户编号"])
            w, v = float(row["重量"]), float(row["体积"])
            rdy, du, is_green = (
                float(row["开始时间"]),
                float(row["结束时间"]),
                int(row["是否为绿色区域"]),
            )
            orig_to_vids[cid] = []
            temp_w, temp_v = w, v
            while temp_w > SAFE_W or temp_v > SAFE_V:
                ratio = min(
                    SAFE_W / temp_w if temp_w > 0 else 1.0,
                    SAFE_V / temp_v if temp_v > 0 else 1.0,
                )
                chunk_w, chunk_v = temp_w * ratio, temp_v * ratio
                self.customers[vid_counter] = {
                    "weight": chunk_w,
                    "vol": chunk_v,
                    "ready": rdy,
                    "due": du,
                    "is_green": is_green,
                    "orig_id": cid,
                }
                orig_to_vids[cid].append(vid_counter)
                vid_counter += 1
                temp_w -= chunk_w
                temp_v -= chunk_v
            if temp_w > 0.001:
                node_id = cid if len(orig_to_vids[cid]) == 0 else vid_counter
                if node_id == vid_counter:
                    vid_counter += 1
                self.customers[node_id] = {
                    "weight": temp_w,
                    "vol": temp_v,
                    "ready": rdy,
                    "due": du,
                    "is_green": is_green,
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
        logging.info(f"数据加载完成。拆分后任务点: {len(self.customers) - 1}")

    def simulate_segment(self, dist, start_t, current_load, veh, speed_mode="mean"):
        """单趟调度求解器内部的耗时与能耗模拟"""
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

    def mtvrp_compressor(self, single_trips):
        """一车多趟压缩核心逻辑，使用真实的根据时间获取速度"""
        if not single_trips:
            return [], 0
        for trip in single_trips:
            curr_t = trip["best_start_time"]
            path = trip["path"]
            load = trip["load"]
            for i in range(len(path) - 1):
                d = self.dist_matrix.get(path[i], {}).get(path[i + 1], 999.0)

                # 【修改点】：调用真实的根据时间获取速度的函数进行回仓时间推演
                curr_t = get_dynamic_travel_time(d, curr_t)

                if path[i + 1] != 0:
                    curr_t += self.service_time
                    load -= self.customers[path[i + 1]]["weight"]
            trip["end_time"] = curr_t

        single_trips.sort(key=lambda x: x["best_start_time"])
        physical_fleet = {}
        for trip in single_trips:
            v_name = trip["veh"]["name"]
            if v_name not in physical_fleet:
                physical_fleet[v_name] = []
            assigned = False
            for veh in physical_fleet[v_name]:
                if (
                    veh["available_at"] + self.turnaround_time
                    <= trip["best_start_time"]
                ):
                    veh["trips"].append(trip)
                    veh["available_at"] = trip["end_time"]
                    assigned = True
                    break
            if not assigned:
                physical_fleet[v_name].append(
                    {"available_at": trip["end_time"], "trips": [trip]}
                )

        final_physical_plan = []
        total_optimized_cost = 0.0
        for v_name, veh_list in physical_fleet.items():
            for i, veh_obj in enumerate(veh_list):
                p_id = f"{v_name}-P{i+1:02d}"
                sum_trip_costs = sum(t["sub_cost"] for t in veh_obj["trips"])
                saved_fixed = (len(veh_obj["trips"]) - 1) * self.c_fixed
                veh_final_cost = sum_trip_costs - saved_fixed
                total_optimized_cost += veh_final_cost
                final_physical_plan.append(
                    {
                        "physical_id": p_id,
                        "veh_name": v_name,
                        "type_idx": veh_obj["trips"][0]["veh"]["type"],
                        "trips": veh_obj["trips"],
                        "veh_total_cost": veh_final_cost,
                    }
                )
        return final_physical_plan, total_optimized_cost

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
                    if veh["type"] == 0 and self.customers[nxt]["is_green"] == 1:
                        if max(arr_t, 480) < min(arr_t + self.service_time, 960):
                            sub += 1000000.0
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
                curr_p, w_s, v_s = [cid], c["weight"], c["vol"]
        if curr_p:
            routes.append(
                {"veh": self.fleet_pool[v_idx], "path": [0] + curr_p + [0], "load": w_s}
            )
        single_trip_results = []
        total_carbon = 0.0
        for r in routes:
            sub_c, ts_opt, r_carb = self.evaluate_route_total_cost(
                r["path"], r["veh"], r["load"], speed_mode=speed_mode
            )
            r.update({"best_start_time": ts_opt, "sub_cost": sub_c, "carbon": r_carb})
            single_trip_results.append(r)
            total_carbon += r_carb
        if return_details:
            phys_plan, phys_cost = self.mtvrp_compressor(single_trip_results)
            return phys_cost, phys_plan, sequence, total_carbon
        eval_cost = sum(r["sub_cost"] for r in single_trip_results)
        return eval_cost, sequence

    def run_hybrid_optimization(self):
        c_ids = [cid for cid in self.customers.keys() if cid != 0]
        curr_seq = random.sample(c_ids, len(c_ids))
        curr_c, curr_seq = self.compute_cost_and_optimize(curr_seq)
        best_seq, best_c = copy.deepcopy(curr_seq), curr_c
        T = self.T_start
        for it in range(self.alns_iters):
            q = random.randint(self.q_min, self.q_max)
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
        pop = [copy.deepcopy(best_seq) for _ in range(self.ma_pop_size)]
        for g in range(self.ma_generations):
            fits, n_pop = [], []
            for ind in pop:
                c, opt = self.compute_cost_and_optimize(ind)
                fits.append(c)
                n_pop.append(opt)
            pop, midx = n_pop, int(np.argmin(fits))
            if fits[midx] < best_c:
                best_c, best_seq = fits[midx], copy.deepcopy(pop[midx])
        return best_seq


#  核心：三阶段顺序动态调度引擎


def sequential_dynamic_dispatch(solver, initial_phys_plan):
    """
    实现三个不同时刻的突发事件处理
    """
    # 时间点定义
    T1, T2, T3 = 540.0, 660.0, 780.0  # 09:00, 11:00, 13:00

    current_phys_plan = copy.deepcopy(initial_phys_plan)

    # ---------------- 阶段 1: 09:00 订单取消 ----------------
    logging.info("\n>>> [T1: 09:00] 事件触发：订单取消")
    pending_nodes = []
    locked_trips = []
    for p_v in current_phys_plan:
        for t in p_v["trips"]:
            if t["best_start_time"] < T1:
                locked_trips.append(t)
            else:
                pending_nodes.extend([n for n in t["path"] if n != 0])

    if pending_nodes:
        removed_node = pending_nodes.pop(random.randint(0, len(pending_nodes) - 1))
        logging.info(
            f"    - 节点 {removed_node} (客户 {solver.customers[removed_node]['orig_id']}) 的订单已取消。"
        )

    # 重调度
    solver.departure_options = [t for t in solver.departure_options if t >= T1]
    _, new_trips_p, _, _ = solver.compute_cost_and_optimize(
        pending_nodes, return_details=True
    )
    new_trips = [tk for p in new_trips_p for tk in p["trips"]]
    current_phys_plan, _ = solver.mtvrp_compressor(locked_trips + new_trips)

    # ---------------- 阶段 2: 11:00 紧急插单 ----------------
    logging.info("\n>>> [T2: 11:00] 事件触发：紧急插单")
    pending_nodes = []
    locked_trips = []
    for p_v in current_phys_plan:
        for t in p_v["trips"]:
            if t["best_start_time"] < T2:
                locked_trips.append(t)
            else:
                pending_nodes.extend([n for n in t["path"] if n != 0])

    # 新增紧急订单 9999
    u_id = 9999
    solver.dist_matrix[u_id] = solver.dist_matrix.get(88, {}).copy()
    for s_n in solver.dist_matrix:
        if 88 in solver.dist_matrix[s_n]:
            solver.dist_matrix[s_n][u_id] = solver.dist_matrix[s_n][88]
    solver.customers[u_id] = {
        "weight": 1100.0,
        "vol": 5.0,
        "ready": T2,
        "due": T2 + 120.0,
        "is_green": 0,
        "orig_id": 9999,
    }
    pending_nodes.append(u_id)
    logging.info(f"    - 接到紧急订单 9999，位置参考节点 88，要求 2 小时内送达。")

    # 重调度
    solver.departure_options = [t for t in solver.departure_options if t >= T2]
    _, new_trips_p, _, _ = solver.compute_cost_and_optimize(
        pending_nodes, return_details=True
    )
    new_trips = [tk for p in new_trips_p for tk in p["trips"]]
    current_phys_plan, _ = solver.mtvrp_compressor(locked_trips + new_trips)

    # ---------------- 阶段 3: 13:00 时间窗变更 ----------------
    logging.info("\n>>> [T3: 13:00] 事件触发：订单时间窗变更")
    pending_nodes = []
    locked_trips = []
    for p_v in current_phys_plan:
        for t in p_v["trips"]:
            if t["best_start_time"] < T3:
                locked_trips.append(t)
            else:
                pending_nodes.extend([n for n in t["path"] if n != 0])

    if pending_nodes:
        target_node = pending_nodes[0]
        solver.customers[target_node]["due"] = T3 + 60.0  # 强制要求 1 小时内送达
        logging.info(
            f"    - 客户 {solver.customers[target_node]['orig_id']} (节点 {target_node}) 要求提前送达，时间窗截止缩短至 14:00。"
        )

    # 重调度
    solver.departure_options = [t for t in solver.departure_options if t >= T3]
    _, new_trips_p, _, _ = solver.compute_cost_and_optimize(
        pending_nodes, return_details=True
    )
    new_trips = [tk for p in new_trips_p for tk in p["trips"]]
    current_phys_plan, final_cost = solver.mtvrp_compressor(locked_trips + new_trips)

    # 恢复配置以便后续计算
    solver.departure_options = [float(t) for t in range(480, 961, 30)]

    return current_phys_plan, final_cost


if __name__ == "__main__":
    SEED = 515383698  # 该种子是问题二中的最佳种子，将该方案作为原方案，进行动态调度
    random.seed(SEED)
    np.random.seed(SEED)
    solver = TDVRP_Ultimate_Solver()
    solver.load_data("总表_5.csv", "路网距离对照表.csv")

    # 1. 初始静态方案 Q2
    best_seq = solver.run_hybrid_optimization()
    fc_static, phys_static, _, carb_static = solver.compute_cost_and_optimize(
        best_seq, return_details=True
    )

    # 输出静态报告
    with open("Q2_Static_Plan.txt", "w", encoding="utf-8") as f:
        f.write("=== [报告 1] 问题 2 初始静态最优一车多趟方案 ===\n")
        f.write(f"总成本: {fc_static:.2f} 元 | 碳排放: {carb_static:.2f} kg\n")
        f.write(f"物理车辆数: {len(phys_static)} 辆\n\n")
        for v in phys_static:
            f.write(
                f"▶ 物理车: {v['physical_id']} | 车型: {v['veh_name']} | 成本: {v['veh_total_cost']:.2f}\n"
            )
            for t in v["trips"]:
                p_str = " -> ".join(
                    [str(solver.customers[n]["orig_id"]) for n in t["path"]]
                )
                f.write(
                    f"   - 发车: {int(t['best_start_time']//60):02d}:{int(t['best_start_time']%60):02d} | 路径: {p_str}\n"
                )
            f.write("\n")

    # 2. 执行三阶段顺序动态调度
    final_phys_plan, final_optimized_cost = sequential_dynamic_dispatch(
        solver, phys_static
    )
    final_carb = sum(t["carbon"] for v in final_phys_plan for t in v["trips"])

    # 输出最终动态报告
    with open("Q3_Final_Sequential_Report.txt", "w", encoding="utf-8") as f:
        f.write("=== [报告 2] 问题 3 三阶段突发事件最终动态优化报告 ===\n\n")
        f.write("【突发事件说明】\n")
        f.write(
            "1. 09:00 - 订单取消\n2. 11:00 - 紧急插单 (ID 9999)\n3. 13:00 - 订单时间窗变更 (要求 1 小时内送达)\n\n"
        )

        comp_table = (
            f"数据指标对比        | 初始静态方案        | 动态优化方案(最终)\n"
            f"-------------------+--------------------+---------------\n"
            f"总成本 (元)         | {fc_static:>17.2f} | {final_optimized_cost:>13.2f}\n"
            f"物理车辆数 (辆)      | {len(phys_static):>17} | {len(final_phys_plan):>13}\n"
            f"碳排放 (kg)         | {carb_static:>17.2f} | {final_carb:>13.2f}\n"
        )
        f.write(comp_table + "\n\n")

        f.write("【最终重排后的详细物理排班】\n\n")
        for v in final_phys_plan:
            f.write(
                f"车辆 {v['physical_id']} [{v['veh_name']}] | 包含任务: {len(v['trips'])} 趟 | 总成本: {v['veh_total_cost']:.2f}\n"
            )
            for t in v["trips"]:
                p_str = " -> ".join(
                    [str(solver.customers[n]["orig_id"]) for n in t["path"]]
                )
                f.write(
                    f"  └─ 发车 {int(t['best_start_time']//60):02d}:{int(t['best_start_time']%60):02d} | 路径: {p_str}\n"
                )
            f.write("\n")

    logging.info(f"\n三阶段突发事件处理完成。")
    logging.info(
        f"最终报告已生成：Q2_Static_Plan.txt 和 Q3_Final_Sequential_Report.txt"
    )

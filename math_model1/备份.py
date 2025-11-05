from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from numpy.fft import rfft, rfftfreq

"""
实验1：综合误差诊断与最优估计

整体流程（确定性，一键运行）：
1. 从 BASE_DIR/data.csv 读取测量数据矩阵 (20 组 × 每组100次测量)。
2. 计算基础统计量并绘制原始测量曲线，导出 fig1.png。
3. 进行粗大误差检测与可视化替代，并绘制对比图 fig2.png。
4. 对粗大误差剔除后的数据进行系统误差诊断（恒值偏差 / 线性趋势 / 周期项），并校正，
   绘制校正前后对比图 fig3.png。
5. 基于系统误差校正后的数据，判断等精度 / 不等精度，计算全局最优估计值，
   并绘制均值与标准差分布图 fig4.png。
6. 生成 results.txt，总结实验结果（m1、s1、p、t1、t2、t3、a、整体分析文字）。

所有输出文件 (fig1.png ~ fig4.png 以及 results.txt)
都会保存到脚本所在目录 BASE_DIR。
"""

# =========================================================
# Matplotlib 中文显示设置，避免中文字符变成乱码或方块
# =========================================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 让坐标轴上的负号正常显示

# =========================================================
# 固定的实验工作目录（即当前脚本所在的位置）
# 数据输入与输出的所有文件都放在这个目录下
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 若路径不存在（一般不会），则创建之
os.makedirs(BASE_DIR, exist_ok=True)
print("实验工作目录（读写路径）:", BASE_DIR)

# =========================================================
# 全局参数（这些参数会在报告中解释其选择理由）
# =========================================================
np.random.seed(12345)            # 固定随机种子，保证复现性
ALPHA_Z = 0.01                   # 用于检测“恒值系统偏差”的显著性水平（双侧检验）
ALPHA_SLOPE = 0.01               # 用于检测“线性趋势”的显著性水平
PERIOD_ENERGY_RATIO = 0.2        # 周期性误差判定的能量占比阈值（主频能量 / 总能量）
TREND_GROUPS_FOR_PLOT = (0, 4, 9, 19)  # 报告要求重点展示的4组：第1、5、10、20组（索引从0开始）
THRESH_EQUAL_PRECISION_RATIO = 2.0     # 等精度判定阈值：max(s)/min(s) < 2 视为近似等精度
SIGMA_THRESHOLD = 3.0            # 粗大误差判据：3σ原则
K_LIMIT = 3.0                    # 极限误差按照 3σ 形式给出
MAX_OUTLIER_ITER = 10            # 粗大误差迭代识别最多迭代次数


# =========================================================
# 数据结构（用于打包函数返回结果，方便主流程调用）
# =========================================================
@dataclass
class OutlierResult:
    """
    用于描述某一组测量数据在粗大误差处理后的结果。
    mask_good: 布尔数组，True 表示该测量点被保留，False 表示被判定为粗大误差的点。
    cleaned:   只包含“保留点”的数据（长度会比原始少，因为粗大误差点被删除）。此数组只用于统计分析。
    display:   用于绘图展示的版本。粗大误差点不会直接删除，而是用邻域中位数等方式替换，
               以保证曲线连贯，便于在图中直观看到“修正后”的趋势。
    """
    mask_good: np.ndarray
    cleaned: np.ndarray
    display: np.ndarray


@dataclass
class SystemCorrectionResult:
    """
    用于描述系统误差诊断与校正的结果。
    corrected:                 系统误差校正后的数据（20×N 数组，N为测量长度，通常100）。
    constant_bias_groups:      存放存在恒值系统偏差（整体偏移）的组号列表（1起始）。
    linear_trend_groups:       存放存在明显线性趋势（随测量序号上升或下降）的组号列表（1起始）。
    periodic_groups:           存放存在明显周期性误差（振荡成分）的组号列表（1起始）。
    before_after_pairs:        每一组对应一个 (before, after) 元组，
                               before 为校正前序列，after 为校正后序列。
                               主要用于后续绘制 fig3.png。
    """
    corrected: np.ndarray
    constant_bias_groups: List[int]
    linear_trend_groups: List[int]
    periodic_groups: List[int]
    before_after_pairs: List[Tuple[np.ndarray, np.ndarray]]


class DataShapeError(ValueError):
    """
    当 data.csv 的形状不符合“至少20行×100列”要求时，抛出该异常。
    """


# =========================================================
# 数据读取函数
# =========================================================
def load_data() -> np.ndarray:
    """
    读取测量数据矩阵，要求文件位于 BASE_DIR/data.csv。

    数据含义：
      - 行方向：20组独立测量（例如20个实验对象或20次独立试验）
      - 列方向：每组内部的100次重复观测

    约束：
      - data.csv 至少为 20 行 × 100 列的数字矩阵
      - 如果 data.csv 更大（比如多余的列或行），本函数会自动裁剪到前20行、前100列

    返回：
      data (20×100 的 numpy 数组，float类型)
    """
    csv_path = os.path.join(BASE_DIR, "data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[致命错误] 未找到测量数据文件 data.csv ：{csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", dtype=float)

    if data.ndim != 2:
        raise DataShapeError("data.csv 必须是二维数值数组（形如20行×100列）")
    if data.shape[0] < 20 or data.shape[1] < 100:
        raise DataShapeError(
            f"data.csv 尺寸为 {data.shape}，不满足至少20行×100列的要求"
        )

    # 只取前 20 行、前 100 列，避免“数据比要求更多”导致后续维度不一致
    data = data[:20, :100]
    return data


# =========================================================
# 基础统计：计算每一组的算术平均值和标准差
# =========================================================
def basic_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入：
        data: shape = (20, N)

    输出：
        means:  每组的算术平均值 (长度20)
        stds:   每组的样本标准差 (ddof=1，无偏估计，长度20)

    对应实验要求的 m1、s1。
    """
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1, ddof=1)
    return means, stds


# =========================================================
# 粗大误差检测（迭代 3σ 法）
# =========================================================
def detect_and_remove_outliers(row: np.ndarray) -> OutlierResult:
    """
    对单独一组测量数据（长度100左右）执行粗大误差检测与替代。

    方法：迭代 3σ 判据
    - 初始假设：该组数据的“正常误差”服从近似高斯分布。
    - 步骤：
        1. 用当前保留点估计均值 mu 和标准差 sd。
        2. 找到 |x - mu| > 3σ 的点，视为粗大误差。
        3. 剔除这些点，重新估计 mu 和 sd。
        4. 反复迭代，直到没有新的粗大误差或达到迭代上限。

    输出：
      mask_good: True/False，表示每个点最终是否保留。
      cleaned:   仅包含保留点的数组（后续统计可用）。
      display:   仅用于绘图对比。粗大误差点不直接“删掉变空”，
                 而是用邻域中位数或全局中位数进行替换，保证曲线连续。

    说明：
      - 这满足实验中对“粗大误差剔除”和“可视化替代方案”的要求。
      - p（含粗大误差的组号）由上层逻辑根据 mask_good 是否出现 False 来记录。
    """
    if row.size == 0:
        return OutlierResult(np.zeros(0, dtype=bool), row.copy(), row.copy())

    mask_good = np.ones_like(row, dtype=bool)

    for _ in range(MAX_OUTLIER_ITER):
        candidate = row[mask_good]
        if candidate.size <= 1:
            # 如果剩余点太少，没法估计标准差，就停止
            break
        mu = candidate.mean()
        sd = candidate.std(ddof=1)
        if sd == 0:
            # 如果标准差为0，说明剩余点几乎相同，没有必要继续迭代
            break

        z = np.abs((row - mu) / sd)
        new_bad = (z > SIGMA_THRESHOLD) & mask_good
        if not np.any(new_bad):
            # 没有新增的异常点，停止迭代
            break
        mask_good[new_bad] = False

    cleaned = row[mask_good]

    # display：把判定为粗大误差的点，用邻域中位数或全局中位数替代，方便画出“剔除后曲线”
    display = row.copy()
    bad_idx = np.where(~mask_good)[0]
    for bi in bad_idx:
        L = max(0, bi - 2)
        R = min(len(row), bi + 3)
        neigh = [j for j in range(L, R) if mask_good[j]]
        if len(neigh) > 0:
            display[bi] = np.median(row[neigh])
        else:
            # 如果附近全是坏点，就退化为使用“全局保留点”的中位数
            display[bi] = np.median(cleaned) if cleaned.size > 0 else row[bi]

    return OutlierResult(mask_good, cleaned, display)


# =========================================================
# 系统误差判定工具函数
# =========================================================
def z_test_mean(
    group_mean: float,
    group_std: float,
    n: int,
    global_mean: float,
    alpha: float = ALPHA_Z,
) -> bool:
    """
    用 z 检验判断某一组是否存在“恒值系统偏差”（整体偏移）。

    统计思想：
      原假设 H0：该组均值 == 全局加权均值
      统计量： z = (m_i - m_global) / (s_i / sqrt(n))
      双侧检验：若 |z| > z_{1-alpha/2} 则拒绝 H0

    返回：
      True  -> 认为该组存在显著的恒值偏差（常偏差）
      False -> 未检测到显著恒值偏差
    """
    if n <= 1:
        return False
    safe_std = max(group_std, 1e-12)  # 防止除零
    z_val = (group_mean - global_mean) / (safe_std / math.sqrt(n))
    if np.isnan(z_val):
        return False
    z_crit = stats.norm.ppf(1 - alpha / 2)  # 双侧检验临界值
    return abs(z_val) > z_crit


def linear_trend_test(y: np.ndarray, alpha: float = ALPHA_SLOPE):
    """
    用线性回归判断是否存在“线性系统误差”（随测量序号逐步上升或下降）。

    步骤：
      1. 拟合 y(k) = a + b*k
      2. 使用回归斜率 b 的显著性检验（p值）来判断是否存在趋势
         - 若 p_value < alpha，视为存在显著线性漂移
      3. 若存在趋势，则生成去趋势序列：y - (a + b*k)

    返回：
      has_trend      : 是否检测到显著线性趋势
      slope, intercept: 回归得到的斜率/截距
      detrended      : 去除线性趋势后的序列
    """
    n = y.size
    if n <= 2:
        # 数据点太少，无法稳健判断趋势
        return False, 0.0, 0.0, y.copy()
    x = np.arange(1, n + 1)
    slope, intercept, _, p_value, _ = stats.linregress(x, y)
    has_trend = bool(p_value < alpha)
    if has_trend:
        detrended = y - (intercept + slope * x)
    else:
        detrended = y.copy()
    return has_trend, slope, intercept, detrended


def detect_periodic(y: np.ndarray, ratio_thresh: float = PERIOD_ENERGY_RATIO):
    """
    检测并校正“周期性系统误差”。

    思路（对应实验书中“周期图检验 / 正弦拟合残差检验”的思路）：
      1. 对序列去均值后做快速傅里叶变换 (FFT)；
      2. 计算频谱能量，寻找最强的非零主频分量；
      3. 如果该主频分量的能量占总能量比例超过阈值 ratio_thresh（例如20%），
         说明序列中存在显著的周期性成分（例如机械振动、电源纹波等）。
      4. 针对该主频 f，构造正弦+余弦+常数项的最小二乘拟合：
            y(k) ≈ A*sin(2π f k) + B*cos(2π f k) + C
         并将该拟合项从原序列中扣除，得到校正后的序列。

    返回：
      has_period      : 是否存在显著周期成分
      f               : 检测到的主频
      fitted          : 拟合得到的周期项（A*sin + B*cos + C）
      corrected       : 去除了主要周期项后的序列
    """
    n = y.size
    if n == 0:
        return False, 0.0, np.zeros_like(y), y.copy()

    # 1. 去均值后做FFT
    demeaned = y - y.mean()
    spectrum = rfft(demeaned)
    power = np.abs(spectrum) ** 2  # 频谱能量

    # 如果只有DC分量，没法判断周期
    if power.size <= 1:
        return False, 0.0, np.zeros_like(y), y.copy()

    # 2. 寻找最强的非零频率分量（跳过直流分量0频）
    idx_nonzero = np.arange(1, power.size)
    dom_idx = idx_nonzero[np.argmax(power[1:])]
    dom_power = power[dom_idx]
    total_power = power.sum()
    ratio = dom_power / total_power if total_power > 0 else 0.0

    has_period = ratio > ratio_thresh

    # 取得对应的物理“频率索引”
    freq_arr = rfftfreq(n, d=1.0)
    f = freq_arr[dom_idx]

    # 3. 如果确实存在显著的周期误差，则拟合正弦+余弦项并扣除
    if has_period and f != 0:
        k = np.arange(n)
        # 设计矩阵 [sin(2πfk), cos(2πfk), 1]
        design = np.column_stack([
            np.sin(2 * np.pi * f * k),
            np.cos(2 * np.pi * f * k),
            np.ones(n)
        ])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef              # 周期拟合项
        corrected = y - fitted              # 去周期后的信号
    else:
        fitted = np.zeros_like(y)
        corrected = y.copy()

    return has_period, f, fitted, corrected


# =========================================================
# 系统误差诊断与校正的主函数
# =========================================================
def apply_system_corrections(data: np.ndarray) -> SystemCorrectionResult:
    """
    输入：
        data: 形状为 (20, N) 的数组，
              这里的 data 是“粗大误差剔除后、但用邻域中位数补点的版本”，
              即 display 序列。长度依然是 N=100，便于逐组比较。

    整个流程：
      1. 计算每组的均值和标准差，并以加权方式求出全局均值
         （权重=1/σ²，标准差小的组权重更大）。
      2. 对每一组依次执行三类系统误差检测与校正：
         (a) 恒值偏差（整体偏移）：用 z 检验判定，
             若显著偏离全局均值，则整体平移回全局均值。
         (b) 线性趋势（随测量序号漂移）：用线性回归斜率的显著性检验，
             若显著，减去该线性趋势。
         (c) 周期项（周期性干扰）：用主频能量占比阈值进行判断，
             若存在，拟合正弦/余弦项并扣除。
      3. 记录每类系统误差出现的组号（1起始编号），
         分别输出为 constant_bias_groups (t1)、linear_trend_groups (t2)、
         periodic_groups (t3)。
      4. 保存每一组校正前/后的曲线，用于 fig3.png 绘图。

    返回：
      SystemCorrectionResult 对象，包含校正后的数据矩阵、t1/t2/t3组号列表、
      以及用于绘图的前后对照数据对。
    """
    group_means, group_stds = basic_stats(data)

    # 计算全局加权均值（标准差越小的组，权重越大）
    weights = 1.0 / (np.maximum(group_stds, 1e-12) ** 2)
    global_mean = np.average(group_means, weights=weights)

    corrected = np.zeros_like(data)
    t1_list: List[int] = []  # 恒值系统偏差的组号
    t2_list: List[int] = []  # 线性趋势系统误差的组号
    t3_list: List[int] = []  # 周期性系统误差的组号
    before_after_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for i, series in enumerate(data):
        n = series.size

        # ---------- (a) 恒值系统偏差：整体偏高或偏低 ----------
        has_bias = z_test_mean(group_means[i], group_stds[i], n, global_mean)
        bias_corrected = series.copy()
        if has_bias:
            offset = group_means[i] - global_mean
            bias_corrected = series - offset  # 平移校正
            t1_list.append(i + 1)  # 组号按 1~20 记录

        # ---------- (b) 线性趋势：是否随测量序号持续漂移 ----------
        has_trend, slope, intercept, detrended = linear_trend_test(
            bias_corrected
        )
        trend_corrected = bias_corrected.copy()
        if has_trend:
            trend_corrected = detrended
            t2_list.append(i + 1)

        # ---------- (c) 周期性干扰：是否存在强主频 ----------
        has_period, dom_f, fitted, periodic_removed = detect_periodic(
            trend_corrected
        )
        final_series = trend_corrected.copy()
        if has_period:
            final_series = periodic_removed
            t3_list.append(i + 1)

        corrected[i] = final_series
        before_after_pairs.append((series.copy(), final_series))

    return SystemCorrectionResult(
        corrected=corrected,
        constant_bias_groups=t1_list,
        linear_trend_groups=t2_list,
        periodic_groups=t3_list,
        before_after_pairs=before_after_pairs,
    )


# =========================================================
# 加权平均（用于不等精度情形）
# =========================================================
def weighted_average(means: np.ndarray, stds: np.ndarray) -> Tuple[float, float]:
    """
    计算不等精度条件下的加权平均值及其极限误差。

    原理：
      - 假设各组观测值（即每组的平均值）独立，标准差已知/可估计。
      - 令权重 w_i = 1 / σ_i^2。
      - 加权平均值 a = Σ(w_i * m_i) / Σ(w_i)。
      - 加权平均的不确定度 σ_a = sqrt(1 / Σ w_i)。
      - 极限误差 = 3 * σ_a （对应“3σ极限误差”的表述）。

    返回：
      (a, limit_err) = (加权平均值, 极限误差)
    """
    safe_stds = np.maximum(stds, 1e-12)  # 防止除零
    w = 1.0 / (safe_stds ** 2)
    a = float(np.sum(w * means) / np.sum(w))
    sigma_a = math.sqrt(1.0 / np.sum(w))
    limit_err = K_LIMIT * sigma_a
    return a, float(limit_err)


# =========================================================
# 文本格式化工具，用于写入 results.txt
# =========================================================
def fmt_float_list(values: Sequence[float]) -> str:
    """
    将一串浮点数格式化为“保留两位小数并用逗号分隔”的形式。
    例如：[1.234, 5.678] -> "1.23,5.68"
    """
    return ",".join(f"{float(v):.2f}" for v in values)

def fmt_int_list(values: Iterable[int]) -> str:
    """
    将一串整数格式化为用逗号分隔的字符串。
    例如：[1, 5, 7] -> "1,5,7"
    """
    return ",".join(str(int(v)) for v in values)


# =========================================================
# 写入 results.txt（实验要求的8行汇总结果）
# =========================================================
def save_results(
    raw_means: np.ndarray,
    raw_stds: np.ndarray,
    outlier_groups: Sequence[int],
    corr_result: SystemCorrectionResult,
    final_means: np.ndarray,
    final_stds: np.ndarray,
    optimal_mean: float,
    limit_error: float,
    equal_precision: bool,
) -> None:
    """
    生成实验要求的 results.txt 文件，包含8行：

    第1行：m1（20个数）                    -> 原始数据每组均值
    第2行：s1（20个数）                    -> 原始数据每组标准差
    第3行：p（出现粗大误差的组号）         -> outlier_groups
    第4行：t1（恒值系统误差的组号）        -> corr_result.constant_bias_groups
    第5行：t2（线性系统误差的组号）        -> corr_result.linear_trend_groups
    第6行：t3（周期性系统误差的组号）      -> corr_result.periodic_groups
    第7行：a（最终测量结果 ± 极限误差）    -> optimal_mean ± limit_error
    第8行：文字性总结                      -> 对整体诊断和可靠性的解释

    注意：数值使用逗号分隔，保留两位小数，符合实验书的输出规范。
    """

    summary_lines = [
        f"粗大误差检测采用迭代3σ法(阈值={SIGMA_THRESHOLD:.1f}σ)，共识别到 {len(outlier_groups)} 组存在疑似粗大误差点。",
        (
            "恒值系统偏差通过 z 检验(alpha={:.2f}) 判断组均值是否显著偏离全局加权均值；"
            "线性趋势通过回归斜率显著性检验(alpha={:.2f})判断是否存在随测量序号的漂移；"
            "周期性误差由主频能量占比阈值{:.2f}判定，并拟合正弦项后扣除。"
        ).format(ALPHA_Z, ALPHA_SLOPE, PERIOD_ENERGY_RATIO),
        (
            "系统误差校正后，对20组数据计算新的均值和标准差，并判断其是否可视为等精度"
            "(标准差最大值/最小值阈值为 {:.1f})。据此选择等权平均或加权平均，"
            "最终给出全局测量结果及其3σ极限误差：{:.2f}±{:.2f}。"
        ).format(THRESH_EQUAL_PRECISION_RATIO, optimal_mean, limit_error),
        "在后续报告中将进一步讨论：阈值选择的合理性、对结果的敏感性、是否可能过拟合周期项、以及是否存在由少数高精度组主导整体结果的风险。",
    ]

    lines = [
        fmt_float_list(raw_means),                                 # 第1行 m1
        fmt_float_list(raw_stds),                                  # 第2行 s1
        fmt_int_list(outlier_groups),                              # 第3行 p
        fmt_int_list(corr_result.constant_bias_groups),            # 第4行 t1
        fmt_int_list(corr_result.linear_trend_groups),             # 第5行 t2
        fmt_int_list(corr_result.periodic_groups),                 # 第6行 t3
        f"{optimal_mean:.2f}±{limit_error:.2f}",                   # 第7行 a
        " ".join(summary_lines),                                   # 第8行 文字总结
    ]

    results_path = os.path.join(BASE_DIR, "results.txt")
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# =========================================================
# 绘图函数（会输出 fig1.png ~ fig4.png）
# 图内标签/标题全部改为中文，字体已在前面统一设置
# =========================================================
def plot_raw_groups(raw: np.ndarray, indices: Sequence[int], filename: str) -> None:
    """
    绘制原始测量曲线（第1、5、10、20组），对应 fig1.png。
    """
    plt.figure()
    n = raw.shape[1]
    x_axis = np.arange(1, n + 1)
    for idx in indices:
        plt.plot(x_axis, raw[idx], label=f"第{idx + 1}组 原始数据")
    plt.xlabel("测量序号")
    plt.ylabel("测量值")
    plt.title("典型4组原始测量曲线（第1/5/10/20组）")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_outlier_comparison(before: np.ndarray,
                            after: np.ndarray,
                            indices: Sequence[int],
                            filename: str) -> None:
    """
    绘制粗大误差剔除前/后的对比曲线（采用邻域中位数替代的可视化版本），对应 fig2.png。
    """
    plt.figure()
    n = before.shape[1]
    x_axis = np.arange(1, n + 1)
    for idx in indices:
        plt.plot(
            x_axis,
            before[idx],
            alpha=0.7,
            label=f"第{idx + 1}组 剔除前"
        )
        plt.plot(
            x_axis,
            after[idx],
            linestyle="--",
            label=f"第{idx + 1}组 剔除后(可视化替代)"
        )
    plt.xlabel("测量序号")
    plt.ylabel("测量值")
    plt.title("粗大误差剔除前后对比（局部异常点以中位数替代显示）")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_system_correction(pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
                           indices: Sequence[int],
                           filename: str) -> None:
    """
    绘制系统误差校正前/后的对比曲线（恒值偏差平移、去趋势、去周期），对应 fig3.png。
    """
    plt.figure()
    if not pairs:
        n = 0
        x_axis = np.arange(1, n + 1)
    else:
        n = pairs[0][0].size
        x_axis = np.arange(1, n + 1)

    for idx in indices:
        before, after = pairs[idx]
        plt.plot(
            x_axis,
            before,
            alpha=0.7,
            label=f"第{idx + 1}组 校正前"
        )
        plt.plot(
            x_axis,
            after,
            linestyle="--",
            label=f"第{idx + 1}组 校正后"
        )
    plt.xlabel("测量序号")
    plt.ylabel("测量值")
    plt.title("系统误差校正前后对比（恒值偏差/线性漂移/周期项）")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_means_stds(means: np.ndarray,
                    stds: np.ndarray,
                    filename: str,
                    title: str) -> None:
    """
    绘制校正后20组的均值与标准差（双y轴显示），对应 fig4.png。
    """
    fig, ax1 = plt.subplots()
    groups = np.arange(1, means.size + 1)

    ax1.set_xlabel("组号")
    ax1.set_ylabel("组均值")
    l1 = ax1.plot(groups, means, marker="o", label="各组均值")

    ax2 = ax1.twinx()
    ax2.set_ylabel("组标准差")
    l2 = ax2.plot(groups, stds, marker="s", linestyle="--", label="各组标准差")

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title(title)
    fig.tight_layout()

    out_path = os.path.join(BASE_DIR, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =========================================================
# 主流程：一键执行整个实验分析管线
# =========================================================
def main() -> None:
    # -------- 第0步：读取数据 --------
    data = load_data()  # 强制从 BASE_DIR/data.csv 读取
    num_groups, n_meas = data.shape

    # -------- 第1步：基础统计与原始可视化（fig1）--------
    raw_means, raw_stds = basic_stats(data)
    plot_raw_groups(data, TREND_GROUPS_FOR_PLOT, "fig1.png")

    # -------- 第2步：粗大误差检测与可视化对比（fig2）--------
    outlier_groups: List[int] = []   # p：哪些组出现了粗大误差
    display_rows: List[np.ndarray] = []

    for gi in range(num_groups):
        res = detect_and_remove_outliers(data[gi])
        # 如果有任何点被标为False，说明该组出现粗大误差
        if np.any(~res.mask_good):
            outlier_groups.append(gi + 1)  # 用 1~20 编号记录
        display_rows.append(res.display)

    # display_arr 是“剔除粗大误差后”的可视化版本（异常点用中位数等方式替代）
    display_arr = np.vstack(display_rows)

    # fig2：展示粗大误差剔除前后（要求挑4组：1/5/10/20）
    plot_outlier_comparison(data, display_arr, TREND_GROUPS_FOR_PLOT, "fig2.png")

    # -------- 第3步：系统误差诊断与校正（fig3）--------
    # 使用已经做过“粗大误差点替代”的 display_arr 作为输入
    sys_result = apply_system_corrections(display_arr)
    corrected = sys_result.corrected

    # fig3：展示系统误差校正前后（同样选4组：1/5/10/20）
    plot_system_correction(sys_result.before_after_pairs,
                           TREND_GROUPS_FOR_PLOT,
                           "fig3.png")

    # -------- 第4步：等/不等精度判定 + 全局最优估计（fig4）--------
    # 在系统误差校正后的数据 corrected 上重新计算均值与标准差
    final_means, final_stds = basic_stats(corrected)

    # 用标准差的最大/最小比值判断是否可以近似认为“等精度”
    s_max = float(np.max(final_stds))
    s_min = float(max(np.min(final_stds), 1e-12))
    equal_precision = (s_max / s_min) < THRESH_EQUAL_PRECISION_RATIO

    if equal_precision:
        # 等精度情况：简单等权平均
        optimal_mean = float(np.mean(final_means))
        if final_means.size > 1:
            # 组均值之间的离散程度 -> 估计整体不确定度
            sigma_groups = float(np.std(final_means, ddof=1))
            sigma_opt = sigma_groups / math.sqrt(final_means.size)
        else:
            sigma_opt = 0.0
        limit_error = K_LIMIT * sigma_opt
    else:
        # 不等精度情况：按 1/σ^2 加权平均
        optimal_mean, limit_error = weighted_average(final_means, final_stds)

    # fig4：绘制“各组均值 & 各组标准差”
    plot_means_stds(
        final_means,
        final_stds,
        "fig4.png",
        "系统误差校正后的各组均值与标准差分布"
    )
    
    

    # -------- 第5步：写入结果文件 results.txt --------
    save_results(
        raw_means,
        raw_stds,
        outlier_groups,
        sys_result,
        final_means,
        final_stds,
        float(optimal_mean),
        float(limit_error),
        equal_precision,
    )

    print("分析完成 ✅，以下文件已生成到目录:", BASE_DIR)
    print(" - fig1.png （原始测量曲线）")
    print(" - fig2.png （粗大误差剔除前后对比）")
    print(" - fig3.png （系统误差校正前后对比）")
    print(" - fig4.png （组均值/标准差分布）")
    print(" - results.txt （m1,s1,p,t1,t2,t3,a,总结）")

        # ======== 终端输出等/不等精度判定结果 ========
    s_ratio = s_max / s_min
    print("\n=== 等精度判定结果 ===")
    print(f"最大标准差 s_max = {s_max:.4f}")
    print(f"最小标准差 s_min = {s_min:.4f}")
    print(f"s_max / s_min = {s_ratio:.3f}")
    if equal_precision:
        print("判定结果：满足等精度条件 (max/min < 2.0)")
    else:
        print("判定结果：不满足等精度条件 (max/min ≥ 2.0)")
    print(f"最终测量结果 a = {optimal_mean:.4f} ± {limit_error:.4f} (3σ 极限误差)\n")



# =========================================================
# 脚本入口
# =========================================================
if __name__ == "__main__":
    main()

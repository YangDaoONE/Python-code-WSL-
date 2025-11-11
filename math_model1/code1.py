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

一键流程：
1. 从 BASE_DIR/data.csv 读取测量数据矩阵 (20 组 × 每组100次测量)。
2. 基础统计，绘制原始测量曲线，输出 fig1.png。
3. 粗大误差检测与替代显示，输出 fig2.png。
4. 系统误差诊断（恒值偏差 / 线性趋势 / 周期项）并校正，输出 fig3.png。
5. 判断等精度 / 不等精度，并计算最终测量结果；绘制 fig4.png。
6. 生成 results.txt（含 m1、s1、p、t1、t2、t3、a、总结）。

所有输出文件均写到脚本所在目录 BASE_DIR。
"""

# =========================================================
# Matplotlib 中文显示设置，避免中文乱码
# =========================================================
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Arial Unicode MS",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False  # 让负号正常显示

# =========================================================
# 固定的实验工作目录（即当前脚本所在的位置）
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE_DIR, exist_ok=True)
print("实验工作目录（读写路径）:", BASE_DIR)

# =========================================================
# 全局参数（在报告中会解释这些参数的依据）
# =========================================================
np.random.seed(12345)            # 固定随机种子，保证复现性
ALPHA_Z = 0.01                   # 恒值系统偏差 z 检验显著性水平（双侧）
ALPHA_SLOPE = 0.01               # 线性趋势显著性水平（回归斜率p值阈值）
PERIOD_ENERGY_RATIO = 0.2        # 周期性误差判定阈值：主频能量占总能量比例
TREND_GROUPS_FOR_PLOT = (0, 4, 9, 19)  # 要重点展示的四组：第1、5、10、20组(索引从0开始)
THRESH_EQUAL_PRECISION_RATIO = 2.0     # 等精度判据：max(s)/min(s) < 2 视为近似等精度
SIGMA_THRESHOLD = 3.0            # 粗大误差判据：迭代 3σ
K_LIMIT = 3.0                    # 极限误差按 3σ 给出
MAX_OUTLIER_ITER = 10            # 粗大误差迭代最大轮数


# =========================================================
# 数据结构（方便传递结果）
# =========================================================
@dataclass
class OutlierResult:
    """
    描述一组数据（长度约100）在粗大误差处理后的结果。

    mask_good: True/False，表示该点是否判为“正常”(True) 或“粗大误差”(False)
    display:   替换后的完整序列（长度不变），
               粗大误差点已用邻域中位数/全局中位数替代。
               这个序列后续也会作为“计算用序列”。
    """
    mask_good: np.ndarray                     # <<< 修改: 只保留 mask_good
    display: np.ndarray                       # <<< 修改: 去掉 cleaned 概念


@dataclass
class SystemCorrectionResult:
    """
    系统误差诊断与校正结果。
    corrected:                 校正后的数据 (20×N)，N≈100。
    constant_bias_groups:      恒值系统偏差(整体偏移)显著的组号列表 -> t1
    linear_trend_groups:       线性趋势(随测量序号漂移)显著的组号列表 -> t2
    periodic_groups:           周期性成分显著的组号列表 -> t3
    before_after_pairs:        [(校正前序列, 校正后序列), ...]，用于画 fig3。
    """
    corrected: np.ndarray
    constant_bias_groups: List[int]
    linear_trend_groups: List[int]
    periodic_groups: List[int]
    before_after_pairs: List[Tuple[np.ndarray, np.ndarray]]


class DataShapeError(ValueError):
    """
    当 data.csv 的形状不满足 20行×100列（至少）时抛出，防止后续处理崩溃。
    """


# =========================================================
# 读取数据
# =========================================================
def load_data() -> np.ndarray:
    """
    从 BASE_DIR/data.csv 读取观测矩阵。

    要求：
      - 行：20组独立测量
      - 列：每组100次重复测量
    若 data.csv 更大，本函数会裁剪前20行、前100列，保持规范一致。
    """
    csv_path = os.path.join(BASE_DIR, "data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[致命错误] 未找到 data.csv ：{csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", dtype=float)

    if data.ndim != 2:
        raise DataShapeError("data.csv 必须是二维数组 (20行×100列 及以上)")
    if data.shape[0] < 20 or data.shape[1] < 100:
        raise DataShapeError(
            f"data.csv 尺寸为 {data.shape}，不满足至少20行×100列的要求"
        )

    # 保留前20行、前100列
    data = data[:20, :100]
    return data


# =========================================================
# 基础统计
# =========================================================
def basic_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入:
        data: shape=(20, N)

    输出:
        means: 每组的算术平均值(长度20) -> 实验要求的 m1
        stds:  每组的样本标准差(长度20, ddof=1) -> 实验要求的 s1
    """
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1, ddof=1)
    return means, stds


# =========================================================
# 粗大误差检测 (迭代3σ法)
# =========================================================
def detect_and_remove_outliers(row: np.ndarray) -> OutlierResult:
    """
    对单组数据进行粗大误差检测，并生成“用于后续一切分析的替代后序列”。

    步骤（迭代3σ）:
      1. 初始假设当前“保留点”服从近似正态。
      2. 计算均值 μ、标准差 σ。
      3. 找出 |x-μ| > 3σ 的点，标记为粗大误差并剔除（仅在 mask 中剔除）。
      4. 重新估计 μ、σ 并继续，直到没有新异常或迭代上限。

    然后：
      对所有判为粗大误差的点，用邻域中位数（或全组中位数）替换，得到 display。
      这个 display：
        - 长度保持不变（例如100点）
        - 后续将作为“真实用于计算的序列”。

    返回:
      mask_good: True=正常, False=粗大误差
      display:   替代后的完整序列（也是后续计算用到的序列）
    """
    if row.size == 0:
        return OutlierResult(
            mask_good=np.zeros(0, dtype=bool),
            display=row.copy(),
        )

    mask_good = np.ones_like(row, dtype=bool)

    for _ in range(MAX_OUTLIER_ITER):
        candidate = row[mask_good]
        if candidate.size <= 1:
            break
        mu = candidate.mean()
        sd = candidate.std(ddof=1)
        if sd == 0:
            break

        z = np.abs((row - mu) / sd)
        new_bad = (z > SIGMA_THRESHOLD) & mask_good
        if not np.any(new_bad):
            break
        mask_good[new_bad] = False

    # 计算“正常点”的中位数，用来兜底
    safe_median = np.median(row[mask_good]) if np.any(mask_good) else row.mean()

    # 构造替代后完整序列
    display = row.copy()
    bad_idx = np.where(~mask_good)[0]
    for bi in bad_idx:
        L = max(0, bi - 2)
        R = min(len(row), bi + 3)
        neigh = [j for j in range(L, R) if mask_good[j]]
        if len(neigh) > 0:
            display[bi] = np.median(row[neigh])
        else:
            display[bi] = safe_median

    return OutlierResult(
        mask_good=mask_good,
        display=display,
    )


# =========================================================
# 系统误差判定工具
# =========================================================
def z_test_mean(
    group_mean: float,
    group_std: float,
    n: int,
    global_mean: float,
    alpha: float = ALPHA_Z,
) -> bool:
    """
    检测恒值系统偏差（整体偏移）：
      H0: 该组均值 == 全局加权均值
      z = (m_i - m_global) / (s_i / sqrt(n))
      |z| 超过正态分布的双侧阈值 => 拒绝H0 => 判为存在恒值偏差
    """
    if n <= 1:
        return False
    safe_std = max(group_std, 1e-12)
    z_val = (group_mean - global_mean) / (safe_std / math.sqrt(n))
    if np.isnan(z_val):
        return False
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return abs(z_val) > z_crit


def linear_trend_test(y: np.ndarray, alpha: float = ALPHA_SLOPE):
    """
    检测线性趋势（线性系统误差）：
      - 拟合 y = a + b x
      - 用斜率 b 的显著性检验判断是否存在趋势
      - 若显著，只扣除“相对均值的线性漂移项” b*(x - mean(x))，从而保留原始均值
    """
    n = y.size
    if n <= 2:
        return False, 0.0, 0.0, y.copy()

    x = np.arange(1, n + 1)
    slope, intercept, _, p_value, _ = stats.linregress(x, y)
    has_trend = bool(p_value < alpha)

    if has_trend:
        # 仅去掉“相对均值的线性漂移”，保留 y 的均值
        x_centered = x - np.mean(x)
        trend_drift = slope * x_centered
        detrended = y - trend_drift
        # 现在：detrended 的均值 ≈ y 的均值（而非 0）
    else:
        detrended = y.copy()

    return has_trend, slope, intercept, detrended



def detect_periodic(y: np.ndarray, ratio_thresh: float = PERIOD_ENERGY_RATIO):
    """
    检测周期性系统误差 (已修正，保留均值)
    ...
    """
    n = y.size
    if n == 0:
        return False, 0.0, np.zeros_like(y), y.copy()

    demeaned = y - y.mean()
    spectrum = rfft(demeaned)
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
        return False, 0.0, np.zeros_like(y), y.copy()

    idx_nonzero = np.arange(1, power.size)
    dom_idx = idx_nonzero[np.argmax(power[1:])]
    dom_power = power[dom_idx]
    total_power = power.sum()
    ratio = dom_power / total_power if total_power > 0 else 0.0

    has_period = ratio > ratio_thresh
    freq_arr = rfftfreq(n, d=1.0)
    f = freq_arr[dom_idx]

    if has_period and f != 0:
        k = np.arange(n)
        
        # ===== 开始修正逻辑 =====
        
        # 1. 设计矩阵 *只* 包含周期项
        design_periodic = np.column_stack([
            np.sin(2 * np.pi * f * k),
            np.cos(2 * np.pi * f * k)
        ])
        
        # 2. 完整的拟合矩阵，包含均值项 (np.ones)
        design_full = np.column_stack([
            design_periodic,
            np.ones(n)
        ])
        
        # 3. 拟合完整模型 (A*sin + B*cos + C)
        try:
            # coef_full 将包含 [A, B, C]
            coef_full, *_ = np.linalg.lstsq(design_full, y, rcond=None)
        except np.linalg.LinAlgError:
            # 如果拟合失败，放弃校正
            return False, 0.0, np.zeros_like(y), y.copy()

        # 4. 只提取周期项的系数 (A 和 B)
        coef_periodic = coef_full[:2]
        
        # 5. 'fitted' 变量现在 *只* 代表周期波动部分 (A*sin + B*cos)
        fitted = design_periodic @ coef_periodic
        
        # 6. 校正：从 y 中 *只* 减去周期波动
        corrected = y - fitted
        
        # ===== 结束修正逻辑 =====
        
        # --- 旧的错误逻辑 ---
        # design = np.column_stack([
        #     np.sin(2 * np.pi * f * k),
        #     np.cos(2 * np.pi * f * k),
        #     np.ones(n)
        # ])
        # coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        # fitted = design @ coef      # 错误：fitted 包含了均值
        # corrected = y - fitted    # 错误：导致均值归零
        # --- 结束旧逻辑 ---

    else:
        fitted = np.zeros_like(y)
        corrected = y.copy()

    return has_period, f, fitted, corrected


# =========================================================
# 系统误差诊断 + 校正
# =========================================================
def apply_system_corrections(data: np.ndarray) -> SystemCorrectionResult:
    """
    输入:
        data: (20,N)，这里的 data 是“粗大误差替换后”的序列，
              长度仍为 N=100（不丢点）。

    对每一组依次进行：
      1. 恒值偏差：z检验 -> 若存在 -> 整体平移回全局加权均值
      2. 线性趋势：线性回归 -> 若斜率显著 -> 去趋势
      3. 周期性误差：FFT主频能量比 -> 若显著 -> 正弦拟合并扣除

    输出：
      corrected                : 校正后的20×N矩阵
      constant_bias_groups (t1): 有恒值偏差的组号(1起始)
      linear_trend_groups  (t2): 有线性趋势的组号(1起始)
      periodic_groups      (t3): 有周期项的组号(1起始)
      before_after_pairs       : 每组(校正前, 校正后)用于画fig3
    """
    group_means, group_stds = basic_stats(data)
    weights = 1.0 / (np.maximum(group_stds, 1e-12) ** 2)
    global_mean = np.average(group_means, weights=weights)

    corrected = np.zeros_like(data)
    t1_list: List[int] = []
    t2_list: List[int] = []
    t3_list: List[int] = []
    before_after_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for i, series in enumerate(data):
        n = series.size

        # (a) 恒值系统偏差
        has_bias = z_test_mean(group_means[i], group_stds[i], n, global_mean)
        bias_corrected = series.copy()
        if has_bias:
            offset = group_means[i] - global_mean
            bias_corrected = series - offset
            t1_list.append(i + 1)  # 组号用 1~20

        # (b) 线性趋势
        has_trend, slope, intercept, detrended = linear_trend_test(
            bias_corrected
        )
        trend_corrected = bias_corrected.copy()
        if has_trend:
            trend_corrected = detrended
            t2_list.append(i + 1)

        # (c) 周期项
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
# 不等精度加权平均
# =========================================================
def weighted_average(means: np.ndarray, stds: np.ndarray) -> Tuple[float, float]:
    """
    不等精度加权平均：
      w_i = 1 / σ_i^2
      a   = Σ(w_i m_i)/Σ(w_i)
      σ_a = sqrt(1/Σw_i)
      极限误差 = 3σ_a
    """
    safe_stds = np.maximum(stds, 1e-12)
    w = 1.0 / (safe_stds ** 2)
    a = float(np.sum(w * means) / np.sum(w))
    sigma_a = math.sqrt(1.0 / np.sum(w))
    limit_err = K_LIMIT * sigma_a
    return a, float(limit_err)


# =========================================================
# 文本格式化（写 results.txt 用）
# =========================================================
def fmt_float_list(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):.2f}" for v in values)

def fmt_int_list(values: Iterable[int]) -> str:
    return ",".join(str(int(v)) for v in values)


# =========================================================
# 写 results.txt
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
    写入 results.txt，格式为8行：
      1行: m1 (20个均值)
      2行: s1 (20个标准差)
      3行: p  (存在粗大误差的组号，可为空)
      4行: t1 (恒值系统误差组号，可为空)
      5行: t2 (线性系统误差组号，可为空)
      6行: t3 (周期性系统误差组号，可为空)
      7行: a  (最终测量结果 ± 极限误差)
      8行: 结果分析（中文总结）
    """

    summary_lines = [
        f"粗大误差检测采用迭代3σ法(阈值={SIGMA_THRESHOLD:.1f}σ)，共识别到 {len(outlier_groups)} 组存在疑似粗大误差点。",
        (
            "恒值系统偏差通过 z 检验(alpha={:.2f}) 判断组均值是否显著偏离全局加权均值；"
            "线性趋势通过回归斜率显著性检验(alpha={:.2f})判断是否存在随测量序号的漂移；"
            "周期性误差由主频能量占比阈值{:.2f}判定，并拟合正弦/余弦后扣除。"
        ).format(ALPHA_Z, ALPHA_SLOPE, PERIOD_ENERGY_RATIO),
        (
            "系统误差校正后，对20组数据计算新的均值和标准差，并判断其是否可视为等精度"
            f"(标准差最大/最小阈值为 {THRESH_EQUAL_PRECISION_RATIO:.1f})。"
            "据此选择等权平均或加权平均，最终给出全局测量结果及其3σ极限误差："
            f"{optimal_mean:.2f}±{limit_error:.2f}。"
        ),
        "在报告中将进一步讨论：阈值选择的合理性、对结果的敏感性、是否可能过拟合周期项、"
        "以及是否存在由少数高精度组主导整体结果的风险。",
    ]

    lines = [
        fmt_float_list(raw_means),                                 # 第1行 m1
        fmt_float_list(raw_stds),                                  # 第2行 s1
        fmt_int_list(outlier_groups),                              # 第3行 p
        fmt_int_list(corr_result.constant_bias_groups),            # 第4行 t1
        fmt_int_list(corr_result.linear_trend_groups),             # 第5行 t2
        fmt_int_list(corr_result.periodic_groups),                 # 第6行 t3
        f"{optimal_mean:.2f}±{limit_error:.2f}",                   # 第7行 a
        " ".join(summary_lines),                                   # 第8行 总结
    ]

    results_path = os.path.join(BASE_DIR, "results.txt")
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# =========================================================
# 子图布局工具 (subplot 版本，从 A 版移植)
# =========================================================
def _subplot_layout(count: int) -> Tuple[plt.Figure, np.ndarray, int, int]:
    """
    根据要画的组数 count，自动生成紧凑的子图网格。
    对4组数据 -> 2列布局，很适合(1,5,10,20)并列比较。
    """
    if count <= 0:
        raise ValueError("subplot count must be positive")

    cols = 2 if count > 1 else 1
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=True,
        figsize=(cols * 4.5, rows * 3.2)
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    return fig, axes_array, rows, cols


# =========================================================
# 绘图函数 (fig1, fig2, fig3 使用 subplot 风格；fig4 单图双轴)
# =========================================================
def plot_raw_groups(raw: np.ndarray, indices: Sequence[int], filename: str) -> None:
    """
    fig1: 原始测量曲线
    每个子图对应一组（例如第1、5、10、20组）。
    """
    fig, axes, rows, cols = _subplot_layout(len(indices))
    n = raw.shape[1]
    x_axis = np.arange(1, n + 1)

    for ax, idx in zip(axes, indices):
        ax.plot(x_axis, raw[idx], label="原始曲线", color="#1f77b4")
        ax.set_title(f"第{idx + 1}组")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    # 如果子图数多于要画的组（一般不会，但防御一下）
    for ax in axes[len(indices):]:
        ax.axis("off")

    # 统一坐标轴标签
    for i, ax in enumerate(axes):
        if i >= len(indices):
            continue
        row_i = i // cols
        if row_i == rows - 1:
            ax.set_xlabel("测量序号")
        ax.set_ylabel("测量值")

    fig.suptitle("典型4组原始测量曲线（第1/5/10/20组）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(BASE_DIR, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_outlier_comparison(
    before: np.ndarray,
    after: np.ndarray,
    indices: Sequence[int],
    filename: str
) -> None:
    """
    fig2: 粗大误差剔除前/后 对比
    before: 原始数据
    after : 粗大误差点用邻域中位数替换后的“平滑展示版”
    """
    fig, axes, rows, cols = _subplot_layout(len(indices))
    n = before.shape[1]
    x_axis = np.arange(1, n + 1)

    for ax, idx in zip(axes, indices):
        ax.plot(
            x_axis,
            before[idx],
            alpha=0.7,
            label="剔除前",
            color="#d62728",
        )
        ax.plot(
            x_axis,
            after[idx],
            linestyle="--",
            label="剔除后(可视化替代)",
            color="#2ca02c",
        )
        ax.set_title(f"第{idx + 1}组")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[len(indices):]:
        ax.axis("off")

    for i, ax in enumerate(axes):
        if i >= len(indices):
            continue
        row_i = i // cols
        if row_i == rows - 1:
            ax.set_xlabel("测量序号")
        ax.set_ylabel("测量值")

    fig.suptitle("粗大误差剔除前后对比（迭代3σ法，异常点中位数替代）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(BASE_DIR, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_system_correction(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    indices: Sequence[int],
    filename: str
) -> None:
    """
    fig3: 系统误差校正前/后 对比
    pairs[i] = (校正前序列, 校正后序列)
    校正包括：恒值偏差平移、线性去趋势、周期项扣除
    """
    if not pairs:
        raise ValueError("没有可用于系统误差校正对比的数据。")

    fig, axes, rows, cols = _subplot_layout(len(indices))
    n = pairs[0][0].size
    x_axis = np.arange(1, n + 1)

    for ax, idx in zip(axes, indices):
        before_i, after_i = pairs[idx]
        ax.plot(
            x_axis,
            before_i,
            alpha=0.7,
            label="校正前",
            color="#9467bd",
        )
        ax.plot(
            x_axis,
            after_i,
            linestyle="--",
            label="校正后",
            color="#ff7f0e",
        )
        ax.set_title(f"第{idx + 1}组")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[len(indices):]:
        ax.axis("off")

    for i, ax in enumerate(axes):
        if i >= len(indices):
            continue
        row_i = i // cols
        if row_i == rows - 1:
            ax.set_xlabel("测量序号")
        ax.set_ylabel("测量值")

    fig.suptitle(
        "系统误差校正前后对比（恒值偏差平移 / 去趋势 / 去周期）",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(BASE_DIR, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_means_stds(
    means: np.ndarray,
    stds: np.ndarray,
    s_ratio: float,
    filename: str
) -> None:
    """
    fig4: 校正后20组的均值与标准差。
    左y轴：组均值
    右y轴：组标准差
    标题中直接显示 max(s)/min(s)，方便判断等/不等精度。
    """
    fig, ax1 = plt.subplots()
    groups = np.arange(1, means.size + 1)

    ax1.set_xlabel("组号")
    ax1.set_ylabel("组均值")
    l1 = ax1.plot(groups, means, marker="o", label="各组均值")

    ax2 = ax1.twinx()
    ax2.set_ylabel("组标准差")
    l2 = ax2.plot(groups, stds, marker="s", linestyle="--", label="各组标准差", color="#ff7f0e")

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    ax1.set_title(
        f"系统误差校正后：各组均值与标准差分布 (max(s)/min(s)={s_ratio:.2f})"
    )

    fig.tight_layout()

    out_path = os.path.join(BASE_DIR, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =========================================================
# 主流程
# =========================================================
def main() -> None:
    # Step 0: 读数据
    data = load_data()
    num_groups, n_meas = data.shape

    # Step 1: 基础统计 + fig1
    raw_means, raw_stds = basic_stats(data)
    plot_raw_groups(data, TREND_GROUPS_FOR_PLOT, "fig1.png")

    # Step 2: 粗大误差检测 + fig2
    outlier_groups: List[int] = []   # p：有粗大误差的组号
    display_rows: List[np.ndarray] = []

    for gi in range(num_groups):
        res = detect_and_remove_outliers(data[gi])
        if np.any(~res.mask_good):
            outlier_groups.append(gi + 1)  # 组号用1~20
        display_rows.append(res.display)

    # 替代后的版本（长度保持100，作为后续计算输入）
    display_arr = np.vstack(display_rows)

    # fig2: 粗大误差剔除前后对比
    plot_outlier_comparison(
        before=data,
        after=display_arr,
        indices=TREND_GROUPS_FOR_PLOT,
        filename="fig2.png"
    )

    # Step 3: 系统误差诊断/校正 + fig3
    sys_result = apply_system_corrections(display_arr)
    corrected = sys_result.corrected

    # fig3: 系统误差校正前后对比
    plot_system_correction(
        pairs=sys_result.before_after_pairs,
        indices=TREND_GROUPS_FOR_PLOT,
        filename="fig3.png"
    )

    # Step 4: 等/不等精度判定 + 最优估计 + fig4
    final_means, final_stds = basic_stats(corrected)

    s_max = float(np.max(final_stds))
    s_min = float(max(np.min(final_stds), 1e-12))
    s_ratio = s_max / s_min
    equal_precision = (s_ratio < THRESH_EQUAL_PRECISION_RATIO)

    if equal_precision:
        # 等精度：等权平均
        optimal_mean = float(np.mean(final_means))
        if final_means.size > 1:
            sigma_groups = float(np.std(final_means, ddof=1))
            sigma_opt = sigma_groups / math.sqrt(final_means.size)
        else:
            sigma_opt = 0.0
        limit_error = K_LIMIT * sigma_opt
    else:
        # 不等精度：按1/σ^2加权平均
        optimal_mean, limit_error = weighted_average(final_means, final_stds)

    # fig4: 绘制“各组均值 & 各组标准差”
    plot_means_stds(
        means=final_means,
        stds=final_stds,
        s_ratio=s_ratio,
        filename="fig4.png"
    )

    # Step 5: 写 results.txt
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

    # ======== 友好提示输出 ========
    print("分析完成 ✅，以下文件已生成到目录:", BASE_DIR)
    print(" - fig1.png （典型4组原始测量曲线）")
    print(" - fig2.png （粗大误差剔除前后对比）")
    print(" - fig3.png （系统误差校正前后对比）")
    print(" - fig4.png （各组均值/标准差分布，含max(s)/min(s)）")
    print(" - results.txt （m1,s1,p,t1,t2,t3,a,总结）")

    # ======== 终端输出等精度判定信息，方便写报告 ========
    print("\n=== 等精度判定结果 ===")
    print(f"最大标准差 s_max = {s_max:.6f}")
    print(f"最小标准差 s_min = {s_min:.6f}")
    print(f"s_max / s_min   = {s_ratio:.6f}")
    if equal_precision:
        print(f"判定：满足等精度条件 (max/min < {THRESH_EQUAL_PRECISION_RATIO})")
    else:
        print(f"判定：不满足等精度条件 (max/min ≥ {THRESH_EQUAL_PRECISION_RATIO})")
    print(f"最终测量结果 a = {optimal_mean:.6f} ± {limit_error:.6f} (3σ极限误差)")
    print()


# =========================================================
# 入口
# =========================================================
if __name__ == "__main__":
    main()

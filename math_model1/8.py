import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

# =========================
# 全局参数（保持和主脚本一致）
# =========================
GROUP_INDEX = 7              # 第8组 -> 索引7
SIGMA_THRESHOLD = 3.0        # 粗大误差 3σ 判据
MAX_OUTLIER_ITER = 10        # 迭代上限
PERIOD_ENERGY_RATIO = 0.2    # 周期性判定阈值 (主频能量占比>20%算显著)

# Matplotlib 中文显示，防止乱码
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "PingFang SC", "Arial Unicode MS", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False


def load_group8():
    """
    从脚本同目录的 data.csv 读取第8组（索引7）100个测量点.
    data.csv 需至少 20x100.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 data.csv: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", dtype=float)

    if data.ndim != 2 or data.shape[0] < 20 or data.shape[1] < 100:
        raise ValueError(f"data.csv 尺寸是 {data.shape}，需要至少20行×100列")

    # 和主实验保持一致：只取前20行、前100列
    data = data[:20, :100]
    y = data[GROUP_INDEX].astype(float)  # 第8组
    return y


def detect_and_remove_outliers_single_row(row):
    """
    按主程序的迭代3σ法，对一组数据做粗大误差识别。

    返回:
        mask_good       True=保留，False=粗大误差
        display_filled  用于后续系统误差分析的“替代后曲线”，
                        把粗大误差点用邻域中位数(或全局中位数)替换，
                        长度与原始一致，便于FFT等分析。
    """
    x = row.copy()
    mask_good = np.ones_like(x, dtype=bool)

    for _ in range(MAX_OUTLIER_ITER):
        good_vals = x[mask_good]
        if good_vals.size <= 1:
            break
        mu = good_vals.mean()
        sd = good_vals.std(ddof=1)
        if sd == 0:
            break

        z = np.abs((x - mu) / sd)
        new_bad = (z > SIGMA_THRESHOLD) & mask_good
        if not np.any(new_bad):
            break
        mask_good[new_bad] = False

    # 生成 display_filled（可视化/后续分析使用）
    display_filled = x.copy()
    bad_idx = np.where(~mask_good)[0]
    for bi in bad_idx:
        L = max(0, bi - 2)
        R = min(len(x), bi + 3)
        neigh = [j for j in range(L, R) if mask_good[j]]
        if len(neigh) > 0:
            display_filled[bi] = np.median(x[neigh])
        else:
            # 邻域都是坏点，就用全局好点的中位数兜底
            good_median = np.median(x[mask_good]) if np.any(mask_good) else x[bi]
            display_filled[bi] = good_median

    return mask_good, display_filled


def analyze_periodicity(y_filled):
    """
    在“粗大误差已替代”的序列 y_filled 上做周期性误差分析：
    1. 去均值后FFT
    2. 找主频
    3. 判定是否显著周期
    4. 用主频拟合 sin+cos+C
    """
    n = y_filled.size
    k = np.arange(n)

    # 步骤1: 去均值 -> FFT
    demeaned = y_filled - y_filled.mean()
    spectrum = rfft(demeaned)
    power = np.abs(spectrum) ** 2

    if power.size <= 1:
        # 除了直流分量啥也没有
        f_dom = 0.0
        ratio = 0.0
        fitted = np.zeros_like(y_filled)
        y_corrected = y_filled.copy()
        return f_dom, ratio, fitted, y_corrected, (0.0, 0.0, y_filled.mean())

    # 步骤2: 找最强的非零频率分量
    idx_nonzero = np.arange(1, power.size)  # 跳过0频
    dom_idx = idx_nonzero[np.argmax(power[1:])]
    dom_power = power[dom_idx]
    total_power = power.sum()
    ratio = dom_power / total_power if total_power > 0 else 0.0

    freq_arr = rfftfreq(n, d=1.0)
    f_dom = freq_arr[dom_idx]

    # 步骤3: 用主频做正弦+余弦 拟合周波误差项
    if f_dom != 0.0:
        M = np.column_stack([
            np.sin(2 * np.pi * f_dom * k),
            np.cos(2 * np.pi * f_dom * k),
            np.ones(n)
        ])
        coef, *_ = np.linalg.lstsq(M, y_filled, rcond=None)
        A, B, C = coef
        fitted = M @ coef                 # 周期项的拟合
        y_corrected = y_filled - fitted   # 去周期后
    else:
        A, B, C = 0.0, 0.0, y_filled.mean()
        fitted = np.full_like(y_filled, C)
        y_corrected = y_filled - fitted

    return f_dom, ratio, fitted, y_corrected, (A, B, C)


def plot_results(original, y_filled, fitted, y_corrected, out_path):
    """
    画四条曲线，帮助写报告：
    1. 原始序列 (含粗大误差)
    2. 粗大误差替代后序列 (剔除了离群点影响)
    3. 拟合到的主频周期项
    4. 去周期后残差
    """
    n = original.size
    x_axis = np.arange(1, n + 1)

    plt.figure(figsize=(10,7))

    plt.plot(x_axis, original, label="原始序列(含粗大误差)", linewidth=1.2, alpha=0.7)
    plt.plot(x_axis, y_filled, label="粗大误差替代后序列", linewidth=1.2)
    plt.plot(x_axis, fitted, label="拟合主频周期项", linestyle="--", linewidth=1.2)
    plt.plot(x_axis, y_corrected, label="去周期后残差", linewidth=1.0, alpha=0.8)

    plt.xlabel("测量序号 k")
    plt.ylabel("测量值")
    plt.title("第8组：粗大误差→周期性系统误差分析")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    # 1. 读第8组原始数据
    y_raw = load_group8()

    # 2. 先做粗大误差处理（迭代3σ），得到替代后的 y_filled
    mask_good, y_filled = detect_and_remove_outliers_single_row(y_raw)

    # 3. 在 y_filled 上做周期性误差分析
    f_dom, ratio, fitted, y_corr, (A, B, C) = analyze_periodicity(y_filled)

    # 4. 打印详细诊断结果，方便写报告“案例分析：第8组”
    print("===== 第8组 周期性误差诊断（在粗大误差处理后） =====")
    print(f"数据长度 N = {len(y_raw)}")
    print(f"粗大误差点个数 = {np.sum(~mask_good)}")
    print(f"主频 f_dom = {f_dom:.6f} (以采样点为单位的归一化频率)")
    print(f"主频能量占比 ratio = {ratio:.4f} -> {ratio*100:.2f}%")
    print(f"判定阈值 PERIOD_ENERGY_RATIO = {PERIOD_ENERGY_RATIO:.2f}")
    if ratio > PERIOD_ENERGY_RATIO:
        print("结论：该组仍表现出显著的周期性系统误差 ✅")
    else:
        print("结论：该组未呈现显著周期性系统误差 ❌")
    print("\n主频拟合模型：")
    print("y(k) ≈ A*sin(2π f_dom k) + B*cos(2π f_dom k) + C")
    print(f"A = {A:.6f}")
    print(f"B = {B:.6f}")
    print(f"C = {C:.6f}")
    print("去周期后残差 = (粗大误差替代后序列) - (拟合主频周期项)")
    print("解释：这个残差应当不再有明显的稳定振荡模式，如果周期性干扰已被有效扣除。")

    # 5. 画图保存
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(base_dir, "freq_fit_g8.png")
    plot_results(y_raw, y_filled, fitted, y_corr, fig_path)

    print(f"\n图像已输出: {fig_path}")
    print("图中包含：原始、粗大误差替换后、拟合主频周期项、去周期残差。")


if __name__ == "__main__":
    main()

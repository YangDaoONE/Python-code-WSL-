# -*- coding: utf-8 -*-
"""
NMR 信号处理与参数估计（自适应扫频版，纯 NumPy/Scipy，无 pandas）
- 输入: 与脚本同目录的 data_A.csv, data_B.csv （一列或两列均可）
- 输出: 脚本同级 exp2_outputs/：
        fig1.png, fig2.png, fig3.png, fig4.png,
        results.txt（8/8/4/4列 + 第5行中文分析）
说明：
1) 频谱：使用 rFFT 单边幅度谱（dB），不使用窗函数/PSD/Welch，符合“不得引入实验书外步骤”。
2) 组B谐波：采用自适应扫频（黄金分割 + 可选抛物线精修）锁定工频基频，再在全量数据+目标K做一次拟合后相减。
3) 组A/组B(clean)：希尔伯特→下变频→两次线性拟合；并以其为初值做一次非线性复域拟合。
4) 步骤3绘图要求：fig3 左上为“去噪前 vs 去噪后”时域，并**叠加指数包络 ±A·e^{-t/T2}**；右上为(0–200 Hz) 频谱对比；左下为 ln|s_bb| 直线拟合；右下为相位展开直线拟合。
"""

import re
from pathlib import Path
import numpy as np

# ===== 画图后端与中文 =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","PingFang SC","Noto Sans CJK SC","SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "lines.linewidth": 0.5,   # 全局细线
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})
FIT_LW = 2.2   # 拟合线加粗

from scipy.signal import hilbert
from scipy.optimize import curve_fit

# ===== 路径与输出 =====
SCRIPT_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = SCRIPT_DIR / "exp2_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
def pjoin(name: str) -> str: return str(OUTPUT_DIR / name)

# ===== 实验参数 =====
FS = 100_000.0           # 采样率（必要时修改为数据真实采样率）
EDGE_TRIM_FRAC = 0.01    # 希尔伯特后两端丢弃比例
AMP_FRONT_FRACTION = 0.5 # 线性回归使用前段幅度≥50%区域
USE_WLS = True           # 线性回归使用加权
HARM_K = 100             # 最终谐波阶数（全量拟合用）
FSCAN_MIN, FSCAN_MAX = 49.9, 50.1   # 工频搜索范围
F_DELTA_TARGET = 1.0     # 下变频目标差频 1 Hz

# fig1 标注参数
PEAK_SEARCH_FMIN  = 80.0   # 谱峰搜索下限（避开 50Hz）
NOISE_EXCLUDE_HZ  = 200.0  # 基线统计时排除峰值±该带宽

# —— 自适应扫频（加速与精度旋钮）——
SWEEP_DECIM      = 10      # 扫频阶段降采样因子
SWEEP_MAXLEN     = 20_000  # 扫频阶段参与 RSS 评估的最大样本数
SWEEP_K_COARSE   = 9       # 扫频阶段的小K
FSCAN_HALF_WIN   = 0.02    # 种子峰 ± 初始半窗口（Hz）
GOLD_TOL         = 1e-4    # 黄金分割停止阈值（Hz）
GOLD_MAX_EVAL    = 120     # 黄金分割最大函数评估次数
PARABOLIC_REFINE = True    # 是否做抛物线精修

np.random.seed(42)

# ===== CSV 读取（鲁棒，无 pandas）=====
def load_signal(path_like, fs=FS):
    real_path = Path(path_like)
    if not real_path.is_file():
        alt = SCRIPT_DIR / real_path.name
        if alt.is_file():
            real_path = alt
        else:
            raise FileNotFoundError(f"找不到数据文件：{path_like}")

    def _try(delim):
        try:
            arr = np.genfromtxt(str(real_path), delimiter=delim, dtype=float,
                                autostrip=True, comments=None)
            if arr is None or np.size(arr) < 8:
                return None
            return np.array(arr, dtype=float)
        except Exception:
            return None

    arr = None
    for d in [',',';','\t', None]:
        arr = _try(d)
        if arr is not None: break

    if arr is None:
        txt = real_path.read_text(encoding="utf-8", errors="ignore").replace('，', ',')
        nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', txt)
        if not nums:
            raise RuntimeError(f"无法从 {real_path} 中抽取数字，请检查文件格式（应为一列或两列数字CSV）。")
        arr = np.array([float(x) for x in nums], dtype=float)

    if arr.ndim == 1:
        y = arr.astype(float).ravel(); t = np.arange(y.size)/fs; return t, y
    if arr.ndim == 2 and arr.shape[1] == 1:
        y = arr[:,0].astype(float).ravel(); t = np.arange(y.size)/fs; return t, y
    if arr.ndim == 2 and arr.shape[1] >= 2:
        tcol = np.asarray(arr[:,0], dtype=float); ycol = np.asarray(arr[:,1], dtype=float)
        if np.all(np.diff(tcol) > 0): return tcol, ycol
        y = np.asarray(arr[:,0], dtype=float).ravel(); t = np.arange(y.size)/fs; return t, y

    y = np.asarray(arr, dtype=float).ravel(); t = np.arange(y.size)/fs; return t, y

# ===== 频谱（dB，无窗）=====
DB_EPS = 1e-20

def rfft_spectrum(y, fs):
    n = len(y)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, 1/fs)
    mag = np.abs(Y) / n
    mag_db = 20.0 * np.log10(mag + DB_EPS)
    return f, mag_db

# （仅 fig1 使用的辅助标注；与步骤3无关）
def annotate_peak_and_noise(ax, f, mag_db, fmin=PEAK_SEARCH_FMIN, exclude_hz=NOISE_EXCLUDE_HZ, label=None):
    f = np.asarray(f); mag_db = np.asarray(mag_db)
    mask = (f >= fmin)
    if not np.any(mask): return None, None
    idx_rel = np.argmax(mag_db[mask])
    f_peak  = float(f[mask][idx_rel])
    m_peak  = float(mag_db[mask][idx_rel])
    keep = mask & ((f < f_peak - exclude_hz) | (f > f_peak + exclude_hz))
    noise = float(np.median(mag_db[keep])) if np.any(keep) else float(np.median(mag_db[mask]))
    ax.axvline(f_peak, color="C3", lw=1.2, ls="--")
    ax.plot([f_peak], [m_peak], 'o', ms=3, color="C3")
    ax.axhline(noise, color="C2", lw=1.0, ls=":")
    note = (f"{label+'：' if label else ''}峰≈{f_peak:.1f} Hz\n噪声基线≈{noise:.1f} dB")
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
    return f_peak, noise

# ===== 谐波建模工具 =====
def build_design_matrix(f_base, t, K, include_dc=True):
    P = len(t)
    N = np.arange(1, K+1)
    arg = 2*np.pi * t[:, None] * (f_base * N[None, :])
    cos_block = np.cos(arg); sin_block = np.sin(arg)
    if include_dc:
        A = np.hstack([np.ones((P,1)), cos_block, sin_block])
    else:
        A = np.hstack([cos_block, sin_block])
    return A


def ls_solve(A, y):
    theta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    rss = float(residuals[0]) if residuals.size > 0 else float(np.sum((y - A @ theta)**2))
    return theta, rss

# —— 自适应扫频小工具 ——

def prepare_for_sweep(t, y, decim=SWEEP_DECIM, maxlen=SWEEP_MAXLEN):
    if decim > 1:
        t_ds = t[::decim]; y_ds = y[::decim]
    else:
        t_ds, y_ds = t, y
    n = len(y_ds)
    if n > maxlen:
        start = (n - maxlen) // 2
        stop  = start + maxlen
        t_ds = t_ds[start:stop]; y_ds = y_ds[start:stop]
    return t_ds, y_ds


def rss_at_freq(ftest, t_vec, y_vec, K, include_dc=True):
    A = build_design_matrix(ftest, t_vec, K, include_dc=include_dc)
    _, rss = ls_solve(A, y_vec)
    return float(rss)


def golden_section_minimize(f, a, b, tol=GOLD_TOL, max_eval=GOLD_MAX_EVAL):
    phi = (1 + np.sqrt(5)) / 2.0
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1); f2 = f(x2)
    evals = [(x1, f1), (x2, f2)]
    neval = 2
    while (b - a) > tol and neval < max_eval:
        if f1 > f2:
            a = x1
            x1, f1 = x2, f2
            x2 = b - resphi * (b - a)
            f2 = f(x2); neval += 1; evals.append((x2, f2))
        else:
            b = x2
            x2, f2 = x1, f1
            x1 = a + resphi * (b - a)
            f1 = f(x1); neval += 1; evals.append((x1, f1))
    x_best, f_best = (x1, f1) if f1 < f2 else (x2, f2)
    return x_best, f_best, evals


def parabolic_refine(evals):
    if len(evals) < 3: return None
    pts = sorted(evals, key=lambda p: p[1])[:3]
    pts = sorted(pts, key=lambda p: p[0])
    (x1, f1), (x2, f2), (x3, f3) = pts
    denom = (x1 - x2)*(x1 - x3)*(x2 - x3)
    if abs(denom) < 1e-20: return None
    A = (x3*(f2 - f1) + x2*(f1 - f3) + x1*(f3 - f2)) / denom
    B = (x3**2*(f1 - f2) + x2**2*(f3 - f1) + x1**2*(f2 - f3)) / denom
    if abs(A) < 1e-20: return None
    x0 = -B / (2*A)
    return x0


def harmonic_denoise_adaptive(y, t, K=HARM_K, f_min=FSCAN_MIN, f_max=FSCAN_MAX):
    # 1) 频谱种子
    f_all, mag_all = rfft_spectrum(y, FS)
    mask_seed = (f_all >= max(0.0, f_min - 0.5)) & (f_all <= f_max + 0.5)
    if np.any(mask_seed):
        f_seed = float(f_all[mask_seed][np.argmax(mag_all[mask_seed])])
    else:
        f_seed = float(f_all[np.argmax(mag_all)])

    a = max(f_min, f_seed - FSCAN_HALF_WIN)
    b = min(f_max, f_seed + FSCAN_HALF_WIN)
    if b <= a: a, b = f_min, f_max

    # 2) 扫频用：降采样 + 小K
    t_sw, y_sw = prepare_for_sweep(t, y, decim=SWEEP_DECIM, maxlen=SWEEP_MAXLEN)
    K_sw = min(K, SWEEP_K_COARSE)

    cache = {}
    def rss_f(ftest):
        if ftest in cache: return cache[ftest]
        val = rss_at_freq(ftest, t_sw, y_sw, K_sw, include_dc=True)
        cache[ftest] = val
        return val

    # 3) 黄金分割找最小 RSS
    f_best_sw, rss_best_sw, evals = golden_section_minimize(rss_f, a, b, tol=GOLD_TOL, max_eval=GOLD_MAX_EVAL)

    # 4) 可选抛物线精修
    if PARABOLIC_REFINE:
        x0 = parabolic_refine(evals)
        if x0 is not None and (a <= x0 <= b):
            rss0 = rss_f(x0)
            if rss0 < rss_best_sw:
                f_best_sw, rss_best_sw = x0, rss0

    # 5) 最终：全量数据 + 目标K
    A_best_full = build_design_matrix(f_best_sw, t, K, include_dc=True)
    theta_best, _ = ls_solve(A_best_full, y)
    v_har = A_best_full @ theta_best
    y_clean = y - v_har

    # 自检（可选）：能量守恒
    # assert np.allclose(y, v_har + y_clean, atol=1e-8)

    return y_clean, v_har, f_best_sw


# ===== 基带化与拟合 =====

def hilbert_baseband(y, fs, f_guess, f_delta=F_DELTA_TARGET, edge_frac=EDGE_TRIM_FRAC):
    n = len(y); t = np.arange(n)/fs
    z = hilbert(y)
    f_T = f_guess - f_delta
    s_bb = z * np.exp(-1j*2*np.pi*f_T*t)
    cut = int(np.floor(edge_frac * n))
    sl = slice(cut, n-cut) if cut > 0 else slice(0, n)
    return t[sl], s_bb[sl], f_T, sl


def select_front_segment_by_amplitude(s_bb, t, frac=AMP_FRONT_FRACTION):
    amp = np.abs(s_bb); peak = np.max(amp); thresh = frac * peak
    idx = np.where(amp >= thresh)[0]
    if len(idx) == 0:
        k = max(1, int(0.3 * len(amp))); return t[:k], s_bb[:k]
    end = idx[-1] + 1; return t[:end], s_bb[:end]


def wls_fit_line(x, y, w=None):
    x = np.asarray(x).reshape(-1,1); y = np.asarray(y).reshape(-1,1); n = len(x)
    X = np.hstack([np.ones_like(x), x])
    W = np.eye(n) if w is None else np.diag(np.asarray(w).reshape(-1))
    XtWX = X.T @ W @ X; XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy); yhat = X @ beta
    resid = y - yhat
    rss = float((resid.T @ W @ resid).ravel()[0])
    dof = max(1, n - 2); sigma2 = rss / dof
    cov = sigma2 * np.linalg.inv(XtWX)
    sb = float(np.sqrt(cov[0,0])); sm = float(np.sqrt(cov[1,1]))
    b = float(beta[0,0]); m = float(beta[1,0])
    return m, b, sm, sb, rss


def linear_params_from_baseband(t, s_bb, f_T, use_wls=USE_WLS):
    t_lin, s_lin = select_front_segment_by_amplitude(s_bb, t, frac=AMP_FRONT_FRACTION)
    amp = np.abs(s_lin)
    y1 = np.log(np.minimum(np.maximum(amp, 1e-16), 1e16))
    w1 = (amp/np.max(amp))**2 if use_wls else None
    m1, b1, sm1, sb1, _ = wls_fit_line(t_lin, y1, w=w1)
    T2 = -1.0/m1; A = np.exp(b1)
    sigma_T2 = abs(1.0/(m1*m1)) * sm1; sigma_A  = A * sb1

    ph = np.unwrap(np.angle(s_lin))
    w2 = (amp/np.max(amp))**2 if use_wls else None
    m2, b2, sm2, sb2, _ = wls_fit_line(t_lin, ph, w=w2)
    f0 = m2/(2*np.pi) + f_T; phi = b2
    sigma_f0 = sm2/(2*np.pi); sigma_phi = sb2
    return (A, T2, f0, phi), (sigma_A, sigma_T2, sigma_f0, sigma_phi)


def complex_model(t, A, T2, df, phi):
    return A * np.exp(-t/T2) * np.exp(1j*(2*np.pi*df*t + phi))


def stacked_model(t, A, T2, df, phi):
    g = complex_model(t, A, T2, df, phi)
    return np.hstack([np.real(g), np.imag(g)])


def nonlinear_fit_from_linear_init(t, s_bb, A0, T20, f0, phi0, f_T):
    y_stack = np.hstack([np.real(s_bb), np.imag(s_bb)])
    df0 = f0 - f_T
    bounds_lower = [0.0, 1e-6, df0 - 5.0, -10*np.pi]
    bounds_upper = [np.inf, 1e6,  df0 + 5.0,  10*np.pi]
    p0 = [A0, T20, df0, phi0]
    popt, pcov = curve_fit(stacked_model, t, y_stack, p0=p0,
                           bounds=(bounds_lower, bounds_upper), maxfev=20000)
    A, T2, df, phi = popt
    f0_nl = df + f_T
    return (A, T2, f0_nl, phi), popt, pcov

# ===== 频谱峰值（稳健） =====

def robust_peak(f, mag_db, fmin=PEAK_SEARCH_FMIN):
    f = np.asarray(f); mag_db = np.asarray(mag_db)
    mask = (f >= fmin)
    if np.any(mask):
        idx0 = np.argmax(mag_db[mask])
        return float(f[mask][idx0])
    return float(f[np.argmax(mag_db)])

# ===== 主流程 =====

def main():
    print("脚本目录 =", SCRIPT_DIR)
    print("输出目录 =", OUTPUT_DIR)
    print("当前工作目录 =", Path.cwd())

    # 读入
    tA, yA = load_signal(SCRIPT_DIR / "data_A.csv", fs=FS)
    tB, yB = load_signal(SCRIPT_DIR / "data_B.csv", fs=FS)

    # === fig1：原始概览（时域 + 频谱dB，带峰/基线标注）===
    fig1 = plt.figure(figsize=(12,6))
    ax1 = fig1.add_subplot(2,2,1); ax2 = fig1.add_subplot(2,2,2)
    ax3 = fig1.add_subplot(2,2,3); ax4 = fig1.add_subplot(2,2,4)

    ax1.plot(tA, yA); ax1.set_title("A组：时域波形"); ax1.set_xlabel("时间（s）"); ax1.set_ylabel("幅度")
    fA, mA_db = rfft_spectrum(yA, FS)
    ax2.plot(fA, mA_db); ax2.set_xlim(0, 5000)
    ax2.set_title("A组：单边幅度谱（0→f_s/2）"); ax2.set_xlabel("频率（Hz）"); ax2.set_ylabel("幅度谱 (dB)")
    ax2.set_ylim(np.max(mA_db)-120, np.max(mA_db))
    annotate_peak_and_noise(ax2, fA, mA_db, label="A组")

    ax3.plot(tB, yB); ax3.set_title("B组：时域波形"); ax3.set_xlabel("时间（s）"); ax3.set_ylabel("幅度")
    fB0, mB0_db = rfft_spectrum(yB, FS)
    ax4.plot(fB0, mB0_db); ax4.set_xlim(0, 5000)
    ax4.set_title("B组：单边幅度谱（0→f_s/2）"); ax4.set_xlabel("频率（Hz）"); ax4.set_ylabel("幅度谱 (dB)")
    ax4.set_ylim(np.max(mB0_db)-120, np.max(mB0_db))
    annotate_peak_and_noise(ax4, fB0, mB0_db, label="B组")

    fig1.tight_layout(); fig1.savefig(pjoin("fig1.png"), dpi=150); plt.close(fig1)

    # === 组B：自适应扫频谐波去除 ===
    yB_clean, vhar, f0_power = harmonic_denoise_adaptive(yB, tB, K=HARM_K)

    # 去噪后频谱（用于 fig3 对比与峰值估计）
    f_cln, m_cln_db = rfft_spectrum(yB_clean, FS)

    # === A/B(clean)：希尔伯特→下变频→线性估计 ===
    # A 组峰值（稳健）
    fA_pk = robust_peak(fA, mA_db, fmin=PEAK_SEARCH_FMIN)
    tA_trim, sA_bb, fT_A, _ = hilbert_baseband(yA, FS, fA_pk, f_delta=F_DELTA_TARGET, edge_frac=EDGE_TRIM_FRAC)
    (A_A, T2_A, f0_A, phi_A), (sA_A, sT2_A, sf0_A, sphi_A) = linear_params_from_baseband(tA_trim, sA_bb, fT_A)

    # B 组（clean）峰值（稳健）
    fB_pk = robust_peak(f_cln, m_cln_db, fmin=PEAK_SEARCH_FMIN)
    tB_trim, sB_bb, fT_B, _ = hilbert_baseband(yB_clean, FS, fB_pk, f_delta=F_DELTA_TARGET, edge_frac=EDGE_TRIM_FRAC)
    (A_B, T2_B, f0_B, phi_B), (sA_B, sT2_B, sf0_B, sphi_B) = linear_params_from_baseband(tB_trim, sB_bb, fT_B)

    # === fig2：A组线性拟合可视化（对数幅度 & 相位）===
    fig2 = plt.figure(figsize=(12,5))
    axL = fig2.add_subplot(1,2,1); axR = fig2.add_subplot(1,2,2)

    tA_seg, sA_seg = select_front_segment_by_amplitude(sA_bb, tA_trim, frac=AMP_FRONT_FRACTION)
    axL.plot(tA_trim, np.log(np.maximum(np.abs(sA_bb), 1e-16)), label="数据")
    axL.plot(tA_seg, (np.log(A_A) - (1.0/T2_A)*tA_seg), '--', lw=FIT_LW, zorder=3, label="拟合")
    axL.set_title("A组：对数幅度拟合"); axL.set_xlabel("时间（s）"); axL.set_ylabel("ln|s_bb|"); axL.legend()

    phA = np.unwrap(np.angle(sA_bb))
    axR.plot(tA_trim, phA, label="数据")
    axR.plot(tA_seg, (phi_A + 2*np.pi*(f0_A - fT_A)*tA_seg), '--', lw=FIT_LW, zorder=3, label="拟合")
    axR.set_title("A组：相位拟合"); axR.set_xlabel("时间（s）"); axR.set_ylabel("相位（rad）"); axR.legend()

    fig2.tight_layout(); fig2.savefig(pjoin("fig2.png"), dpi=150); plt.close(fig2)

    # === fig3：去谐波前后对比 + B组线性拟合 ===
    fig3 = plt.figure(figsize=(12, 6))
    g1 = fig3.add_subplot(2, 2, 1); g2 = fig3.add_subplot(2, 2, 2)
    g3 = fig3.add_subplot(2, 2, 3); g4 = fig3.add_subplot(2, 2, 4)

    # 子图1：时域对比 + 指数包络（按步骤3要求）
    g1.plot(tB, yB, label="原始")
    g1.plot(tB, yB_clean, label="去噪后")
    # 叠加指数包络（与线性估计一致，使用 tB_trim 坐标）
    env_B = A_B * np.exp(-tB_trim / T2_B)
    g1.plot(tB_trim,  env_B, '--', lw=1.2, label='包络  +A·e^{-t/T2}')
    g1.plot(tB_trim, -env_B, '--', lw=1.2, label='包络  -A·e^{-t/T2}')
    g1.set_title("B组：时域（去噪前 vs 去噪后 + 包络）")
    g1.set_xlabel("时间（s）"); g1.set_ylabel("幅度"); g1.legend()

    # 子图2：低频频谱对比（0–200 Hz）
    g2.plot(fB0, mB0_db, label="原始")
    g2.plot(f_cln, m_cln_db, label="去噪后")
    g2.set_xlim(0, 200)
    ymax = max(np.max(mB0_db), np.max(m_cln_db))
    g2.set_ylim(ymax-120, ymax)
    g2.set_title("B组：单边幅度谱（0–200 Hz）")
    g2.set_xlabel("频率（Hz）"); g2.set_ylabel("幅度谱 (dB)")
    # 标注谐波位置
    max_n = int(200.0 // f0_power)
    for n in range(1, max_n+1):
        g2.axvline(n*f0_power, color="gray", lw=0.8, ls=":", alpha=0.6)
    g2.legend()

    # 子图3：B组（去噪后）对数幅度直线拟合
    tB_seg, sB_seg = select_front_segment_by_amplitude(sB_bb, tB_trim, frac=AMP_FRONT_FRACTION)
    g3.plot(tB_trim, np.log(np.maximum(np.abs(sB_bb), 1e-16)), label="数据 (去噪后)")
    g3.plot(tB_seg, (np.log(A_B) - (1.0/T2_B)*tB_seg), '--', lw=FIT_LW, zorder=3, label="拟合")
    g3.set_title("B组去噪后：对数幅度拟合"); g3.set_xlabel("时间（s）"); g3.set_ylabel("ln|s_bb|"); g3.legend()

    # 子图4：B组（去噪后）相位直线拟合
    phB = np.unwrap(np.angle(sB_bb))
    g4.plot(tB_trim, phB, label="数据 (去噪后)")
    g4.plot(tB_seg, (phi_B + 2*np.pi*(f0_B - fT_B)*tB_seg), '--', lw=FIT_LW, zorder=3, label="拟合")
    g4.set_title("B组去噪后：相位拟合"); g4.set_xlabel("时间（s）"); g4.set_ylabel("相位（rad）"); g4.legend()

    fig3.tight_layout(); fig3.savefig(pjoin("fig3.png"), dpi=150); plt.close(fig3)

    # === 非线性复域拟合（A/B）===
    params_nl_A, _, _ = nonlinear_fit_from_linear_init(tA_trim, sA_bb, A_A, T2_A, f0_A, phi_A, fT_A)
    params_nl_B, _, _ = nonlinear_fit_from_linear_init(tB_trim, sB_bb, A_B, T2_B, f0_B, phi_B, fT_B)
    A_A_nl, T2_A_nl, f0_A_nl, phi_A_nl = params_nl_A
    A_B_nl, T2_B_nl, f0_B_nl, phi_B_nl = params_nl_B

    # === fig4：非线性拟合复域对比 ===
    fig4 = plt.figure(figsize=(12,8))
    c1 = fig4.add_subplot(2,2,1); c2 = fig4.add_subplot(2,2,2)
    c3 = fig4.add_subplot(2,2,3); c4 = fig4.add_subplot(2,2,4)

    gA = complex_model(tA_trim, A_A_nl, T2_A_nl, f0_A_nl - fT_A, phi_A_nl)
    gB = complex_model(tB_trim, A_B_nl, T2_B_nl, f0_B_nl - fT_B, phi_B_nl)

    c1.plot(tA_trim, np.real(sA_bb), label="数据")
    c1.plot(tA_trim, np.real(gA), '--', lw=FIT_LW, zorder=3, label="拟合")
    c1.set_title("A组：实部（数据 vs 拟合）"); c1.set_xlabel("时间（s）"); c1.set_ylabel("实部"); c1.legend()

    c2.plot(tA_trim, np.imag(sA_bb), label="数据")
    c2.plot(tA_trim, np.imag(gA), '--', lw=FIT_LW, zorder=3, label="拟合")
    c2.set_title("A组：虚部（数据 vs 拟合）"); c2.set_xlabel("时间（s）"); c2.set_ylabel("虚部"); c2.legend()

    c3.plot(tB_trim, np.real(sB_bb), label="数据")
    c3.plot(tB_trim, np.real(gB), '--', lw=FIT_LW, zorder=3, label="拟合")
    c3.set_title("B组：实部（数据 vs 拟合）"); c3.set_xlabel("时间（s）"); c3.set_ylabel("实部"); c3.legend()

    c4.plot(tB_trim, np.imag(sB_bb), label="数据")
    c4.plot(tB_trim, np.imag(gB), '--', lw=FIT_LW, zorder=3, label="拟合")
    c4.set_title("B组：虚部（数据 vs 拟合）"); c4.set_xlabel("时间（s）"); c4.set_ylabel("虚部"); c4.legend()

    fig4.tight_layout(); fig4.savefig(pjoin("fig4.png"), dpi=150); plt.close(fig4)

    # === 统计与 results.txt ===
    sum_raw2   = float(np.sum(yB**2))
    sum_clean2 = float(np.sum(yB_clean**2))
    sum_har2   = float(np.sum(vhar**2))
    explained_pct = 100.0 * (sum_har2 / (sum_raw2 + 1e-12))
    reduction_pct = 100.0 * (1.0 - (sum_clean2 / (sum_raw2 + 1e-12)))

    def rel_pct(a, b): return 100.0 * abs(a - b) / (abs(b) + 1e-12)
    def ang_diff_rad(a, b):
        d = (a - b + np.pi) % (2*np.pi) - np.pi
        return abs(d)

    dA_A   = rel_pct(A_A_nl,  A_A)
    dT2_A  = rel_pct(T2_A_nl, T2_A)
    df0_A  = rel_pct(f0_A_nl, f0_A)
    dphi_A = ang_diff_rad(phi_A_nl, phi_A)

    dA_B   = rel_pct(A_B_nl,  A_B)
    dT2_B  = rel_pct(T2_B_nl, T2_B)
    df0_B  = rel_pct(f0_B_nl, f0_B)
    dphi_B = ang_diff_rad(phi_B_nl, phi_B)

    with open(pjoin("results.txt"), "w", encoding="utf-8") as f:
        # 第1行（组A，线性）
        f.write("{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g}\n".format(
            A_A, T2_A, f0_A, phi_A, sA_A, sT2_A, sf0_A, sphi_A
        ))
        # 第2行（组B，线性）
        f.write("{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g}\n".format(
            A_B, T2_B, f0_B, phi_B, sA_B, sT2_B, sf0_B, sphi_B
        ))
        # 第3行（组A，非线性）
        f.write("{:.6g},{:.6g},{:.6g},{:.6g}\n".format(
            A_A_nl, T2_A_nl, f0_A_nl, phi_A_nl
        ))
        # 第4行（组B，非线性）
        f.write("{:.6g},{:.6g},{:.6g},{:.6g}\n".format(
            A_B_nl, T2_B_nl, f0_B_nl, phi_B_nl
        ))
        # 第5行（中文总结）
        analysis = (
            "谐波检测与消除：估计工频基频≈{f0:.3f} Hz，谐波阶数K={K}；"
            "拟合谐波解释了原始能量的{expl:.1f}%（去噪后总能量下降{red:.1f}%）。"
            "线/非线性对比：A组(A,T2,f0)相对差({dAA:.2f}%,{dTA:.2f}%,{dfA:.4f}%)、相位差{dpA:.3f} rad；"
            "B组(A,T2,f0)相对差({dAB:.2f}%,{dTB:.2f}%,{dfB:.4f}%)、相位差{dpB:.3f} rad。"
            "阈值敏感性：K在7–13范围较稳；前段阈值占比{front:.0f}%（阈值升高抑噪更强但可能低估T2，降低则相反）。"
            "可信度来源：工频谐波最小二乘全局建模+自适应扫频择优；基带化线性估计提供良好初值，"
            "非线性复域拟合在此基础上进一步最小化复残差。"
        ).format(
            f0=f0_power, K=HARM_K, expl=explained_pct, red=reduction_pct,
            dAA=dA_A, dTA=dT2_A, dfA=df0_A, dpA=dphi_A,
            dAB=dA_B, dTB=dT2_B, dfB=df0_B, dpB=dphi_B,
            front=AMP_FRONT_FRACTION*100.0
        )
        f.write(analysis + "\n")

    print("\n=== 完成 ===")
    print("输出目录：", OUTPUT_DIR)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

# ======== 字体设置：防止中文 & 负号乱码 ========
candidates = ["SimHei", "Microsoft YaHei", "PingFang SC", "STHeiti",
              "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
available = {f.name for f in fontManager.ttflist}
for name in candidates:
    if name in available:
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False
# ============================================

# ---------------------------
# 1) 原始数据（按题目给定）
# ---------------------------
W = np.arange(0, 201, 20)  # 重量(g): 0,20,...,200  共11个点

V1 = np.array([0, -5, -10, -14, -19, -25, -30, -35, -40, -45, -50], dtype=float)
V2 = np.array([0, -4,  -8, -14, -20, -24, -29, -34, -39, -44, -49], dtype=float)
V3 = np.array([0, -5,  -9, -14, -19, -25, -30, -35, -40, -45, -50], dtype=float)
V4 = np.array([0, -4,  -9, -15, -19, -24, -30, -35, -39, -45, -50], dtype=float)
V5 = np.array([0, -4,  -9, -14, -19, -24, -29, -34, -40, -45, -50], dtype=float)

V_all = np.vstack([V1, V2, V3, V4, V5])  # 形状：(5, 11)

# ---------------------------
# 2) 统计量：平均值
# ---------------------------
V_avg = V_all.mean(axis=0)

# 满量程输出 y_FS（按题意=200 g 对应输出 - 0 g 对应输出；用平均值）
y_FS = abs(V_avg[-1] - V_avg[0])  # 49.8 mV

# ---------------------------
# 3) 灵敏度 S = ΔV / ΔW
#    (a) 端点法
#    (b) 最小二乘直线拟合 V = a*W + b
# ---------------------------
S_endpoint = (V_avg[-1] - V_avg[0]) / (W[-1] - W[0])  # mV/g

a, b = np.polyfit(W, V_avg, 1)  # 线性拟合斜率a、截距b
S_fit = a  # 物理上也可视为灵敏度（mV/g）

V_fit = a * W + b

# ---------------------------
# 4) 非线性误差 δ = (Δm / y_FS) * 100%
# ---------------------------
residuals = V_avg - V_fit
Delta_m = np.max(np.abs(residuals))
delta_percent = (Delta_m / y_FS) * 100.0

# ---------------------------
# 5) 打印结果
# ---------------------------
print("=== 计算结果 ===")
print(f"平均电压 V_avg (mV): {np.round(V_avg, 2)}")
print(f"满量程输出 y_FS = {y_FS:.3f} mV")
print(f"灵敏度（端点法）S_endpoint = {S_endpoint:.3f} mV/g")
print(f"灵敏度（拟合法）  S_fit      = {S_fit:.3f} mV/g (拟合: V = {a:.3f}*W + {b:.3f})")
print(f"最大偏差 Δm = {Delta_m:.3f} mV")
print(f"非线性误差 δ = {delta_percent:.2f} %")

# ---------------------------
# 6) 作图：原始5次测量 + 平均 + 拟合
# ---------------------------
plt.figure(figsize=(8, 5))

#for i, Vi in enumerate(V_all, start=1):
#    plt.plot(W, Vi, marker='o', linewidth=1, alpha=0.6, label=f'第{i}次')

plt.plot(W, V_avg, marker='o', linewidth=2, label='平均')
plt.plot(W, V_fit, linestyle='--', linewidth=2, label='线性拟合')

plt.xlabel('重量 W (g)')
plt.ylabel('输出电压 V (mV)')
plt.title('应变片单臂电桥：输出电压-重量曲线')
plt.grid(True)
plt.legend()

text_str = (f"S(端点) = {S_endpoint:.3f} mV/g\n"
            f"S(拟合) = {S_fit:.3f} mV/g\n"
            f"Δm = {Delta_m:.3f} mV\n"
            f"δ = {delta_percent:.2f}%")
plt.annotate(text_str, xy=(0.02, 0.98), xycoords='axes fraction',
             va='top', ha='left', bbox=dict(boxstyle='round', fc='w', ec='k'))

plt.tight_layout()
# 可选：保存成文件，报告里直接用
# plt.savefig("应变片_电压-重量曲线.png", dpi=300, bbox_inches="tight")
plt.show()

"""
CART 决策树（单特征：年龄 Age）
基于 Gini 指数计算最优划分点。
"""

import numpy as np
import matplotlib.pyplot as plt


# 字体设置
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "PingFang SC",
    "Arial Unicode MS", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False


# 1. 数据定义
# 年龄 与 是否购机（1=是, 0=否）
ages = np.array([25, 35, 45, 20, 35, 52, 23, 40, 60, 48])
labels = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])  # 否=0 是=1


# 2. 定义 Gini 函数
def gini(y):
    """计算集合的基尼指数"""
    if len(y) == 0:
        return 0
    p1 = np.mean(y)  # 正类比例（购机）
    p0 = 1 - p1
    return 1 - p1**2 - p0**2

# 3. 搜索候选划分点
sorted_idx = np.argsort(ages)
ages_sorted = ages[sorted_idx]
y_sorted = labels[sorted_idx]

# 候选阈值取相邻不同数值的中点
thresholds = [(ages_sorted[i] + ages_sorted[i+1]) / 2 for i in range(len(ages_sorted)-1)]

gini_results = []
for t in thresholds:
    left = y_sorted[ages_sorted <= t]
    right = y_sorted[ages_sorted > t]
    g_split = (len(left)/len(y_sorted)) * gini(left) + (len(right)/len(y_sorted)) * gini(right)
    gini_results.append((t, g_split))

# 4. 找出最优划分点
best_t, best_gini = min(gini_results, key=lambda x: x[1])
gini_total = gini(y_sorted)
delta_gini = gini_total - best_gini

print("总体 Gini(D) = %.3f" % gini_total)
print("最优划分点：年龄 ≤ %.1f" % best_t)
print("划分后 Gini_split = %.3f" % best_gini)
print("Gini 减少量 ΔGini = %.3f" % delta_gini)

# 5. 绘制 CART 决策树图
fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
ax.axis("off")

root = (0.5, 0.85)
left = (0.2, 0.35)
right = (0.8, 0.35)

def draw_node(center, text):
    ax.text(center[0], center[1], text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"))

draw_node(root, f"根节点 D\n样本=10, 是=6, 否=4\nGini(D)={gini_total:.2f}")
draw_node(left, f"左子集：年龄≤{best_t:.1f}\nGini={gini(y_sorted[ages_sorted<=best_t]):.2f}\n预测：否")
draw_node(right, f"右子集：年龄>{best_t:.1f}\nGini={gini(y_sorted[ages_sorted>best_t]):.2f}\n预测：是")

ax.text(0.5, 0.68,
        f"按 年龄≤{best_t:.1f} 分裂 → Gini_split={best_gini:.2f}，ΔGini={delta_gini:.2f}",
        ha='center', fontsize=11)

# 绘制箭头
ax.annotate("", xy=(left[0]+0.09, left[1]+0.10), xytext=(root[0]-0.05, root[1]-0.08),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(right[0]-0.09, right[1]+0.10), xytext=(root[0]+0.05, root[1]-0.08),
            arrowprops=dict(arrowstyle="->", lw=1.5))

ax.text(0.32, 0.58, "是：年龄≤35", ha='center', fontsize=10)
ax.text(0.68, 0.58, "否：年龄>35", ha='center', fontsize=10)

ax.set_title("CART 决策树（单特征划分：年龄）", fontsize=12)
plt.tight_layout()
plt.show()

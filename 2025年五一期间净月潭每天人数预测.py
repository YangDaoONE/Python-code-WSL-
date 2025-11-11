import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
dates = ['5月1日', '5月2日', '5月3日', '5月4日', '5月5日']
percentages = [23.5, 26.7, 24.5, 15.2, 10.1]  # 百分比
visitors = [4.02, 4.56, 4.19, 2.60, 1.73]  # 单位：万人

x = np.arange(len(dates))

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 柱状图：预测游客量
bars = ax1.bar(x, visitors, color='skyblue', label='预测游客量（万人）')

# 设置y轴上限，留出空间
ax1.set_ylim(0, max(visitors) * 1.25)

# 添加数值标注（全部保持黑色，放在柱顶上方）
for bar, percent, num in zip(bars, percentages, visitors):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
             f'{num:.2f}万\n({percent:.1f}%)', ha='center', va='bottom',
             fontsize=10, color='black')

# 设置标签
ax1.set_xlabel('日期')
ax1.set_ylabel('预测游客量（万人）')
ax1.set_title('净月潭五一期间每日预测游客量（总量约17.1万人次）')
ax1.set_xticks(x)
ax1.set_xticklabels(dates)

# 显示网格
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图例
ax1.legend()

plt.tight_layout()
plt.show()

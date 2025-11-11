import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
years = ['2023年', '2024年', '2025年']
visitors = [11.00, 15.95, 17.10]  # 单位：万人次

# 计算同比增长
growth_rates = [None, (visitors[1] - visitors[0]) / visitors[0] * 100,
                      (visitors[2] - visitors[1]) / visitors[1] * 100]

x = np.arange(len(years))

# 创建图形
fig, ax = plt.subplots(figsize=(8, 5))

# 画折线图
ax.plot(x, visitors, marker='o', linestyle='-', color='blue', label='五一游客总量（万人次）')

# 添加数值标注
for i, (num, growth) in enumerate(zip(visitors, growth_rates)):
    ax.text(x[i], num + 0.5, f'{num:.2f}万', ha='center', fontsize=10, color='black')
    if growth is not None:
        ax.text((x[i-1] + x[i]) / 2, (visitors[i-1] + visitors[i]) / 2 + 0.7,
                f'同比增长约 {growth:.1f}%', color='green', ha='center', fontsize=10)

# 设置Y轴上限（加20%余量，防止标注超出）
ax.set_ylim(0, max(visitors) * 1.3)

# 设置标签
ax.set_xlabel('年份')
ax.set_ylabel('游客总量（万人次）')
ax.set_title('2023-2025年五一假期游客总量变化预测及同比增长')
ax.set_xticks(x)
ax.set_xticklabels(years)

# 显示网格
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

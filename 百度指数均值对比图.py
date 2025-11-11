import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
years = ['2023年', '2024年', '2025年']
categories = ['假期前7天均值', '假期前3天均值', '假期5天均值']

# 数值
data = [
    [23307, 31347, 107056],        # 2023年
    [19427, 29823, 71243],         # 2024年
    [10819, 17452, None],          # 2025年 (None 表示暂无数据)
]

# 转换为numpy数组
data_np = np.array(data, dtype=object)

# 绘图参数
x = np.arange(len(categories))
bar_width = 0.25

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 画柱状图
bars1 = ax.bar(x - bar_width, [d if d is not None else 0 for d in data_np[0]], width=bar_width, label='2023年')
bars2 = ax.bar(x, [d if d is not None else 0 for d in data_np[1]], width=bar_width, label='2024年')
bars3 = ax.bar(x + bar_width, [d if d is not None else 0 for d in data_np[2]], width=bar_width, label='2025年')

# 添加数值标注
for bars, year_data in zip([bars1, bars2, bars3], data_np):
    for bar, val in zip(bars, year_data):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width()/2, val + 1000, f'{val:,}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 1000, '暂无数据', ha='center', va='bottom', fontsize=9, color='red')

# 绘制趋势线
# 2023年趋势线
y_2023 = [d for d in data_np[0]]
ax.plot(x - bar_width, y_2023, marker='o', linestyle='-', color='C0')

# 2024年趋势线
y_2024 = [d for d in data_np[1]]
ax.plot(x, y_2024, marker='o', linestyle='-', color='C1')

# 2025年趋势线（仅连接有效数据点）
y_2025 = [d if d is not None else np.nan for d in data_np[2]]
ax.plot(x + bar_width, y_2025, marker='o', linestyle='-', color='C2')

# 设置图表属性
ax.set_xlabel('时间段')
ax.set_ylabel('百度指数均值')
ax.set_title('2023-2025年假期百度指数热度均值对比（含趋势线）')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()

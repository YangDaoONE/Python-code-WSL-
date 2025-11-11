import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
time_blocks = [
    '05:00–08:00',
    '08:00–11:00',
    '11:00–14:00',
    '14:00–17:00',
    '17:00–20:00',
    '20:00–23:00',
    '23:00–04:00'
]

# 对应小时编号和权重（注意：这里列出对应的小时编号）
weights = {
    '05:00–08:00': [(5, 0.20), (6, 0.30), (7, 0.50)],
    '08:00–11:00': [(8, 0.30), (9, 0.35), (10, 0.35)],
    '11:00–14:00': [(11, 0.30), (12, 0.40), (13, 0.30)],
    '14:00–17:00': [(14, 0.30), (15, 0.40), (16, 0.30)],
    '17:00–20:00': [(17, 0.30), (18, 0.40), (19, 0.30)],
    '20:00–23:00': [(20, 0.40), (21, 0.35), (22, 0.25)],
    '23:00–04:00': [(23, 0), (0, 0), (1, 0)]  # 夜间低峰
}

x = np.arange(len(time_blocks))
bottom = np.zeros(len(time_blocks))

fig, ax = plt.subplots(figsize=(10, 6))

# 最多三个小时一个时间段，画三层
for i in range(3):
    layer = [weights[block][i][1] for block in time_blocks]
    hours = [weights[block][i][0] for block in time_blocks]
    bars = ax.bar(x, layer, bottom=bottom, label=f'小时 {i+1}', edgecolor='black')
    
    # 标注小时编号
    for bar, hour, val in zip(bars, hours, layer):
        if val > 0:  # 只标非0的
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_y() + val / 2,
                f'{hour}时',
                ha='center', va='center', fontsize=9, color='black'
            )
    
    bottom += layer  # 更新底部位置

# 设置标签
ax.set_xlabel('时间段')
ax.set_ylabel('权重')
ax.set_title('按时间段的经验权重设定图（堆叠柱状图）')
ax.set_xticks(x)
ax.set_xticklabels(time_blocks)

# 添加图例（可选，也可以去掉）
ax.legend(title='对应小时顺序')

# 显示网格
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

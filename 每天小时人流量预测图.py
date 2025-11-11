import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# 设置中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 构建数据（这里以你截图为例，实际可读取CSV文件）
data = {
    '日期': [],
    '小时': [],
    '预测游客量': []
}

# 5天数据
days = ['5月1日', '5月2日', '5月3日', '5月4日', '5月5日']
hours = [f'{i:02d}:00' for i in range(24)]

# 粘贴的数据示例（只列一小部分，实际请替换为完整数据）
# 以下仅示例 5月1日 0-23 时数据：
values_5_1 = [0, 0, 0, 0, 0, 0.024, 0.036, 0.06, 0.237, 0.276, 0.276, 0.423,
              0.564, 0.423, 0.3, 0.4, 0.3, 0.147, 0.196, 0.147, 0.084, 0.073, 0.052, 0]
values_5_2 = [0, 0, 0, 0, 0, 0.026, 0.039, 0.065, 0.267, 0.311, 0.311, 0.474,
              0.632, 0.474, 0.351, 0.468, 0.351, 0.168, 0.224, 0.168, 0.092, 0.081, 0.058, 0]
values_5_3 = [0, 0, 0, 0, 0, 0.026, 0.039, 0.065, 0.276, 0.322, 0.322, 0.417,
              0.556, 0.417, 0.297, 0.396, 0.297, 0.165, 0.22, 0.165, 0.088, 0.077, 0.055, 0]
values_5_4 = [0, 0, 0, 0, 0, 0.016, 0.024, 0.04, 0.153, 0.178, 0.178, 0.273,
              0.364, 0.273, 0.195, 0.26, 0.195, 0.096, 0.128, 0.096, 0.052, 0.045, 0.033, 0]
values_5_5 = [0, 0, 0, 0, 0, 0.01, 0.015, 0.025, 0.105, 0.122, 0.122, 0.183,
              0.244, 0.183, 0.129, 0.172, 0.129, 0.063, 0.084, 0.063, 0.032, 0.028, 0.02, 0]

# 填充数据到字典
for day, vals in zip(days, [values_5_1, values_5_2, values_5_3, values_5_4, values_5_5]):
    for h, val in zip(hours, vals):
        data['日期'].append(day)
        data['小时'].append(h)
        data['预测游客量'].append(val)

# 转换为DataFrame
df = pd.DataFrame(data)

# 画图
fig, ax = plt.subplots(figsize=(12, 6))

for day in days:
    df_day = df[df['日期'] == day]
    ax.plot(df_day['小时'], df_day['预测游客量'], marker='o', label=day)

# 设置标签
ax.set_xlabel('小时')
ax.set_ylabel('预测游客量（万人）')
ax.set_title('五一期间每日每小时游客预测量')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

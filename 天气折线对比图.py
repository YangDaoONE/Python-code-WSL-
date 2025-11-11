import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体，防止乱码（适用于Windows）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 日期对齐调整：2023.4.29 对齐 2024.5.1 和 2025.5.1
dates_aligned = ['第1天', '第2天', '第3天', '第4天', '第5天']

# 2023年数据（从4.29开始）
temp_max_2023 = [11, 11, 18, 24, 25]
temp_min_2023 = [3, 6, 8, 14, 12]

# 2024年数据（从5.1开始）
temp_max_2024 = [24, 26, 30, 27, 22]
temp_min_2024 = [13, 14, 15, 16, 13]

# 2025年数据（从5.1开始）
temp_max_2025 = [13, 17, 13, 16, 16]
temp_min_2025 = [4, 6, 4, 4, 5]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制最高气温
plt.plot(dates_aligned, temp_max_2023, marker='o', label='2023年最高气温')
plt.plot(dates_aligned, temp_max_2024, marker='o', label='2024年最高气温')
plt.plot(dates_aligned, temp_max_2025, marker='o', label='2025年最高气温')

# 绘制最低气温
plt.plot(dates_aligned, temp_min_2023, marker='s', label='2023年最低气温')
plt.plot(dates_aligned, temp_min_2024, marker='s', label='2024年最低气温')
plt.plot(dates_aligned, temp_min_2025, marker='s', label='2025年最低气温')

# 标注最高和最低气温
all_max = {
    '2023年': max(temp_max_2023),
    '2024年': max(temp_max_2024),
    '2025年': max(temp_max_2025)
}
all_min = {
    '2023年': min(temp_min_2023),
    '2024年': min(temp_min_2024),
    '2025年': min(temp_min_2025)
}

# 添加标注
for year, max_val in all_max.items():
    idx = eval(f"temp_max_{year[:4]}").index(max_val)
    plt.text(dates_aligned[idx], max_val + 0.5, f'{year}最高: {max_val}℃', ha='center', color='red')

for year, min_val in all_min.items():
    idx = eval(f"temp_min_{year[:4]}").index(min_val)
    plt.text(dates_aligned[idx], min_val - 1.5, f'{year}最低: {min_val}℃', ha='center', color='blue')

# 图形美化
plt.title('2023/2024/2025年五一期间每日气温变化趋势（2023.4.29与2024.5.1对齐）')
plt.xlabel('日期（对齐后的第几天）')
plt.ylabel('气温（℃）')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

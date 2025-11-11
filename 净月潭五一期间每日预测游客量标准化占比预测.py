import matplotlib.pyplot as plt
import matplotlib

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
labels = ['5月1日', '5月2日', '5月3日', '5月4日', '5月5日']
sizes = [23.5, 26.7, 24.5, 15.2, 10.1]  # 标准化占比

# 设置颜色（可选，不设置则用默认颜色）
# colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 画饼状图
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10}
)

# 设置标题
ax.set_title('净月潭五一期间每日预测游客量标准化占比')

# 保证饼图是圆形
ax.axis('equal')

plt.tight_layout()
plt.show()

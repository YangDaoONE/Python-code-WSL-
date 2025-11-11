import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

# ========== 设置中文字体 ==========
# 这里使用 SimHei（黑体），适用于大多数 Windows 系统
rcParams['font.sans-serif'] = ['SimHei']   # 显示中文
rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# ========== 加载背景图 ==========
background_path = r"C:\Users\YangDaoONE\Downloads\岗位分布.png"
background = Image.open(background_path)

fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(background)

# ========== 示例岗位坐标 ==========
# 注意：请替换为你自己的实际坐标
# 引导岗（蓝色三角形） - 共30处
guidance_posts = [
    (100, 150), (130, 180), (160, 210), (190, 240), (220, 270),
    (250, 300), (280, 330), (310, 360), (340, 390), (370, 420),
    (400, 450), (430, 480), (460, 510), (490, 540), (520, 570),
    (550, 600), (580, 630), (610, 660), (640, 690), (670, 720),
    (700, 750), (730, 780), (760, 810), (790, 840), (820, 870),
    (850, 900), (880, 930), (910, 960), (940, 990), (970, 1020)
]

# 咨询岗（橙色方块） - 共15处
consulting_posts = [
    (200, 250), (250, 300), (300, 350), (350, 400), (400, 450),
    (450, 500), (500, 550), (550, 600), (600, 650), (650, 700),
    (700, 750), (750, 800), (800, 850), (850, 900), (900, 950)
]

# 应急岗（红色叉号） - 共36处
emergency_posts = [
    (120, 180), (150, 220), (180, 260), (210, 300), (240, 340), (270, 380),
    (300, 420), (330, 460), (360, 500), (390, 540), (420, 580), (450, 620),
    (480, 660), (510, 700), (540, 740), (570, 780), (600, 820), (630, 860),
    (660, 900), (690, 940), (720, 980), (750, 1020), (780, 1060), (810, 1100),
    (840, 1140), (870, 1180), (900, 1220), (930, 1260), (960, 1300), (990, 1340),
    (1020, 1380), (1050, 1420), (1080, 1460), (1110, 1500), (1140, 1540), (1170, 1580)
]

# ========== 绘制岗位 ==========
for (x, y) in guidance_posts:
    ax.scatter(x, y, marker='^', s=100, color='blue', label='引导岗位 (Guidance Post)')

for (x, y) in consulting_posts:
    ax.scatter(x, y, marker='s', s=100, color='orange', label='咨询岗位 (Consulting Post)')

for (x, y) in emergency_posts:
    ax.scatter(x, y, marker='x', s=100, color='red', label='应急岗位 (Emergency Post)')

# 图例去重处理
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

# 标题和格式
plt.title('净月潭国家森林公园志愿者岗位布设图\n蓝色三角形=引导岗，橙色方块=咨询岗，红色叉号=应急岗', fontsize=14)
plt.axis('off')

# 保存和展示
plt.savefig('岗位分布标注版.png', dpi=300)
plt.show()

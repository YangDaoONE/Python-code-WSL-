import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import numpy as np

# 注册字体文件
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
prop_sc = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")

# 设置 Matplotlib 基本参数
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 兜底，不影响中文显示

# 绘图时手动指定 fontproperties
x = np.linspace(0, 2*np.pi, 200)
plt.plot(x, np.sin(x), label="正弦波 sin(x) 负号-OK", )
plt.title("中文标题 - 简体 SC", fontproperties=prop_sc)
plt.xlabel("时间 (秒)", fontproperties=prop_sc)
plt.ylabel("幅度", fontproperties=prop_sc)

# 图例文字也指定字体
leg = plt.legend(prop=prop_sc)
for t in leg.get_texts():
    t.set_fontproperties(prop_sc)

plt.tight_layout()
plt.savefig("font_check.png", dpi=150)
print("✅ 已生成 font_check.png")

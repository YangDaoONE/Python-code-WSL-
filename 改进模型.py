import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1️⃣ 生成模拟数据
np.random.seed(42)
dates = pd.date_range(start="2025-04-26", end="2025-05-05", freq="H")
df = pd.DataFrame({"datetime": dates})
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=周一 ... 6=周日

# 假期标记（5月1–5日）
holidays = pd.date_range("2025-05-01", "2025-05-05", freq="D")
df["is_holiday"] = df["datetime"].dt.normalize().isin(holidays).astype(int)
df["is_weekend"] = ((df["day_of_week"] >= 5) & (df["is_holiday"] == 0)).astype(int)

# 特殊活动标记：假设5月1-3每天11:00有活动
df["event_flag"] = 0
for d in ["2025-05-01 11:00", "2025-05-02 11:00", "2025-05-03 11:00"]:
    df.loc[df["datetime"] == d, "event_flag"] = 1

# 天气模拟：温度随时间变化，5月3日降雨
df["temperature"] = 20 + 5 * np.sin(np.linspace(0, 2 * np.pi, len(df))) + np.random.randn(len(df)) * 2
df["rainfall"] = 0.0
rain_idx = (df["datetime"].dt.date == pd.to_datetime("2025-05-03").date()) & (df["hour"] >= 9) & (df["hour"] <= 17)
df.loc[rain_idx, "rainfall"] = np.random.uniform(1, 5, rain_idx.sum())

# 去年同期客流（用当前数据的缩减版模拟）
df_last_year = df.copy()
df_last_year["datetime"] = df_last_year["datetime"] - pd.DateOffset(years=1)

def simulate_visitors(data, holiday_coef=1000, weekend_coef=100, temp_coef=10, rain_coef=-100, event_coef=300,
                      prev_coef=0.3, lastyear_coef=0.5, base=50):
    visitors = []
    prev = 0
    for _, row in data.iterrows():
        y = (base
             + holiday_coef * row["is_holiday"]
             + weekend_coef * row["is_weekend"]
             + temp_coef * row["temperature"]
             + rain_coef * row["rainfall"]
             + event_coef * row["event_flag"]
             + lastyear_coef * row.get("last_year_visitors", 0)
             + prev_coef * prev)
        y += np.random.randn() * 50
        if row["hour"] >= 23 or row["hour"] <= 4:
            y = 0
        prev = max(y, 0)
        visitors.append(max(int(round(y)), 0))
    return visitors

# 去年客流生成
df_last_year["visitors"] = simulate_visitors(df_last_year, holiday_coef=800, weekend_coef=80,
                                             temp_coef=8, rain_coef=-80, event_coef=240)

# 合并去年数据
df = df.merge(df_last_year[["datetime", "visitors"]].rename(columns={"datetime": "curr_datetime", "visitors": "last_year_visitors"}),
              how="left", left_on="datetime", right_on="curr_datetime")
df["last_year_visitors"] = df["last_year_visitors"].fillna(0)

# 本年客流生成
df["visitors"] = simulate_visitors(df, lastyear_coef=0.5)
df["prev_visitors"] = df["visitors"].shift(1).fillna(0)

# 最终特征
features = ["is_holiday", "is_weekend", "hour", "temperature", "rainfall", "event_flag", "prev_visitors", "last_year_visitors"]

# 2️⃣ 训练回归模型
X = df[features]
y = df["visitors"]
model = LinearRegression().fit(X, y)

# 打印回归系数
coef_df = pd.DataFrame({"特征": features, "系数": model.coef_.round(2)})
coef_df.loc[len(coef_df)] = ["Intercept (截距)", round(model.intercept_, 2)]
print("回归系数：\n", coef_df)

# 3️⃣ 预测与可视化
df["predicted"] = model.predict(X)

# 绘制实际 vs 预测曲线（示例：4月30日至5月3日）
plot_df = df[(df["datetime"] >= "2025-04-30") & (df["datetime"] < "2025-05-04")]
plt.figure(figsize=(10, 5))
plt.plot(plot_df["datetime"], plot_df["visitors"], label="实际客流")
plt.plot(plot_df["datetime"], plot_df["predicted"], label="预测客流", linestyle="--")
plt.xlabel("时间")
plt.ylabel("客流量")
plt.title("实际 vs 预测（部分时段）")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4️⃣ 残差分析
residuals = df["visitors"] - df["predicted"]
print(f"平均残差: {residuals.mean():.2f}，残差标准差: {residuals.std():.2f}")
night_mask = (df["hour"] <= 4) | (df["hour"] >= 23)
print(f"夜间时段平均残差: {residuals[night_mask].mean():.2f}")

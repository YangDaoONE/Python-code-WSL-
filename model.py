import pandas as pd
from datetime import datetime, timedelta

# === 1. 定义预测时间范围（2025年五一假期5月1日-5日，每小时） ===
start_date = "2025-05-01"
end_date = "2025-05-05"
# 生成从开始日期到结束日期（包含）的每小时时间索引
hourly_index = pd.date_range(start_date, end_date + " 23:00", freq='H')

# === 2. 输入变量数据结构准备 ===
# 天气预报数据：键为时间戳，值为天气信息（如天气状况和对客流的影响系数）
weather_forecast = {
    # 例如：假设5月3日中午有降雨，影响系数设为0.5（客流减半）
    datetime(2025, 5, 3, 12): {"condition": "rain", "impact_factor": 0.5}
    # 可以扩展为每小时的天气预报数据字典
}

# 活动日程数据：列表包含假期期间景区特别活动的时间点及预期额外吸引的游客数
events = [
    {"time": datetime(2025, 5, 2, 14, 0), "expected_additional_visitors": 500}  # 例：5月2日14点有活动，预计增加500人
    # 可添加更多活动
]

# （可选）外地游客比例等其它特征：可以根据需要设计，例如按日期的外地游客占比，用于调整基准客流
# non_local_ratio = {"2025-05-01": 0.3, ...}  # 例：假设5月1日外地游客占30%

# === 3. 定义基准日客流分布模式（小时比例） ===
# 基于经验规则假设一个典型假期日内不同时段的客流占比模式（例如上午适中、中午最高、傍晚下降、夜间极低）
baseline_daily_pattern = [0.0] * 24
for h in range(24):
    if 8 <= h < 10:
        baseline_daily_pattern[h] = 0.08  # 上午8-9点客流开始上升
    elif 10 <= h < 14:
        baseline_daily_pattern[h] = 0.10  # 10点到13点达到高峰（中午时段较高）
    elif 14 <= h < 17:
        baseline_daily_pattern[h] = 0.08  # 午后稍有回落
    elif 17 <= h < 19:
        baseline_daily_pattern[h] = 0.05  # 傍晚时段逐渐减少
    else:
        baseline_daily_pattern[h] = 0.01  # 夜间或清晨客流很低（公园可能闭园）
# 将模式归一化为比例和为1（方便按日总客流量分配）
total = sum(baseline_daily_pattern)
baseline_daily_pattern = [x / total for x in baseline_daily_pattern]

# === 4. 定义每天的基准游客总量（根据经验、类比或假设） ===
# 这里采用类比方法假设五一假期各天的游客总量（可根据历年趋势调整）：
daily_base_visitors = {
    "2025-05-01": 8000,   # 假设5月1日游客量 8000 人
    "2025-05-02": 12000,  # 假设5月2日游客量 12000 人（假期中段上升）
    "2025-05-03": 15000,  # 假设5月3日游客量 15000 人（峰值）
    "2025-05-04": 10000,  # 假设5月4日游客量 10000 人
    "2025-05-05": 6000    # 假设5月5日游客量 6000 人（假期最后一天回落）
}

# （将来可在此基础上，通过历史数据训练模型来确定 daily_base_visitors 或调整模式）

# === 5. 根据基准分布和调整因子计算每小时游客量预测 ===
forecast = []  # 将收集每小时预测结果的列表
for timestamp in hourly_index:
    day_str = timestamp.strftime("%Y-%m-%d")
    hour = timestamp.hour
    # 取当天基准总客流，如果当天不在预测范围则为0
    base_total = daily_base_visitors.get(day_str, 0)
    # 按基准日内分布比例分配到该小时的基准客流量
    base_count = base_total * baseline_daily_pattern[hour]
    # 根据天气因素调整客流：如果该小时有天气影响数据，则乘以影响系数
    if timestamp in weather_forecast:
        impact = weather_forecast[timestamp].get("impact_factor", 1.0)
        base_count *= impact
    # 根据活动因素调整客流：如果该小时有活动，叠加额外游客量
    for evt in events:
        # 若事件时间与当前小时匹配，则增加对应的游客量
        if evt["time"].year == timestamp.year and evt["time"].hour == timestamp.hour \
           and evt["time"].date() == timestamp.date():
            base_count += evt["expected_additional_visitors"]
    # （可选）根据外地游客占比等特征调整：例如外地游客多的日子整体上调客流等
    # ratio = non_local_ratio.get(day_str, None)
    # if ratio is not None:
    #     base_count *= (1 + (ratio - 0.5))  # 简例：若外地游客占比较高（>50%），上调客流

    # 将结果添加到列表
    forecast.append({"time": timestamp, "predicted_visitors": int(base_count)})

# 将预测结果转换为 DataFrame（方便后续处理或输出）
forecast_df = pd.DataFrame(forecast)
# 打印前几行结果查看结构
print(forecast_df.head())

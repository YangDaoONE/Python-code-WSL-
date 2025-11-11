import requests
from bs4 import BeautifulSoup
import pandas as pd

def crawl_history_weather(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://lishi.tianqi.com/',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    resp = requests.get(url, headers=headers, proxies={})
    resp.encoding = 'utf-8'

    soup = BeautifulSoup(resp.text, 'html.parser')
    weather_section = soup.find('div', class_='tian_three')
    data = []

    if not weather_section:
        print(f"❌ 没找到天气数据，请检查页面结构或是否触发了反爬虫。URL: {url}")
        return []

    lis = weather_section.find_all('li')
    for li in lis:
        divs = li.find_all('div')
        if len(divs) < 5:
            continue  # 跳过不完整数据

        date_raw = divs[0].get_text(strip=True)
        date = date_raw.split(' ')[0]  # 提取日期部分
        max_temp = divs[1].get_text(strip=True)
        min_temp = divs[2].get_text(strip=True)
        weather = divs[3].get_text(strip=True)
        wind = divs[4].get_text(strip=True)

        print(f"{date} | {max_temp} | {min_temp} | {weather} | {wind}")

        data.append({
            '日期': date,
            '最高气温': max_temp,
            '最低气温': min_temp,
            '天气': weather,
            '风力风向': wind,
        })
    
    return data

# 爬取2023年数据
url_202304 = 'https://lishi.tianqi.com/changchun/202304.html'
url_202305 = 'https://lishi.tianqi.com/changchun/202305.html'
data_202304 = crawl_history_weather(url_202304)
data_202305 = crawl_history_weather(url_202305)

# 爬取2024年数据
url_202405 = 'https://lishi.tianqi.com/changchun/202405.html'
data_202405 = crawl_history_weather(url_202405)

# 合并数据
all_data = data_202304 + data_202305 + data_202405
df = pd.DataFrame(all_data)

# 筛选目标日期
wanted_dates = [
    '2023-04-29', '2023-04-30', '2023-05-01', '2023-05-02', '2023-05-03',
    '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05'
]
df_filtered = df[df['日期'].isin(wanted_dates)]

# 保存文件
df_filtered.to_csv('长春2023_2024五一历史天气.csv', index=False, encoding='utf-8-sig')
df_filtered.to_excel('长春2023_2024五一历史天气.xlsx', index=False)

print("\n✅ 2023年+2024年五一历史天气数据已保存！")

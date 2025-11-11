from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time

# ===============================
# 1️⃣ 设置 Selenium 选项
# ===============================
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

chromedriver_path = r"C:\Users\YangDaoONE\Desktop\Project\code\Python\chromedriver-win64\chromedriver.exe"
service = Service(chromedriver_path)

driver = webdriver.Chrome(service=service, options=chrome_options)

# 打开网页
url = 'https://weather.cma.cn/web/weather/54161.html'
driver.get(url)
time.sleep(5)

# ===============================
# 2️⃣ 爬取总览数据
# ===============================
print("\n====5.1开始的5天天气总览====")
day_list = driver.find_elements('css selector', '#dayList > div.pull-left.day')

overview_data = []
hourly_data = []

for day_element in day_list:
    items = day_element.find_elements('css selector', '.day-item')
    date_info = items[0].text.strip().replace('\n', ' ')
    date_part = date_info.strip()[-5:]

    if '05/01' <= date_part <= '05/05':
        day_weather = items[2].text.strip()
        day_wind_dir = items[3].text.strip()
        day_wind_power = items[4].text.strip()

        bardiv = items[5]
        high_temp = bardiv.find_element('css selector', '.high').text.strip()
        low_temp = bardiv.find_element('css selector', '.low').text.strip()

        night_weather = items[7].text.strip()
        night_wind_dir = items[8].text.strip()
        night_wind_power = items[9].text.strip()

        print(f"{date_info}: {day_weather}/{night_weather}, {high_temp}/{low_temp}")

        overview_data.append({
            '日期': date_info,
            '白天现象': day_weather,
            '白天风向': day_wind_dir,
            '白天风力': day_wind_power,
            '夜间现象': night_weather,
            '夜间风向': night_wind_dir,
            '夜间风力': night_wind_power,
            '最高温': high_temp,
            '最低温': low_temp
        })

# ===============================
# 3️⃣ 爬取小时数据
# ===============================
print("\n====5.1开始的5天小时数据====")

# 天气图标映射表（根据网站的图标文件名）
icon_map = {
    'w0.png': '晴',
    'w1.png': '多云',
    'w2.png': '阴',
    'w3.png': '阵雨',
    'w4.png': '雷阵雨',
    'w7.png': '小雨',
    'w8.png': '中雨',
    # 可根据实际补充其他
}

for idx, day_element in enumerate(day_list):
    date_text = day_element.find_element('css selector', '.day-item').text.strip().replace('\n', ' ')
    date_part = date_text.strip()[-5:]

    if '05/01' <= date_part <= '05/05':
        print(f"\n👉 点击：{date_text}")
        # 点击块触发小时数据刷新
        driver.execute_script("arguments[0].click();", day_element)
        time.sleep(2)

        # 解析网页
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # 查找所有小时数据表
        hour_tables = soup.find_all('table', class_='hour-table')

        # 找到当前“显示”的那个表（style 不含 'display: none'）
        visible_table = None
        for table in hour_tables:
            style = table.get('style', '')
            if 'display: none' not in style:
                visible_table = table
                break

        if visible_table is None:
            print(f"⚠ {date_text} 没找到可见的小时数据表")
            continue

        # 解析小时数据表
        rows = visible_table.find_all('tr')
        if len(rows) < 9:
            print(f"⚠ {date_text} 小时表结构异常")
            continue

        times = [td.get_text(strip=True) for td in rows[0].find_all('td')[1:]]
        
        # 解析天气（从图片文件名识别）
        weather_tds = rows[1].find_all('td')[1:]
        weathers = []
        for td in weather_tds:
            img = td.find('img')
            if img:
                icon_file = img['src'].split('/')[-1]
                weathers.append(icon_map.get(icon_file, '未知'))
            else:
                weathers.append(td.get_text(strip=True))

        temps = [td.get_text(strip=True) for td in rows[2].find_all('td')[1:]]
        rainfalls = [td.get_text(strip=True) for td in rows[3].find_all('td')[1:]]
        windspeeds = [td.get_text(strip=True) for td in rows[4].find_all('td')[1:]]
        winddirs = [td.get_text(strip=True) for td in rows[5].find_all('td')[1:]]
        pressures = [td.get_text(strip=True) for td in rows[6].find_all('td')[1:]]
        humidities = [td.get_text(strip=True) for td in rows[7].find_all('td')[1:]]
        cloudcovers = [td.get_text(strip=True) for td in rows[8].find_all('td')[1:]]

        for i in range(len(times)):
            hourly_data.append({
                '日期': date_text,
                '时间': times[i],
                '天气': weathers[i],
                '温度': temps[i],
                '降水': rainfalls[i],
                '风速': windspeeds[i],
                '风向': winddirs[i],
                '气压': pressures[i],
                '湿度': humidities[i],
                '云量': cloudcovers[i]
            })

        print(f"✅ 已获取 {date_text} 的小时数据，共 {len(times)} 条")

# ===============================
# 4️⃣ 保存数据
# ===============================
overview_df = pd.DataFrame(overview_data)
overview_df.to_excel('五天天气总览_完整版.xlsx', index=False)

hourly_df = pd.DataFrame(hourly_data)
hourly_df.to_excel('五天小时数据_完整版.xlsx', index=False)

print("\n🎉 已保存完整5.1~5.5的天气总览和小时数据！")
driver.quit()

import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.edge.options import Options

# === 用户账号密码 ===
userid = 'liaojh1723'
password = '0827LJHydy660'
url = 'https://ievaluate.jlu.edu.cn/'

# === Edge 浏览器配置 ===
options = Options()
options.add_argument('--ignore-certificate-errors')
options.add_experimental_option("detach", True)

driver = webdriver.Edge(options=options)
wait = WebDriverWait(driver, 10)

# === 登录系统 ===
driver.get(url)
wait.until(EC.presence_of_element_located((By.XPATH, '//*[@type="text"]'))).send_keys(userid)
driver.find_element(By.XPATH, '//*[@type="password"]').send_keys(password)
driver.find_element(By.XPATH, '//*[@type="button"]').click()

# === 切换到最新窗口 ===
def switch_to_latest_window():
    driver.switch_to.window(driver.window_handles[-1])
    print("\n🪟 已切换窗口：", driver.title)

# === 自动评教表单填写 ===
def fill_evaluation():
    # 输入评分项
    score_inputs = driver.find_elements(By.CLASS_NAME, "ant-input-number-input")
    for i, input_box in enumerate(score_inputs):
        try:
            max_score = int(float(input_box.get_attribute("aria-valuemax")))
            score = random.choice([max_score - 1, max_score])
            driver.execute_script("""
                arguments[0].value = arguments[1];
                arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
                arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
            """, input_box, str(score))
            print(f"📝 输入评分 | 第 {i+1} 题：满分 {max_score}，打分 {score}")
        except Exception as e:
            print(f"⚠️ 输入评分失败：第 {i+1} 题，错误：{e}")

    # 滑动条评分项
    slider_blocks = driver.find_elements(By.CLASS_NAME, "index__custom-slider--LQbrv")
    for j, slider in enumerate(slider_blocks):
        try:
            ActionChains(driver).move_to_element_with_offset(slider, 180, 0).click().perform()
            print(f"🎚 滑动评分 | 第 {j+1} 题滑块点击完成")
            time.sleep(0.2)
        except Exception as e:
            print(f"⚠️ 滑动评分失败：第 {j+1} 题，错误：{e}")

    # 填写文字评价
    try:
        textarea = driver.find_element(By.XPATH, '//textarea')
        textarea.clear()
        textarea.send_keys("老师讲得非常清晰，课程内容充实，受益匪浅！")
        print("✍️ 已填写文字评价")
    except:
        print("📝 未检测到文字评价项，跳过")

# === 自动评教主循环 ===
try:
    while True:
        time.sleep(1)
        evaluate_buttons = driver.find_elements(By.XPATH, '//table/tbody/tr/td[7]/span')
        if not evaluate_buttons:
            print("\n✅ 所有课程已评教完毕。")
            break

        evaluate_buttons[0].click()
        time.sleep(1)
        switch_to_latest_window()

        wait.until(EC.element_to_be_clickable((By.XPATH, '//table/tbody/tr[1]/td[7]/span'))).click()
        time.sleep(1)
        switch_to_latest_window()

        fill_evaluation()

        submit_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, '//span[contains(text(), "提 交")]/ancestor::button')
        ))
        submit_btn.click()
        print("\n✅ 点击提交按钮")

        try:
            confirm_btn = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//span[contains(text(), "确 定") or contains(text(), "确定")]/ancestor::button')
            ))
            confirm_btn.click()
            print("🎉 成功提交问卷")
        except TimeoutException:
            print("⚠️ 没有检测到确认弹窗，可能已自动提交")

        try:
            next_course_btn = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//button[span[text()="下一课程"]]')
            ))
            next_course_btn.click()
            print("➡️ 已点击下一课程，继续处理")
            switch_to_latest_window()
            time.sleep(1)
            fill_evaluation()  # 新增：下一课程后也重新填写问卷

            submit_btn = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//span[contains(text(), "提 交")]/ancestor::button')
            ))
            submit_btn.click()
            print("✅ 提交下一课程问卷")

            try:
                confirm_btn = wait.until(EC.element_to_be_clickable(
                    (By.XPATH, '//span[contains(text(), "确 定") or contains(text(), "确定")]/ancestor::button')
                ))
                confirm_btn.click()
                print("🎉 下一课程提交成功")
            except TimeoutException:
                print("⚠️ 下一课程未检测到确认弹窗")

        except TimeoutException:
            print("🔚 未检测到“下一课程”按钮，关闭当前窗口返回主页面。")
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            else:
                print("❌ 会话已失效，无法返回主窗口。")
                break

except Exception as e:
    print("\n❌ 脚本运行中出错：", e)
    try:
        driver.save_screenshot("error.png")
        print("📸 错误截图已保存：error.png")
    except:
        print("⚠️ 无法截图，浏览器已关闭或失效。")

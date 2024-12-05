from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import requests

# 검색할 키워드 설정
search_query = "영수증"  # 원하는 검색어로 변경

# 이미지 저장할 디렉토리 설정
save_dir = "downloaded_images"
os.makedirs(save_dir, exist_ok=True)

# Chrome 드라이버 설정
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# 구글 이미지 검색 페이지 열기
driver.get(f"https://www.google.com/search?hl=ko&tbm=isch&q={search_query}")

# 스크롤 내려서 이미지 로딩
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # 이미지 요소들 찾기
    images = driver.find_elements(By.XPATH, '//img[@class="Q4LuWd"]')
    
    # 이미지 URL 추출
    for img in images:
        try:
            # src 대신 data-src 속성을 사용해보기
            img_url = img.get_attribute('src') or img.get_attribute('data-src')
            if img_url and img_url.startswith('http'):
                # 이미지 다운로드
                img_data = requests.get(img_url).content
                img_name = os.path.join(save_dir, f"{search_query}_{len(os.listdir(save_dir))}.jpg")
                with open(img_name, 'wb') as handler:
                    handler.write(img_data)
                print(f"Downloaded: {img_name}")  # 다운로드 확인 메시지
        except Exception as e:
            print(f"Error: {e}")

    # 스크롤 내리기
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 잠시 대기 후 스크롤 높이 확인
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    # 더 이상 스크롤 할 수 없으면 종료
    if new_height == last_height:
        break
    last_height = new_height

# 드라이버 종료
driver.quit()

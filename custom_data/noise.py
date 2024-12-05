import cv2
import numpy as np
import os
from pathlib import Path

# 이미지 파일이 있는 디렉토리 경로
image_dir = r'C:\Users\User\Desktop\prprprpr\concat_dataset\5_background\pre'

# 증강된 이미지를 저장할 디렉토리 경로
save_directory = r'C:\Users\User\Desktop\prprprpr\concat_dataset\5_background\pre'
os.makedirs(save_directory, exist_ok=True)

# 노이즈 추가 함수 정의 (가우시안 노이즈)
def add_gaussian_noise(image, mean=0, var_range=(50, 150)):
    """
    이미지에 가우시안 노이즈를 추가하는 함수.

    Parameters:
        image (numpy.ndarray): 입력 이미지 (RGB).
        mean (float): 노이즈의 평균. 기본값은 0.
        var_range (tuple): 노이즈의 분산 범위. 기본값은 (50, 150).

    Returns:
        tuple: 노이즈가 추가된 이미지와 사용된 분산 값(var).
    """
    # var_range에서 랜덤하게 분산 값을 선택
    var = np.random.randint(var_range[0], var_range[1] + 1)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, var

# 디렉토리 내 모든 이미지 파일 처리
supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

for filename in os.listdir(image_dir):
    if filename.lower().endswith(supported_extensions):
        # 이미지 파일 경로
        image_path = os.path.join(image_dir, filename)
        
        # 이미지 읽기 (BGR 형식으로 읽은 후 RGB로 변환)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 노이즈 추가 및 저장 (5번 반복)
        for i in range(5):
            noised_image, var = add_gaussian_noise(image)
            
            # 저장할 파일 이름 생성
            name, ext = os.path.splitext(filename)
            noised_filename = f"{name}_noised_var{var}_i{i+1}{ext}"
            save_path = os.path.join(save_directory, noised_filename)
            
            # RGB 이미지를 BGR로 다시 변환하여 저장
            noised_image_bgr = cv2.cvtColor(noised_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, noised_image_bgr)
        
        print(f"노이즈 추가 완료: {filename}")

print("모든 이미지에 노이즈 추가가 완료되었습니다.")

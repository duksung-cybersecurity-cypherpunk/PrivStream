import os
from glob import glob
from sklearn.model_selection import train_test_split
import yaml

# 데이터셋 경로 정의
dataset_path = r'경로' 
train_images_path = os.path.join(dataset_path, 'images/*')

# 이미지 리스트 가져오기
img_list = glob(train_images_path)

# 이미지 수 출력
print(f"Number of images: {len(img_list)}")

# 이미지들을 train과 validation 세트로 나누기
train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)

# train과 validation 세트의 이미지 수 출력
print(f"Train images: {len(train_img_list)}, Validation images: {len(val_img_list)}")

# train과 validation 이미지 경로를 텍스트 파일로 저장
with open(os.path.join(dataset_path, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open(os.path.join(dataset_path, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

# YAML 파일 로드
yaml_file_path = os.path.join(dataset_path, 'data.yaml')
with open(yaml_file_path, 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

print(data)

# YAML 파일의 경로 업데이트
data['train'] = os.path.join(dataset_path, 'train.txt')
data['val'] = os.path.join(dataset_path, 'val.txt')

with open(yaml_file_path, 'w') as f:
    yaml.dump(data, f)

print(data)

# YOLOv5 디렉토리로 이동
os.chdir('욜로 경로') 

# 훈련 실행
os.system(f'python train.py --img 416 --batch 16 --epochs 50 --data {yaml_file_path} --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name concat_result')

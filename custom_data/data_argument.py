import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 이미지 파일이 있는 디렉토리 경로
image_dir = r'경로'

# 증강된 이미지를 저장할 디렉토리 경로
save_directory = r'경로'
os.makedirs(save_directory, exist_ok=True)

# 디렉토리 내 모든 이미지 파일 처리
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 이미지 파일 읽기
        image_path = os.path.join(image_dir, filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # 증강 이미지 생성 및 저장
        def save_augmented_images(image, generator, n_images=4):
            # ImageDataGenerator는 여러개의 image를 입력으로 받기 때문에 4차원으로 입력 해야함.
            image_batch = np.expand_dims(image, axis=0)
            # featurewise_center or featurewise_std_normalization or zca_whitening 가 True일때만 fit 해주어야함
            generator.fit(image_batch)
            # flow로 image batch를 generator에 넣어주어야함.
            data_gen_iter = generator.flow(image_batch, save_to_dir=save_directory, save_prefix=os.path.splitext(filename)[0]+'_aug', save_format='jpeg')

            for i in range(n_images):
                # generator에 batch size 만큼 augmentation 적용(매번 적용이 다름)
                next(data_gen_iter)

        # 다양한 증강 기법 적용

        # data_generator = ImageDataGenerator(vertical_flip=True)
        # save_augmented_images(image, data_generator, n_images=1)

        # data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        # save_augmented_images(image, data_generator, n_images=1)

        # data_generator = ImageDataGenerator(zoom_range=[0.5, 0.8])
        # save_augmented_images(image, data_generator, n_images=1)

        data_generator = ImageDataGenerator(rotation_range=200)
        save_augmented_images(image, data_generator, n_images=1)

        # data_generator = ImageDataGenerator(zoom_range=[1.3, 1.5], fill_mode='constant', cval=0)
        # save_augmented_images(image, data_generator, n_images=1)

        # data_generator = ImageDataGenerator(brightness_range=(1.3, 1.6))
        # save_augmented_images(image, data_generator, n_images=1)

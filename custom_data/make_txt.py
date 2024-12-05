import os

def create_empty_txt_for_images(directory):
    # 이미지 확장자 목록
    image_extensions = ('.jpg', '.jpeg' ,'.png')

    # 디렉터리 내 파일 목록을 확인
    for filename in os.listdir(directory):
        # 파일 확장자가 jpg 또는 jpeg인지 확인
        if filename.lower().endswith(image_extensions):
            # 이미지 파일명에서 확장자를 제거한 텍스트 파일 경로 생성
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(directory, txt_filename)
            
            # 동일한 이름을 가진 txt 파일이 없다면 생성
            if not os.path.exists(txt_path):
                with open(txt_path, 'w') as txt_file:
                    pass  # 빈 txt 파일 생성

# 예시로 사용할 디렉터리 경로 설정
#example_directory = r"C:\Users\DS\Desktop\test\images"  # 여기에 실제 디렉터리 경로를 입력
example_directory =r"C:\Users\DS\Desktop\mk_dataset\background"
create_empty_txt_for_images(example_directory)

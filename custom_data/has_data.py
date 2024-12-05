import os

# 대상 디렉터리 설정
directory = r'C:\Users\DS\Desktop\custom_data\images'

# 지정된 디렉터리 내의 모든 파일을 순회
for filename in os.listdir(directory):
    # 파일이 .jpg 또는 .jpeg로 끝나는 경우
    if filename.endswith(('.jpg', '.jpeg', 'png')):
        # 해당 이미지에 대한 txt 파일 경로 설정
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_filepath = os.path.join(directory, txt_filename)
        
        # txt 파일이 존재하지 않으면 새로 생성
        if not os.path.exists(txt_filepath):
            with open(txt_filepath, 'w', encoding='utf-8') as file:
                file.write("")  # 빈 파일 생성
            print(f"{txt_filename} 파일이 생성되었습니다.")
        else:
            print(f"{txt_filename} 파일이 이미 존재합니다.")

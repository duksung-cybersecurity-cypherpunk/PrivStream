import os

# 대상 디렉터리 설정
directory = r'C:\Users\User\Desktop\prprprpr\concat_dataset\4_knife\label'

# 지정된 디렉터리 내의 모든 파일을 순회
for filename in os.listdir(directory):
    # 파일이 'knife'로 시작하고 '.txt' 확장자를 가진 경우
    if filename.startswith("") and filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        with open(filepath, 'w', encoding='utf-8') as file:
            for line in lines:
                # 각 라인의 시작 부분에서 첫 번째 '0'을 '3'으로 변경
                if line.startswith('0'):
                    line = '4' + line[1:]
                file.write(line)
                
        print(f"{filename} 내의 변경이 완료되었습니다.")

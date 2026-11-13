import pandas as pd

# 1. 원본 데이터 불러오기 (한글 인코딩 고려)
file_name = '경기도_농가데이터셋_토마토_연동_대규모.csv'
df = pd.read_csv(file_name, encoding='cp949')

# 2. 사용할 독립 변수와 종속 변수 컬럼 정의
feature_cols = ['내부온도', '내부습도', '내부CO2', '지온', '줄기직경', '엽장', '엽폭']
target_col = '열매수'
selected_cols = feature_cols + [target_col]

# 3. 데이터 전처리: 필요한 컬럼만 추출하고 결측치(빈 값) 제거
preprocessed_df = df[selected_cols].dropna()

# 4. 전처리된 데이터를 새로운 CSV 파일로 저장 (엑셀에서 깨지지 않도록 utf-8-sig 인코딩 사용)
output_filename = 'preprocessed_tomato_data.csv'
preprocessed_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"✅ 전처리 완료! '{output_filename}' 파일이 성공적으로 생성되었습니다.")
print(f"총 데이터 행 수: {len(preprocessed_df)}개")

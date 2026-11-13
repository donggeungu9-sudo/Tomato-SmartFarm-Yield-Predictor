import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("🔄 [1/3] 원본 데이터 파일을 찾는 중...")
# 원본 데이터 파일명 (본인의 파일명에 맞게 수정 가능)
raw_filename = '경기도_농가데이터셋_토마토_연동_대규모.csv'

if not os.path.exists(raw_filename):
    print(f"❌ 오류: '{raw_filename}' 파일을 찾을 수 없습니다. 원본 CSV 파일이 폴더 안에 있는지 확인해 주세요.")
    exit()

# 원본 데이터 로드
df = pd.read_csv(raw_filename, encoding='cp949')

# 사용할 변수 정의
feature_cols = ['내부온도', '내부습도', '내부CO2', '지온', '줄기직경', '엽장', '엽폭']
target_col = '열매수'
selected_cols = feature_cols + [target_col]

print("🔄 [2/3] 데이터 전처리 및 'preprocessed_tomato_data.csv' 생성 중...")
# 결측치 제거 및 필요한 컬럼 추출
preprocessed_df = df[selected_cols].dropna()

# 전처리된 데이터 CSV로 저장
output_csv = 'preprocessed_tomato_data.csv'
preprocessed_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"✅ '{output_csv}' 파일 생성 완료! (총 데이터: {len(preprocessed_df)}행)")

print("🔄 [3/3] 머신러닝 모델 학습 및 'model.pkl' 생성 중...")
# X, y 분리
X = preprocessed_df[feature_cols]
y = preprocessed_df[target_col]

# 랜덤 포레스트 회귀 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 학습된 모델 저장
output_model = 'model.pkl'
joblib.dump(model, output_model)
print(f"✅ '{output_model}' 파일 생성 완료!")

print("\n🎉 모든 준비가 완료되었습니다! 이제 터미널에서 'streamlit run main.py'를 실행해 보세요.")

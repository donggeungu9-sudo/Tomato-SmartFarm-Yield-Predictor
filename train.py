import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. 데이터 불러오기 (파일명이 다를 경우 실제 파일명으로 수정해 주세요)
df = pd.read_csv('경기도_농가데이터셋_토마토_연동_대규모.csv', encoding='cp949')

# 2. 독립 변수(X)와 종속 변수(Y) 컬럼 지정
feature_cols = ['내부온도', '내부습도', '내부CO2', '지온', '줄기직경', '엽장', '엽폭']
target_col = '열매수'

# 결측치 제거 후 X, y 분리
model_df = df[feature_cols + [target_col]].dropna()
X = model_df[feature_cols]
y = model_df[target_col]

# 3. 랜덤 포레스트 회귀 모델 생성 및 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. 학습된 모델을 'model.pkl' 파일로 저장
joblib.dump(model, 'model.pkl')

print("✅ 모델 학습이 완료되었으며 'model.pkl' 파일이 성공적으로 생성되었습니다!")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 페이지 기본 설정
st.set_page_config(
    page_title="토마토 열매 수량 예측 스마트팜",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 토마토 스마트팜 열매 수량 예측 모델")
st.markdown("온실 내부 환경 데이터와 작물 생장 지표를 입력하여 예상 열매 수량을 확인하세요.")

# 모델 파일 로드 함수 (모델이 없는 경우를 대비한 예외 처리 포함)
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

if model is None:
    st.warning("⚠️ 학습된 모델 파일(`model.pkl`)을 찾을 수 없습니다. 먼저 모델 학습 후 `joblib.dump()`로 모델을 저장해 주세요.")

st.divider()

st.subheader("📊 온실 환경 및 생장 지표 입력")

# 사용자 입력을 받기 위한 레이아웃 구성 (2단 컬럼)
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### [환경 조건]")
    inner_temp = st.slider("내부온도 (°C)", min_value=10.0, max_value=40.0, value=24.0, step=0.1)
    inner_humid = st.slider("내부습도 (%)", min_value=20.0, max_value=100.0, value=75.0, step=1.0)
    inner_co2 = st.slider("내부 CO2 (ppm)", min_value=300, max_value=1500, value=600, step=10)
    ground_temp = st.slider("지온 (°C)", min_value=10.0, max_value=35.0, value=21.0, step=0.1)

with col2:
    st.markdown("##### [생장 조사 지표]")
    stem_diam = st.slider("줄기직경 (mm)", min_value=5.0, max_value=25.0, value=12.0, step=0.1)
    leaf_length = st.slider("엽장 (cm)", min_value=10.0, max_value=60.0, value=35.0, step=0.5)
    leaf_width = st.slider("엽폭 (cm)", min_value=5.0, max_value=50.0, value=25.0, step=0.5)

st.divider()

# 예측 버튼
if st.button("🚀 예상 열매 수량 예측하기", type="primary", use_container_width=True):
    # 입력값을 데이터프레임으로 변환 (학습 시 사용한 컬럼 순서와 일치시켜야 함)
    input_data = pd.DataFrame([[
        inner_temp, inner_humid, inner_co2, ground_temp, 
        stem_diam, leaf_length, leaf_width
    ]], columns=['내부온도', '내부습도', '내부CO2', '지온', '줄기직경', '엽장', '엽폭'])
    
    if model is not None:
        # 모델 예측 수행
        prediction = model.predict(input_data)[0]
        
        st.success(f"### 🌱 예측된 최종 토마토 열매 수량: **{prediction:.1f} 개**")
        st.balloons()
    else:
        # 모델 파일이 없을 경우 임시 시뮬레이션 결과 출력 (테스트용)
        st.info(f"💡 [테스트 모드] 입력하신 조건에 따른 예상 열매 수량은 약 **{(stem_diam * 1.5 + leaf_length * 0.3):.1f} 개**입니다. (모델 연동 시 정확한 값이 출력됩니다)")

st.markdown("---")
st.caption("경기도 스마트팜 토마토 연동 대규모 데이터셋 기반 머신러닝 프로젝트")

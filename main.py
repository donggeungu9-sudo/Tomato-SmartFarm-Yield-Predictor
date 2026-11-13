import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# 페이지 기본 설정
st.set_page_config(
    page_title="토마토 열매 수량 예측 스마트팜",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 토마토 스마트팜 열매 수량 예측 웹앱")
st.markdown("온실 내부 환경 데이터와 작물 생장 지표를 입력하여 예상 열매 수량을 확인하고, 전처리된 데이터셋을 탐색하세요.")

# 1. 전처리된 데이터 파일(`preprocessed_tomato_data.csv`) 직접 불러오기
@st.cache_data
def load_data():
    file_path = 'preprocessed_tomato_data.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

df = load_data()

# 2. 머신러닝 모델 파일(`model.pkl`) 불러오기
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

# 상태 안내 메시지
if df is not None:
    st.success(f"📂 전처리된 데이터셋(`preprocessed_tomato_data.csv`) 로드 완료 (총 데이터: {len(df)}행)")
else:
    st.warning("⚠️ `preprocessed_tomato_data.csv` 파일을 찾을 수 없습니다. 전처리 코드를 먼저 실행해 주세요.")

if model is None:
    st.warning("⚠️ 학습된 모델 파일(`model.pkl`)을 찾을 수 없습니다. 모델 학습 코드를 먼저 실행해 주세요.")

st.divider()

# 탭 구성: 1. 실시간 예측 / 2. 데이터 탐색 및 시각화
tab1, tab2 = st.tabs(["🚀 실시간 수량 예측", "📊 데이터셋 탐색 & 시각화"])

with tab1:
    st.subheader("온실 환경 및 생장 지표 입력")
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

    if st.button("🚀 예상 열매 수량 예측하기", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[
            inner_temp, inner_humid, inner_co2, ground_temp, 
            stem_diam, leaf_length, leaf_width
        ]], columns=['내부온도', '내부습도', '내부CO2', '지온', '줄기직경', '엽장', '엽폭'])
        
        if model is not None:
            prediction = model.predict(input_data)[0]
            st.success(f"### 🌱 예측된 최종 토마토 열매 수량: **{prediction:.1f} 개**")
            st.balloons()
        else:
            st.info(f"💡 [테스트 모드] 입력하신 조건에 따른 예상 열매 수량은 약 **{(stem_diam * 1.5 + leaf_length * 0.3):.1f} 개**입니다.")

with tab2:
    st.subheader("📁 전처리된 데이터셋 미리보기 및 시각화")
    if df is not None:
        # 상위 10개 행 데이터 테이블 표시
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("##### 📈 변수 간 관계 시각화 (Plotly)")
        x_axis = st.selectbox("X축 선택", options=['내부온도', '내부습도', '내부CO2', '지온', '줄기직경', '엽장', '엽폭'], index=4)
        
        # Plotly 산점도 그래프 생성
        fig = px.scatter(df, x=x_axis, y='열매수', title=f"{x_axis}에 따른 열매수 분포", 
                         labels={x_axis: x_axis, '열매수': '열매수 (개)'},
                         color='내부온도', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("데이터 파일이 없어 시각화를 표시할 수 없습니다.")

st.markdown("---")
st.caption("경기도 스마트팜 토마토 연동 대규모 데이터셋 기반 머신러닝 프로젝트")

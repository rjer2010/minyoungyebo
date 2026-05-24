# app.py — GitHub 루트에 저장

import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
from datetime import datetime, timedelta
import tensorflow as tf
import plotly.graph_objects as go

# ──────────────────────────────────────────
# 1. 설정
# ──────────────────────────────────────────
st.set_page_config(page_title="AI 기온 예보", page_icon="🌡️", layout="wide")

API_KEY = st.secrets["KMA_API_KEY"]   # Streamlit Secrets에 저장
SEQ_LEN = 24
FEATURE_COLS = [
    "hour_sin","hour_cos","month_sin","month_cos",
    "TA_lag_1h","TA_lag_2h","TA_lag_3h","TA_lag_6h","TA_lag_12h","TA_lag_24h",
    "TA_roll_mean_6h","TA_roll_std_6h","TA_roll_mean_24h",
    "diurnal_range","HM","WS"
]

# ──────────────────────────────────────────
# 2. 모델 로딩 (캐시로 반복 로딩 방지)
# ──────────────────────────────────────────
@st.cache_resource
def load_models():
    lstm  = tf.keras.models.load_model("models/lstm_model.h5")
    with open("models/xgb_model.pkl","rb") as f:  xgb  = pickle.load(f)
    with open("models/lgb_model.pkl","rb") as f:  lgb  = pickle.load(f)
    with open("models/rf_model.pkl","rb") as f:   rf   = pickle.load(f)
    with open("models/meta_model.pkl","rb") as f: meta = pickle.load(f)
    with open("models/scaler_X.pkl","rb") as f:   scX  = pickle.load(f)
    with open("models/scaler_y.pkl","rb") as f:   scY  = pickle.load(f)
    return lstm, xgb, lgb, rf, meta, scX, scY

lstm_m, xgb_m, lgb_m, rf_m, meta_m, scX, scY = load_models()

# ──────────────────────────────────────────
# 3. 실시간 데이터 수집 및 피처 생성
# ──────────────────────────────────────────
@st.cache_data(ttl=3600)   # 1시간마다 갱신
def fetch_recent_obs(nx=60, ny=127):
    """최근 48시간 실황 데이터 수집 및 피처 생성"""
    now  = datetime.now()
    rows = []
    for h in range(48, 0, -1):
        dt = now - timedelta(hours=h)
        base_date = dt.strftime("%Y%m%d")
        base_time = f"{(dt.hour // 1):02d}00"
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
        params = {"serviceKey": API_KEY, "pageNo":1, "numOfRows":50,
                  "dataType":"JSON", "base_date":base_date,
                  "base_time":base_time, "nx":nx, "ny":ny}
        try:
            items = requests.get(url, params=params, timeout=5).json()
            items = items["response"]["body"]["items"]["item"]
            row = {"datetime": dt}
            for it in items:
                row[it["category"]] = float(it["obsrValue"])
            rows.append(row)
        except:
            pass
    df = pd.DataFrame(rows)
    df = df.rename(columns={"T1H":"TA","REH":"HM","WSD":"WS"})
    df = build_features(df)
    return df

def build_features(df):
    """STEP 3과 동일한 피처 생성 함수 (app.py 내 재정의)"""
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    for lag in [1,2,3,6,12,24]:
        df[f"TA_lag_{lag}h"] = df["TA"].shift(lag)
    df["TA_roll_mean_6h"]  = df["TA"].shift(1).rolling(6).mean()
    df["TA_roll_std_6h"]   = df["TA"].shift(1).rolling(6).std()
    df["TA_roll_mean_24h"] = df["TA"].shift(1).rolling(24).mean()
    daily = df.groupby(df["datetime"].dt.date)["TA"].agg(["max","min"])
    daily["diurnal_range"] = daily["max"] - daily["min"]
    df["date"] = df["datetime"].dt.date
    df = df.merge(daily[["diurnal_range"]], left_on="date", right_index=True, how="left")
    return df.dropna()

# ──────────────────────────────────────────
# 4. 예측 함수
# ──────────────────────────────────────────
def predict_tomorrow(df_feat):
    """내일 0시~23시 기온 예측 (재귀적 24스텝 예측)"""
    history = df_feat[FEATURE_COLS].values
    history_sc = scX.transform(history)
    predictions_sc = []

    for step in range(24):  # 내일 24시간 순차 예측
        # LSTM: 마지막 SEQ_LEN 시퀀스 사용
        lstm_input = history_sc[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)
        lstm_p = lstm_m.predict(lstm_input, verbose=0)[0][0]

        # 트리 계열: 마지막 행 피처 사용
        tree_input = history_sc[-1].reshape(1, -1)
        xgb_p  = xgb_m.predict(tree_input)[0]
        lgb_p  = lgb_m.predict(tree_input)[0]
        rf_p   = rf_m.predict(tree_input)[0]

        # 스태킹
        stack  = np.array([[lstm_p, xgb_p, lgb_p, rf_p]])
        final_sc = meta_m.predict(stack)[0]
        predictions_sc.append(final_sc)

        # 다음 스텝을 위한 피처 업데이트 (재귀 예측)
        next_row = history_sc[-1].copy()
        # lag 피처 시프트 (단순화 버전)
        next_row[4] = final_sc   # TA_lag_1h ← 방금 예측값
        history_sc = np.vstack([history_sc, next_row])

    pred_temps = scY.inverse_transform(
        np.array(predictions_sc).reshape(-1,1)
    ).ravel()
    return pred_temps

# ──────────────────────────────────────────
# 5. UI 렌더링
# ──────────────────────────────────────────
st.title("🌡️ AI 기온 예보 — 내일 시간대별 예측")
st.caption(f"스태킹 앙상블 (LSTM + XGBoost + LightGBM + RandomForest) | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

col1, col2 = st.columns([1, 3])
with col1:
    city = st.selectbox("도시 선택", ["서울(60,127)", "부산(98,76)", "대구(89,90)"])
    nx, ny = map(int, city.split("(")[1].rstrip(")").split(","))
    run_btn = st.button("🔄 예측 실행", type="primary")

if run_btn:
    with st.spinner("기상 데이터 수집 중..."):
        df_feat = fetch_recent_obs(nx, ny)
    with st.spinner("AI 예측 중..."):
        temps = predict_tomorrow(df_feat)

    hours = [f"{h:02d}:00" for h in range(24)]
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%m월 %d일")

    # 지표 카드
    c1, c2, c3 = st.columns(3)
    c1.metric("🌅 예상 최저기온", f"{temps.min():.1f}°C", f"{temps.argmin()}시")
    c2.metric("☀️ 예상 최고기온", f"{temps.max():.1f}°C", f"{temps.argmax()}시")
    c3.metric("📊 일교차", f"{temps.max()-temps.min():.1f}°C")

    # Plotly 차트
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=temps, mode="lines+markers",
        line=dict(color="#FF6B6B", width=3),
        marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(255,107,107,0.1)",
        name="예측 기온"
    ))
    fig.update_layout(
        title=f"{tomorrow} 시간별 예측 기온",
        xaxis_title="시각", yaxis_title="기온 (°C)",
        template="plotly_white", height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # 상세 테이블
    df_result = pd.DataFrame({"시각": hours, "예측기온(°C)": [f"{t:.1f}" for t in temps]})
    st.dataframe(df_result, use_container_width=True, hide_index=True)

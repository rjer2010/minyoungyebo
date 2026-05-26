import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ──────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────
st.set_page_config(
    page_title="AI 기온 예보",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}
.block-container { padding: 2rem 2.5rem; }

.weather-header {
    background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 50%, #1a3a5c 100%);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.weather-header::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.weather-header::after {
    content: "";
    position: absolute;
    bottom: -40px; left: 40px;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.header-title {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.header-sub {
    font-size: 0.85rem;
    opacity: 0.6;
    font-weight: 300;
    font-family: 'DM Mono', monospace;
}

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #eef0f5;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    color: #8a92a0;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1f3a;
    font-family: 'DM Mono', monospace;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.8rem;
    color: #a0a8b8;
    margin-top: 0.2rem;
}

.status-box {
    background: #f0f7ff;
    border-left: 4px solid #3b6fd4;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #2d4a8a;
    margin-bottom: 1rem;
}
.error-box {
    background: #fff0f0;
    border-left: 4px solid #e05a5a;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #8a2020;
    margin-bottom: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #2d3561, #3b6fd4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: 'Noto Sans KR', sans-serif;
    cursor: pointer;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

.stSelectbox > div > div {
    border-radius: 10px;
    border-color: #dde2ee;
}

.hour-table {
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────
SEQ_LEN = 24
FEATURE_COLS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "TA_lag_1h", "TA_lag_2h", "TA_lag_3h", "TA_lag_6h", "TA_lag_12h", "TA_lag_24h",
    "TA_roll_mean_6h", "TA_roll_std_6h", "TA_roll_mean_24h",
    "diurnal_range", "HM", "WS"
]

CITIES = {
    "서울": (60, 127),
    "부산": (98, 76),
    "대구": (89, 90),
    "인천": (55, 124),
    "광주": (58, 74),
    "대전": (67, 100),
}

# ──────────────────────────────────────────
# numpy LSTM 순전파 (TensorFlow 불필요)
# ──────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x, h, c, W, U, b):
    z = x @ W + h @ U + b
    units = W.shape[1] // 4
    i = sigmoid(z[..., :units])
    f = sigmoid(z[..., units:2*units])
    g = np.tanh(z[..., 2*units:3*units])
    o = sigmoid(z[..., 3*units:])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new

def lstm_predict_numpy(x_seq, weights):
    """
    x_seq: (seq_len, features) numpy array
    weights: Colab에서 저장한 layer별 가중치 딕셔너리
    반환: 스칼라 예측값 (스케일된 상태)
    """
    layer_names = list(weights.keys())

    # Layer 1: LSTM(128, return_sequences=True)
    W1, U1, b1 = weights[layer_names[0]]
    units1 = W1.shape[1] // 4
    h1 = np.zeros(units1)
    c1 = np.zeros(units1)
    seq_out = []
    for t in range(len(x_seq)):
        h1, c1 = lstm_cell(x_seq[t], h1, c1, W1, U1, b1)
        seq_out.append(h1.copy())

    # Layer 2: LSTM(64, return_sequences=False)
    W2, U2, b2 = weights[layer_names[1]]
    units2 = W2.shape[1] // 4
    h2 = np.zeros(units2)
    c2 = np.zeros(units2)
    for t in range(len(seq_out)):
        h2, c2 = lstm_cell(seq_out[t], h2, c2, W2, U2, b2)

    # Dense(32, relu)
    W3, b3 = weights[layer_names[2]]
    out = np.maximum(0, h2 @ W3 + b3)

    # Dense(1, linear)
    W4, b4 = weights[layer_names[3]]
    out = out @ W4 + b4

    return float(out.ravel()[0])

# ──────────────────────────────────────────
# 모델 로딩 (캐시)
# ──────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open("models/lstm_weights.pkl", "rb") as f:
            lstm_w = pickle.load(f)
        from xgboost import XGBRegressor
        xgb = XGBRegressor()
        xgb.load_model("models/xgb_model.json")
        with open("models/lgb_model.pkl", "rb") as f:
            lgb = pickle.load(f)
        with open("models/rf_model.pkl", "rb") as f:
            rf = pickle.load(f)
        with open("models/meta_model.pkl", "rb") as f:
            meta = pickle.load(f)
        with open("models/scaler_X.pkl", "rb") as f:
            scX = pickle.load(f)
        with open("models/scaler_y.pkl", "rb") as f:
            scY = pickle.load(f)
        return lstm_w, xgb, lgb, rf, meta, scX, scY
    except FileNotFoundError as e:
        return None, None, None, None, None, None, None

# ──────────────────────────────────────────
# 피처 엔지니어링 (학습 시와 동일한 함수)
# ──────────────────────────────────────────
def build_features(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"TA_lag_{lag}h"] = df["TA"].shift(lag)

    df["TA_roll_mean_6h"] = df["TA"].shift(1).rolling(6).mean()
    df["TA_roll_std_6h"] = df["TA"].shift(1).rolling(6).std()
    df["TA_roll_mean_24h"] = df["TA"].shift(1).rolling(24).mean()

    daily = df.groupby(df["datetime"].dt.date)["TA"].agg(["max", "min"])
    daily["diurnal_range"] = daily["max"] - daily["min"]
    df["date"] = df["datetime"].dt.date
    df = df.merge(daily[["diurnal_range"]], left_on="date", right_index=True, how="left")

    return df.dropna()

# ──────────────────────────────────────────
# 기상청 초단기실황 API (최근 48시간)
# ──────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_recent_obs(nx, ny, api_key):
    """
    초단기실황(최근 6시간) + 단기예보(최근 4일) 조합
    - 직전 6시간: 실제 관측값 사용 (정확)
    - 그 이전: 단기예보 예측값으로 보완 (차선)
    """
    from datetime import datetime, timedelta
    rows_obs = {}   # datetime → row
    now = datetime.now()

    # ── 1단계: 단기예보로 최근 4일치 베이스 수집 (3시간 간격) ──
    for day_offset in range(4, 0, -1):
        dt = now - timedelta(days=day_offset)
        base_date = dt.strftime("%Y%m%d")
        for base_time in ["0200","0500","0800","1100","1400","1700","2000","2300"]:
            url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
            params = {
                "serviceKey": api_key,
                "pageNo": 1, "numOfRows": 1000,
                "dataType": "JSON",
                "base_date": base_date,
                "base_time": base_time,
                "nx": nx, "ny": ny
            }
            try:
                items = requests.get(url, params=params, timeout=10).json()
                items = items["response"]["body"]["items"]["item"]
                df_tmp = pd.DataFrame(items)
                for category, col in [("TMP","TA"),("REH","HM"),("WSD","WS")]:
                    subset = df_tmp[df_tmp["category"] == category]
                    for _, row in subset.iterrows():
                        fcst_dt = datetime.strptime(
                            row["fcstDate"] + row["fcstTime"], "%Y%m%d%H%M"
                        )
                        if fcst_dt not in rows_obs:
                            rows_obs[fcst_dt] = {"datetime": fcst_dt}
                        rows_obs[fcst_dt][col] = float(row["fcstValue"])
            except Exception:
                continue

    # ── 2단계: 초단기실황으로 최근 6시간 덮어쓰기 (실제 관측값) ──
    for h in range(6, 0, -1):
        dt = now - timedelta(hours=h)
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
        params = {
            "serviceKey": api_key,
            "pageNo": 1, "numOfRows": 50,
            "dataType": "JSON",
            "base_date": dt.strftime("%Y%m%d"),
            "base_time": f"{dt.hour:02d}00",
            "nx": nx, "ny": ny
        }
        try:
            items = requests.get(url, params=params, timeout=6).json()
            items = items["response"]["body"]["items"]["item"]
            row = {"datetime": dt}
            for it in items:
                if it["category"] == "T1H": row["TA"] = float(it["obsrValue"])
                if it["category"] == "REH": row["HM"] = float(it["obsrValue"])
                if it["category"] == "WSD": row["WS"] = float(it["obsrValue"])
            # 기존 단기예보 값을 실제 관측값으로 덮어쓰기
            closest = min(rows_obs.keys(),
                         key=lambda x: abs((x - dt).total_seconds()),
                         default=None)
            if closest and abs((closest - dt).total_seconds()) <= 1800:
                rows_obs[closest].update(row)
            else:
                rows_obs[dt] = row
        except Exception:
            continue

    if not rows_obs:
        return None

    df = pd.DataFrame(list(rows_obs.values()))
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.dropna(subset=["TA"])
    df[["HM","WS"]] = df[["HM","WS"]].ffill().bfill()

    return df

# ──────────────────────────────────────────
# 예측 함수 (재귀적 24스텝)
# ──────────────────────────────────────────
def predict_tomorrow(df_raw, models):
    lstm_w, xgb_m, lgb_m, rf_m, meta_m, scX, scY = models

    df_feat = build_features(df_raw)
    if df_feat.empty or len(df_feat) < 6:
        return None, f"피처 생성 후 데이터가 부족합니다 (수집: {len(df_feat)}행 / 최소 6행 필요)"

    # 수집량이 SEQ_LEN 미만이면 반복 패딩으로 보완
    if len(df_feat) < SEQ_LEN:
        repeat_times = (SEQ_LEN // len(df_feat)) + 1
        df_feat = pd.concat([df_feat] * repeat_times, ignore_index=True).iloc[-SEQ_LEN:]

    # 사용 가능한 피처 컬럼만 선택
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    if len(available) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available)
        return None, f"피처 누락: {missing}"

    history_sc = scX.transform(df_feat[FEATURE_COLS].values)
    predictions_sc = []

    for step in range(24):
        seq = history_sc[-SEQ_LEN:]  # (24, features)
        tree_row = history_sc[-1].reshape(1, -1)

        # 베이스 모델 예측
        p_lstm = lstm_predict_numpy(seq, lstm_w)
        p_xgb = xgb_m.predict(tree_row)[0]
        p_lgb = lgb_m.predict(tree_row)[0]
        p_rf = rf_m.predict(tree_row)[0]

        # 메타 모델 스태킹
        stack = np.array([[p_lstm, p_xgb, p_lgb, p_rf]])
        final_sc = meta_m.predict(stack)[0]
        predictions_sc.append(final_sc)

        # 다음 스텝 피처 업데이트 (lag 시프트)
        next_row = history_sc[-1].copy()
        # FEATURE_COLS 인덱스 기준으로 lag 업데이트
        lag_indices = {
            "TA_lag_1h": 4, "TA_lag_2h": 5, "TA_lag_3h": 6,
            "TA_lag_6h": 7, "TA_lag_12h": 8, "TA_lag_24h": 9
        }
        # 이전 lag 값들을 한 칸씩 밀기
        next_row[lag_indices["TA_lag_2h"]] = next_row[lag_indices["TA_lag_1h"]]
        next_row[lag_indices["TA_lag_3h"]] = next_row[lag_indices["TA_lag_2h"]]
        next_row[lag_indices["TA_lag_6h"]] = next_row[lag_indices["TA_lag_3h"]]
        next_row[lag_indices["TA_lag_12h"]] = next_row[lag_indices["TA_lag_6h"]]
        next_row[lag_indices["TA_lag_24h"]] = next_row[lag_indices["TA_lag_12h"]]
        next_row[lag_indices["TA_lag_1h"]] = final_sc  # 방금 예측값

        # 시간 피처 업데이트
        next_hour = (step + 1) % 24
        next_row[0] = np.sin(2 * np.pi * next_hour / 24)
        next_row[1] = np.cos(2 * np.pi * next_hour / 24)

        history_sc = np.vstack([history_sc, next_row])

    temps = scY.inverse_transform(
        np.array(predictions_sc).reshape(-1, 1)
    ).ravel()

    return temps, None

# ──────────────────────────────────────────
# UI 렌더링
# ──────────────────────────────────────────

# 헤더
tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y년 %m월 %d일")
st.markdown(f"""
<div class="weather-header">
    <div class="header-title">🌡️ AI 기온 예보</div>
    <div class="header-sub">
        LSTM · XGBoost · LightGBM · RandomForest → 스태킹 앙상블 &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M')} 기준
    </div>
</div>
""", unsafe_allow_html=True)

# 사이드바: 설정
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    city_name = st.selectbox("도시 선택", list(CITIES.keys()))
    nx, ny = CITIES[city_name]
    try:
        api_key = st.secrets["KMA_API_KEY"]
    except Exception:
        st.error("⚠️ Streamlit Secrets에 KMA_API_KEY가 설정되지 않았습니다.")
        st.stop()
    run_btn = st.button("🔄 예측 실행", type="primary")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#888; line-height:1.7">
    <b>모델 구성</b><br>
    · LSTM (numpy 순전파)<br>
    · XGBoost<br>
    · LightGBM<br>
    · RandomForest<br>
    · Ridge 메타 모델<br><br>
    <b>데이터</b><br>
    기상청 초단기실황 (최근 48h)<br><br>
    <b>예측 방식</b><br>
    재귀적 24스텝 예측
    </div>
    """, unsafe_allow_html=True)

# 메인 영역
if not run_btn:
    st.markdown(f"""
    <div class="status-box">
        📍 <b>{city_name}</b> — 내일 ({tomorrow_str}) 시간대별 기온 예측을 시작하려면
        사이드바에서 도시를 선택하고 <b>예측 실행</b>을 클릭하세요.
    </div>
    """, unsafe_allow_html=True)

else:
    # 모델 로딩
    models = load_models()
    if models[0] is None:
        st.markdown("""
        <div class="error-box">
        ❗ <b>모델 파일을 찾을 수 없습니다.</b><br>
        <code>models/</code> 폴더에 다음 파일이 있는지 확인하세요:<br>
        lstm_weights.pkl · xgb_model.pkl · lgb_model.pkl · rf_model.pkl · meta_model.pkl · scaler_X.pkl · scaler_y.pkl
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # 데이터 수집
    with st.spinner(f"🌐 {city_name} 최근 기상 데이터 수집 중..."):
        df_raw = fetch_recent_obs(nx, ny, api_key)

    if df_raw is None or len(df_raw) < SEQ_LEN:
        st.markdown(f"""
        <div class="error-box">
        ❗ 관측 데이터가 부족합니다 (수집: {len(df_raw) if df_raw is not None else 0}행 / 필요: {SEQ_LEN}행 이상).<br>
        API 키와 격자 좌표를 확인하거나 잠시 후 다시 시도해 주세요.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # 예측 실행
    with st.spinner("🤖 AI 예측 중..."):
        temps, err = predict_tomorrow(df_raw, models)

    if err:
        st.markdown(f'<div class="error-box">❗ {err}</div>', unsafe_allow_html=True)
        st.stop()

    # ── 결과 표시 ──
    hours = [f"{h:02d}:00" for h in range(24)]
    min_temp = float(temps.min())
    max_temp = float(temps.max())
    min_hour = int(temps.argmin())
    max_hour = int(temps.argmax())
    diurnal = max_temp - min_temp

    st.markdown(f"#### 📅 {city_name} · {tomorrow_str} 예측 결과")

    # 지표 카드
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌅 예상 최저기온</div>
            <div class="metric-value">{min_temp:.1f}°</div>
            <div class="metric-sub">{min_hour:02d}:00 최저</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">☀️ 예상 최고기온</div>
            <div class="metric-value">{max_temp:.1f}°</div>
            <div class="metric-sub">{max_hour:02d}:00 최고</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 일교차</div>
            <div class="metric-value">{diurnal:.1f}°</div>
            <div class="metric-sub">최고 - 최저</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_temp = float(temps.mean())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌡️ 일 평균기온</div>
            <div class="metric-value">{avg_temp:.1f}°</div>
            <div class="metric-sub">24시간 평균</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly 차트
    color_scale = [
        (t - min_temp) / max(diurnal, 0.1)
        for t in temps
    ]

    fig = go.Figure()

    # 배경 영역
    fig.add_trace(go.Scatter(
        x=hours, y=temps,
        fill="tozeroy",
        fillcolor="rgba(59, 111, 212, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # 메인 라인
    fig.add_trace(go.Scatter(
        x=hours,
        y=temps,
        mode="lines+markers",
        name="예측 기온",
        line=dict(color="#3b6fd4", width=2.5, shape="spline"),
        marker=dict(
            size=[10 if i in (min_hour, max_hour) else 6 for i in range(24)],
            color=["#e05a5a" if i == max_hour else "#3b6fd4" if i == min_hour else "#3b6fd4" for i in range(24)],
            line=dict(color="white", width=2)
        ),
        hovertemplate="<b>%{x}</b><br>기온: %{y:.1f}°C<extra></extra>"
    ))

    # 최고/최저 주석
    fig.add_annotation(
        x=hours[max_hour], y=max_temp,
        text=f"최고 {max_temp:.1f}°C",
        showarrow=True, arrowhead=2, arrowcolor="#e05a5a",
        font=dict(size=12, color="#e05a5a", family="DM Mono"),
        bgcolor="white", bordercolor="#e05a5a", borderwidth=1,
        borderpad=4, ay=-36
    )
    fig.add_annotation(
        x=hours[min_hour], y=min_temp,
        text=f"최저 {min_temp:.1f}°C",
        showarrow=True, arrowhead=2, arrowcolor="#3b6fd4",
        font=dict(size=12, color="#3b6fd4", family="DM Mono"),
        bgcolor="white", bordercolor="#3b6fd4", borderwidth=1,
        borderpad=4, ay=36
    )

    fig.update_layout(
        title="",  # undefined 방지
        xaxis=dict(
            title="시각",
            showgrid=False,
            tickfont=dict(family="DM Mono", size=11, color="#8a92a0"),
            linecolor="#eef0f5"
        ),
        yaxis=dict(
            title="기온 (°C)",
            showgrid=True,
            gridcolor="#f0f2f7",
            tickfont=dict(family="DM Mono", size=11, color="#8a92a0"),
            zeroline=False
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 시간대별 상세 테이블
    with st.expander("📋 시간대별 상세 기온 보기"):
        df_result = pd.DataFrame({
            "시각": hours,
            "예측 기온 (°C)": [f"{t:.1f}" for t in temps],
            "비고": [
                "🔴 최고" if i == max_hour else
                "🔵 최저" if i == min_hour else ""
                for i in range(24)
            ]
        })
        st.dataframe(df_result, use_container_width=True, hide_index=True)

    # 최근 수집 데이터 확인
    with st.expander("📡 수집된 최근 관측 데이터 확인"):
        show_cols = [c for c in ["datetime", "TA", "HM", "WS"] if c in df_raw.columns]
        st.dataframe(
            df_raw[show_cols].tail(12).reset_index(drop=True),
            use_container_width=True
        )

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.ad.anomaly_model import ForecastingAnomalyModel
from darts.ad.scorers import NormScorer, KMeansScorer, WassersteinScorer
from darts.ad.detectors import QuantileDetector
sns.set_theme(style='darkgrid')

st.markdown("""
    <style>
    /* Primary 버튼의 배경색과 테두리 색상 변경 */
    div.stButton > button[kind="primary"] {
        background-color: #334155;
        color: white;
        border: none;}
    /* 마우스를 올렸을 때(hover) 색상 */
    div.stButton > button[kind="primary"]:hover {
        background-color: #1E293B;
        color: white;}
    /* 클릭했을 때 색상 */
    div.stButton > button[kind="primary"]:active {
        background-color: #0F172A;
        color: white;}
    </style>
    """, unsafe_allow_html=True)

# =======================================================================
# 2. 메인 화면 및 UI 구성
# =======================================================================

st.set_page_config(page_title="다변량 시계열 이상탐지", layout="wide")
st.title("📈 시계열분석 프로젝트 2")
st.markdown("임의의 시계열 파일(다변량, CSV)을 업로드하여 예측을 수행합니다.")
st.markdown("업로드하는 CSV는 Timestamp 열과 하나 이상의 수치형 Feature를 포함해야 합니다.")
st.divider() 

## CSV 파일 업로드
st.sidebar.markdown("### 📂 CSV 파일 업로드") 
uploaded_file = st.sidebar.file_uploader("upload", type=['csv'],  label_visibility="collapsed")

if "file_loaded" not in st.session_state:       # session_state 초기화 
    st.session_state["file_loaded"] = False

# =======================================================================
# 1. 파일 업로드 처리
# =======================================================================

if uploaded_file is not None:

    if ("file_name" not in st.session_state or st.session_state["file_name"] != uploaded_file.name):
        try:
            df_init = pd.read_csv(uploaded_file)
            st.sidebar.success("📂 파일 로드 완료!")

            # Timestamp 컬럼 처리
            if "Timestamp" in df_init.columns:
                df_init["Timestamp"] = pd.to_datetime(df_init["Timestamp"])
                df_init.set_index("Timestamp", inplace=True)
            else:
                df_init.iloc[:, 0] = pd.to_datetime(df_init.iloc[:, 0])
                df_init.set_index(df_init.columns[0], inplace=True)

            # 중복 타임스탬프 제거
            if df_init.index.duplicated().any():
                df_init = df_init[~df_init.index.duplicated(keep="first")]

            # Sampling Frequency 추정
            inferred_freq = pd.infer_freq(df_init.index)

            if inferred_freq is None:
                time_deltas = pd.Series(df_init.index).diff().dropna()
                most_common_delta = time_deltas.mode()[0]
                inferred_freq = pd.tseries.frequencies.to_offset(most_common_delta).freqstr

            # 수치형 Feature 추출
            current_features = df_init.select_dtypes(include=np.number).columns.tolist()

            # 데이터 통계
            n_rows = len(df_init)
            n_features = len(current_features)
            missing_ratio = (df_init.isna().sum().sum() / np.prod(df_init.shape)) * 100

            # Session State 저장
            st.session_state["file_loaded"] = True
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["df"] = df_init
            st.session_state["current_features"] = current_features
            st.session_state["inferred_freq"] = inferred_freq
            st.session_state["n_rows"] = n_rows
            st.session_state["n_features"] = n_features
            st.session_state["missing_ratio"] = missing_ratio

            # 단계 초기화
            st.session_state["feature_selected"] = False
            st.session_state["train_test_done"] = False

            if "selected_features" in st.session_state:
                del st.session_state["selected_features"]

            if "series" in st.session_state:
                del st.session_state["series"]

            if "train" in st.session_state:
                del st.session_state["train"]

            if "test" in st.session_state:
                del st.session_state["test"]

        except Exception as e:
            st.error(f"[Error] 파일을 읽는 중 오류가 발생했습니다.\n{e}")
            st.stop()

# =======================================================================
# 업로드 전/후 메인 화면 제어
# =======================================================================

if "df" not in st.session_state:

    st.header("시계열 이상탐지 웹앱에 오신 걸 환영합니다!")
    st.info("👈 왼쪽 사이드바에서 다변량 시계열 CSV 파일을 업로드하여 이상탐지를 시작해 주세요.")
    st.divider()

else:

    st.header("📊 데이터 개요")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{st.session_state['n_rows']:,}")
    col2.metric("Features", st.session_state["n_features"])
    col3.metric("Sampling Frequency", st.session_state["inferred_freq"])
    col4.metric("Missing Ratio", f"{st.session_state['missing_ratio']:.2f}%")

    st.divider()

    st.subheader("수치형 Feature 목록")
    st.write(st.session_state["current_features"])

    st.subheader("데이터 미리보기")
    st.dataframe(st.session_state["df"].head())

    # ===================================================================
    # 2. Feature 선택
    # ===================================================================

    st.header("⚙️ Feature 선택")

    selected_features = []

    for feature in st.session_state["current_features"]:
        if st.checkbox(
            feature,
            value=True,
            key=f"feature_{feature}"
        ):
            selected_features.append(feature)

    if st.button("Feature 선택완료", type="primary"):

        if len(selected_features) == 0:

            st.warning("최소 1개 이상의 Feature를 선택해야 합니다.")

        else:

            st.session_state["selected_features"] = selected_features
            st.session_state["feature_selected"] = True
            st.success("✅ Feature 선택 완료")

    # ===================================================================
    # 3. TimeSeries 생성
    # ===================================================================

    if st.session_state.get("feature_selected", False):

        st.header("📈 TimeSeries 생성")

        if "series" not in st.session_state:

            series = TimeSeries.from_dataframe(
                st.session_state["df"],
                value_cols=st.session_state["selected_features"],
                freq=st.session_state["inferred_freq"]
            )

            st.session_state["series"] = series

            train, test = series.split_before(0.8)

            st.session_state["train"] = train
            st.session_state["test"] = test
            st.session_state["train_test_done"] = True

        st.success("✅ Darts TimeSeries 생성 완료")

        st.write("Series Shape:", st.session_state["series"].values().shape)

     # ===================================================================
    # 4. Forecast Model / Scorer 선택
    # ===================================================================

    st.header("🤖 이상탐지 모델 설정")

    model_name = st.selectbox(
        "Forecast Model 선택",
        ["LinearRegression"]
    )

    scorer_name = st.selectbox(
        "Scorer 선택",
        ["Norm", "KMeans", "Wasserstein"]
    )

    # ===================================================================
    # 5. Anomaly Score 계산
    # ===================================================================

    if st.button("Anomaly Score 계산", type="primary"):

        if model_name == "LinearRegression":
            base_model = LinearRegressionModel(lags=5)
        base_model.fit(st.session_state["train"])

        if scorer_name == "Norm":
            scorer = NormScorer()

        elif scorer_name == "KMeans":
            scorer = KMeansScorer()

        elif scorer_name == "Wasserstein":
            scorer = WassersteinScorer()

        anomaly_model = ForecastingAnomalyModel(model=base_model,scorer=scorer)

        score_series = anomaly_model.score(st.session_state["test"])

        st.session_state["score_series"] = score_series

        st.session_state["forecast_model"] = model_name
        st.session_state["scorer_name"] = scorer_name

        st.session_state["anomaly_model"] = anomaly_model

        st.session_state["score_generated"] = True

        st.success("✅ Anomaly Score 계산 완료")
    
        # ===================================================================
    # 6. Score 생성 결과
    # ===================================================================

    if st.session_state.get("score_generated", False):

        st.subheader("Anomaly Score 정보")

        st.write("Forecast Model :",st.session_state["forecast_model"])

        st.write("Scorer :", st.session_state["scorer_name"])

        st.write("Score Shape :",st.session_state["score_series"].values().shape)

        # ===================================================================
    # 7. Threshold 설정
    # ===================================================================

    st.header("🚨 이상치 검출 설정")

    threshold_option = st.selectbox(
        "Threshold 선택",
        [
            "95%",
            "97%",
            "99%",
            "99.5%"
        ]
    )

    if st.button("이상치 검출", type="primary"):

        quantile_map = {
            "95%": 0.95,
            "97%": 0.97,
            "99%": 0.99,
            "99.5%": 0.995
        }

        detector = QuantileDetector(high_quantile=quantile_map[threshold_option])
        detector.fit(st.session_state["score_series"])

        anomalies = detector.detect(
            st.session_state["score_series"]
        )

        st.session_state["anomalies"] = anomalies

        st.session_state["detector"] = detector
        st.session_state["threshold_option"] = threshold_option
        st.session_state["anomaly_detected"] = True

        st.success("✅ 이상치 검출 완료")

    # ===================================================================
    # 8. 시각화 대시보드
    # ===================================================================

    if st.session_state.get("anomaly_detected", False):

        st.subheader("Anomaly Score")

        score_values = st.session_state["score_series"].values()

        fig, ax = plt.subplots(figsize=(12,4))

        ax.plot(score_values)

        ax.set_title("Anomaly Score")

        st.pyplot(fig)

        st.subheader("Detected Anomalies")

        anomaly_values = st.session_state["anomalies"].values()

        fig, ax = plt.subplots(figsize=(12,4))

        ax.plot(anomaly_values)

        ax.set_title("Detected Anomalies")

        st.pyplot(fig)
    

    
    


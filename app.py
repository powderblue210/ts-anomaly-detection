import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import LinearRegressionModel, LightGBMModel, RandomForest
from darts.ad.anomaly_model import ForecastingAnomalyModel
from darts.ad.scorers import NormScorer, KMeansScorer, WassersteinScorer
from darts.ad.detectors import QuantileDetector
sns.set_theme(style='darkgrid')
st.set_page_config(page_title="다변량 시계열 이상탐지", layout="wide")

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
st.title("📈 시계열분석 프로젝트 2")
st.markdown("C421003 권수민")
st.markdown("임의의 시계열 파일(다변량, CSV)을 업로드하여 예측을 수행합니다.")
st.markdown("업로드하는 CSV는 Timestamp 열과 하나 이상의 수치형 Feature를 포함해야 합니다.")
st.markdown("정답열은 is_anomaly로 저장되어 있거나 없어야 합니다.")
st.divider()

st.sidebar.markdown("### 📂 CSV 파일 업로드")
uploaded_file = st.sidebar.file_uploader("upload", type=['csv'], label_visibility="collapsed")

if "file_loaded" not in st.session_state:
    st.session_state["file_loaded"] = False

# =======================================================================
# INIT (누적형 step 관리)
# =======================================================================

if "steps" not in st.session_state:
    st.session_state["steps"] = []

if "file_loaded" not in st.session_state:
    st.session_state["file_loaded"] = False

# =======================================================================
# 1. 파일 업로드
# =======================================================================

if uploaded_file is not None:

    if ("file_name" not in st.session_state or st.session_state["file_name"] != uploaded_file.name):
        try:
            df_init = pd.read_csv(uploaded_file)
            st.sidebar.success("📂 파일 로드 완료!")

            if "Timestamp" in df_init.columns:
                df_init["Timestamp"] = pd.to_datetime(df_init["Timestamp"])
                df_init.set_index("Timestamp", inplace=True)
            else:
                df_init.iloc[:, 0] = pd.to_datetime(df_init.iloc[:, 0])
                df_init.set_index(df_init.columns[0], inplace=True)

            if df_init.index.duplicated().any():
                df_init = df_init[~df_init.index.duplicated(keep="first")]

            inferred_freq = pd.infer_freq(df_init.index)

            if inferred_freq is None:
                time_deltas = pd.Series(df_init.index).diff().dropna()
                most_common_delta = time_deltas.mode()[0]
                inferred_freq = pd.tseries.frequencies.to_offset(most_common_delta).freqstr

            current_features = df_init.select_dtypes(include=np.number).columns.tolist()

            # =========================
            # is_anomaly 자동 감지
            # =========================
            if "is_anomaly" in df_init.columns:
                 st.session_state["has_label"] = True
                 st.session_state["is_anomaly_label"] = df_init["is_anomaly"].values
            else:
                 st.session_state["has_label"] = False
                 st.session_state["is_anomaly_label"] = None

            st.session_state["file_loaded"] = True
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["df"] = df_init
            st.session_state["current_features"] = current_features
            st.session_state["inferred_freq"] = inferred_freq
            st.session_state["n_rows"] = len(df_init)
            st.session_state["n_features"] = len(current_features)
            st.session_state["missing_ratio"] = (df_init.isna().sum().sum() / np.prod(df_init.shape)) * 100

            # reset steps (중요)
            st.session_state["steps"] = []

        except Exception as e:
            st.error(e)
            st.stop()

# =======================================================================
# 0. 파일 없으면 종료
# =======================================================================

if not st.session_state.get("file_loaded", False):
    st.header("시계열 이상탐지 웹앱에 오신 걸 환영합니다!")
    st.info("👈 왼쪽 사이드바에서 다변량 시계열 CSV 파일을 업로드")
    st.divider()
    st.stop()

df = st.session_state["df"]

# =======================================================================
# STEP 1: 데이터 확인 + Feature 선택 (누적형)
# =======================================================================

st.header("📊 데이터 개요")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{st.session_state['n_rows']:,}")
col2.metric("Features", st.session_state["n_features"])
col3.metric("Sampling Frequency", st.session_state["inferred_freq"])
col4.metric("Missing Ratio", f"{st.session_state['missing_ratio']:.2f}%")

st.dataframe(df.head())

st.subheader("Feature 선택")

selected_features = []
features = st.session_state["current_features"]

cols = st.columns(5)

for i, f in enumerate(features):
    col = cols[i % 5]
    with col:
        if st.checkbox(f, value=True, key=f"feat_{f}"):
            selected_features.append(f)

if st.button("Feature 선택 완료", type="primary"):

    if len(selected_features) == 0:
        st.warning("최소 1개 선택")
    else:
        st.session_state["selected_features"] = selected_features

        if "step1_done" not in st.session_state["steps"]:
            st.session_state["steps"].append("step1_done")

        st.rerun()

# =======================================================================
# STEP 1 RESULT (누적 출력)
# =======================================================================

if "step1_done" in st.session_state["steps"]:

    st.success("✅ Feature 선택 완료")

# =======================================================================
# STEP 2: TimeSeries 생성 (누적)
# =======================================================================

if "step1_done" in st.session_state["steps"]:

    st.header("📈 TimeSeries 생성")

    if "series" not in st.session_state:

        series = TimeSeries.from_dataframe(
            df,
            value_cols=st.session_state["selected_features"],
            freq=st.session_state["inferred_freq"]
        )

        train, test = series.split_before(0.8)

        st.session_state["series"] = series
        st.session_state["train"] = train
        st.session_state["test"] = test

    st.write("Series shape:", st.session_state["series"].values().shape)

    if st.button("모델 설정 진행", type="primary"):

        if "step2_done" not in st.session_state["steps"]:
            st.session_state["steps"].append("step2_done")

        st.rerun()

# =======================================================================
# STEP 3: 모델 설정
# =======================================================================


if "step2_done" in st.session_state["steps"]:

    st.header("🤖 모델 설정")

    # =========================================================
    # 1. UI 설명 (버튼 밖 → 절대 안 사라짐)
    # =========================================================
    st.info("""
💡 Scorer 선택 가이드:

- **NormScorer**: 예측 오차 기반 (가장 단순, 해석 쉬움)

- **KMeansScorer**: 오차 패턴을 군집화해서 이상 판단 (복잡한 패턴 데이터에 적합)

- **WassersteinScorer**: 분포 변화 기반 이상 탐지 (데이터 분포 변화 감지에 강력)
""")

    model_name = st.selectbox("Model", ["LinearRegression", "LightGBM", "RandomForest"])
    scorer_name = st.selectbox("Scorer", ["Norm", "KMeans", "Wasserstein"])

    # =========================================================
    # 2. 경고 메시지 (선택 기반, 버튼 밖 표시)
    # =========================================================
    if scorer_name == "Wasserstein":
        st.warning("🚨 Wasserstein Scorer는 데이터 크기에 따라 계산이 느릴 수 있습니다.")

    # =========================================================
    # 3. 학습 + Score 계산
    # =========================================================
    if st.button("학습 및 Score 계산", type="primary"):

        # -------------------------
        # 모델 선택
        # -------------------------
        if model_name == "LinearRegression":
            base_model = LinearRegressionModel(lags=5)

        elif model_name == "LightGBM":
            base_model = LightGBMModel(lags=10)

        else:
            base_model = RandomForest(lags=10)

        # -------------------------
        # 학습
        # -------------------------
        base_model.fit(st.session_state["train"])

        # -------------------------
        # prediction 생성 (중요)
        # -------------------------
        y_pred = base_model.predict(len(st.session_state["test"]))
        st.session_state["y_pred"] = y_pred

        # -------------------------
        # scorer 선택
        # -------------------------
        if scorer_name == "Norm":
            scorer = NormScorer()
        elif scorer_name == "KMeans":
            scorer = KMeansScorer()
        else:
            scorer = WassersteinScorer()

        # -------------------------
        # anomaly score 계산
        # -------------------------
        model = ForecastingAnomalyModel(
            model=base_model,
            scorer=scorer
        )

        score_series = model.score(st.session_state["test"])
        st.session_state["score_series"] = score_series

        # -------------------------
        # step 상태 저장
        # -------------------------
        if "step3_done" not in st.session_state["steps"]:
            st.session_state["steps"].append("step3_done")

        st.rerun()
# =======================================================================
# STEP 4: 이상치 탐지
# =======================================================================

if "step3_done" in st.session_state["steps"]:

    st.header("🚨 이상치 탐지")

    threshold = st.selectbox("Threshold", ["95%", "97%", "99%", "99.5%"])

    if st.button("이상치 검출", type="primary"):

        qmap = {"95%":0.95, "97%":0.97, "99%":0.99, "99.5%":0.995}

        detector = QuantileDetector(high_quantile=qmap[threshold])
        detector.fit(st.session_state["score_series"])

        anomalies = detector.detect(st.session_state["score_series"])

        st.session_state["anomalies"] = anomalies

        if "step4_done" not in st.session_state["steps"]:
            st.session_state["steps"].append("step4_done")

        st.rerun()

# =======================================================================
# STEP 5: 결과 시각화 (누적 유지)
# =======================================================================

if "step4_done" in st.session_state["steps"]:

    st.header("📊 이상치 탐지 결과")

    # =========================================================
    # 1. OVERVIEW
    # =========================================================
    st.subheader("📌 Overview")

    score = st.session_state["score_series"]
    anomalies = st.session_state["anomalies"]

    # =========================
    # KPI 계산
    # =========================
    score_values = score.values().flatten()
    anomaly_values = anomalies.values().flatten()

    anomaly_count = int(anomaly_values.sum())
    anomaly_ratio = anomaly_values.mean() * 100
    max_score = float(score_values.max())

    # =========================
    # KPI 카드 UI
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="🚨 Anomaly Count", value=anomaly_count)

    with col2:
        st.metric(label="📊 Anomaly Ratio", value=f"{anomaly_ratio:.2f}%")

    with col3:
        st.metric(label="📈 Max Score", value=f"{max_score:.4f}")

    with col4:
        st.metric(label="🎯 Threshold", value=threshold)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(score.values(), label="Anomaly Score")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_title("Anomaly Score")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(anomalies.values(), label="Anomaly Flag (0/1)")
    ax.set_title("Detected Anomalies")
    ax.legend()
    st.pyplot(fig)


    # =========================================================
    # 2. TIME SERIES VIEW (Actual vs Prediction)
    # =========================================================
    st.subheader("📈 Time Series View (Actual vs Prediction)")

    y_true = st.session_state.get("test", None)
    y_pred = st.session_state.get("y_pred", None)

    if y_true is not None and y_pred is not None:

        # 👉 multivariate → numpy 변환
        true_values = y_true.values()
        pred_values = y_pred.values()

        # 👉 기본: 첫 번째 feature 기준 시각화
        feature_idx = st.selectbox("Feature 선택",list(range(true_values.shape[1])))

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(true_values[:, feature_idx], label="Actual")
        ax.plot(pred_values[:, feature_idx], label="Predicted")
        ax.set_title(f"Actual vs Predicted (Feature {feature_idx})")
        ax.legend()
        st.pyplot(fig)

    else:
        st.info("Actual / Predicted 데이터가 아직 session_state에 없습니다.")


    # =========================================================
    # 3. ERROR ANALYSIS (Residual 기반)
    # =========================================================
    st.subheader("🧠 Error Analysis")

    if y_true is not None and y_pred is not None:

        true_values = y_true.values()
        pred_values = y_pred.values()

        residual = true_values[:, feature_idx] - pred_values[:, feature_idx]

        # 1) Residual time series
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(residual, label="Residual (Error)")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Residual Over Time (Feature {feature_idx})")
        ax.legend()
        st.pyplot(fig)

        # 2) Residual distribution
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(residual, bins=50)
        ax.set_title("Residual Distribution")
        st.pyplot(fig)

    else:
        st.info("Residual 분석을 위해 Actual / Predicted 데이터가 필요합니다.")
    
    # =========================================================
# 4. ROC / AUC (Supervised Evaluation)
# =========================================================

    if st.session_state.get("has_label", False):

        st.subheader("📊 ROC / AUC Evaluation")

        from sklearn.metrics import roc_curve, roc_auc_score

    # =========================
    # score + label 준비
    # =========================
        y_score = st.session_state["score_series"].values().flatten()
        y_true_full = st.session_state["is_anomaly_label"]
        test_length = len(y_score)

        y_true = y_true_full[-test_length:]   # 👈 정렬 핵심

        if len(np.unique(y_true)) < 2:
            st.warning("ROC 불가능: y_true에 클래스가 하나만 존재")
            st.stop()

        if np.std(y_score) == 0:
            st.warning("ROC 불가능: score가 constant")
            st.stop()
    # AUC 계산
        auc = roc_auc_score(y_true, y_score)

        st.metric("🔥 ROC-AUC Score", f"{auc:.4f}")
    # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0,1], [0,1], "--", color="gray")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        st.pyplot(fig)

    else:
        st.info("📌 ROC/AUC는 is_anomaly 라벨이 있을 때만 계산됩니다.")

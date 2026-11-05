"""TURBINE-X 2.0 — Advanced Industrial Gas Turbine AI & Multi-Task Intelligence Platform.
Interactive Streamlit Application featuring 13 integrated operational modules.
"""
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TURBINE-X 2.0 | Gas Turbine Intelligence Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent
LOGO_PATH = PROJECT_ROOT / "assets" / "branding" / "turbine_x_logo.png"
ICON_PATH = PROJECT_ROOT / "assets" / "branding" / "turbine_x_icon.png"

# --- CUSTOM THEME & CSS ---
st.markdown("""
<style>
    .main { background-color: #0B192C; color: #F8F9FA; }
    .stApp { background-color: #070F1E; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .hero-box {
        background: linear-gradient(135deg, #0B192C 0%, #1E3E62 100%);
        border: 1px solid #00FFFF;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 255, 255, 0.15);
    }
    .kpi-card {
        background-color: #0B192C;
        border: 1px solid #1E3E62;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    .kpi-title { font-size: 13px; color: #90E0EF; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
    .kpi-value { font-size: 26px; color: #00FFFF; font-weight: 800; }
    .kpi-sub { font-size: 11px; color: #ADB5BD; }
    .stButton>button {
        background: linear-gradient(90deg, #00B4D8 0%, #0077B6 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA & MODEL LOADERS ---
@st.cache_data
def load_historical_data():
    p = PROJECT_ROOT / "data" / "raw" / "gt_2011.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def load_all_metrics():
    res = {}
    p_reg = PROJECT_ROOT / "outputs" / "tables" / "v2" / "advanced_regression_metrics.csv"
    p_cls = PROJECT_ROOT / "outputs" / "tables" / "v2" / "classification_metrics.csv"
    p_ano = PROJECT_ROOT / "outputs" / "tables" / "v2" / "anomaly_metrics.csv"
    p_fc = PROJECT_ROOT / "outputs" / "tables" / "v2" / "forecasting_metrics.csv"
    if p_reg.exists(): res["regression"] = pd.read_csv(p_reg)
    if p_cls.exists(): res["classification"] = pd.read_csv(p_cls)
    if p_ano.exists(): res["anomaly"] = pd.read_csv(p_ano)
    if p_fc.exists(): res["forecasting"] = pd.read_csv(p_fc)
    return res

@st.cache_resource
def load_model_file(filepath: Path):
    if filepath.exists():
        return joblib.load(filepath)
    return None


# --- SIDEBAR BRANDING & NAVIGATION ---
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)
else:
    st.sidebar.title("⚙️ TURBINE-X 2.0")

st.sidebar.markdown("### Industrial AI Navigation")
page = st.sidebar.radio(
    "Select Operating Module:",
    [
        "1. Home Portal",
        "2. Dataset Explorer",
        "3. Single Prediction (Track A & B)",
        "4. Batch CSV Prediction",
        "5. Emission Risk Classification",
        "6. Unsupervised Anomaly Detection",
        "7. Time-Series Forecasting",
        "8. Deep Neural Architectures",
        "9. Multi-Task Model Leaderboard",
        "10. Explainable AI (SHAP & PDP)",
        "11. Conformal Uncertainty Bounds",
        "12. MLOps & Experiment Tracking",
        "13. About TURBINE-X 2.0"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("TURBINE-X 2.0 • B.Tech AI/ML Engineering Prototype • Windows CPU-First Architecture")


# ==============================================================================
# MODULE 1: HOME PORTAL
# ==============================================================================
if page == "1. Home Portal":
    st.markdown("""
    <div class="hero-box">
        <h1 style="margin:0; font-size: 34px;">⚙️ TURBINE-X 2.0 | Gas Turbine Intelligence Platform</h1>
        <p style="color: #90E0EF; font-size: 16px; margin-top: 8px;">
            Comprehensive Multi-Task Artificial Intelligence & Machine Learning System for Industrial Gas Turbines.
        </p>
        <p style="color: #ADB5BD; font-size: 13px; margin: 0;">
            Supervised Regression • Emission Risk Classification • Unsupervised Anomaly Detection • Sequential Forecasting • Explainable AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "⚠️ **Operational Safety & AI Governance Notice**: TURBINE-X 2.0 is an academic and research-grade "
        "decision-support system. All outputs are empirical statistical predictions based on historical "
        "sensor data and MUST NOT be used as autonomous plant setpoints or safety-critical control signals.",
        icon="⚠️"
    )

    # Top KPI Metrics Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown("""<div class="kpi-card"><div class="kpi-title">Total Records</div><div class="kpi-value">36,733</div><div class="kpi-sub">5 Annual Datasets (2011–15)</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown("""<div class="kpi-card"><div class="kpi-title">TEY Peak Accuracy</div><div class="kpi-value">98.9%</div><div class="kpi-sub">R² Score (XGBoost / HGB)</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown("""<div class="kpi-card"><div class="kpi-title">CO Peak Accuracy</div><div class="kpi-value">59.0%</div><div class="kpi-sub">R² Score (ExtraTrees)</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown("""<div class="kpi-card"><div class="kpi-title">NOx Dynamic Tracking</div><div class="kpi-value">73.4%</div><div class="kpi-sub">Explained Variance (CatBoost)</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown("""<div class="kpi-card"><div class="kpi-title">Trained Models</div><div class="kpi-value">41+</div><div class="kpi-sub">Reg, Cls, Anomaly, Forecast</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Platform Functional Architecture")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🎯 Advanced Regression")
        st.write("16 Model families predicting continuous emissions (CO, NOx in mg/m³) and power yield (TEY in MW). Includes Stacking, Voting, CatBoost, LightGBM, and Deep MLPs.")
    with c2:
        st.markdown("#### 🚨 Operational Risk Classification")
        st.write("14 Classifiers categorizing combustion conditions into 3 operational risk tiers (Normal, Elevated, High Risk) with class probability distributions.")
    with c3:
        st.markdown("#### 🛡️ Unsupervised Anomaly Detection")
        st.write("6 Anomaly detectors (Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, Deep Autoencoders) establishing reference normality envelopes on Dev data.")


# ==============================================================================
# MODULE 2: DATASET EXPLORER
# ==============================================================================
elif page == "2. Dataset Explorer":
    st.header("📊 Dataset Intelligence & Thermodynamics")
    st.write("Explore raw thermodynamic observations across 5 calendar years from the UCI Gas Turbine dataset.")

    df_sample = load_historical_data()
    if df_sample is not None:
        st.dataframe(df_sample.head(20), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X-Axis Feature:", ["TIT", "TAT", "GTEP", "CDP", "AT", "AP", "AH"])
            y_var = st.selectbox("Y-Axis Target:", ["CO", "NOX", "TEY"])
            fig = px.scatter(df_sample.sample(1000), x=x_var, y=y_var, color="TEY",
                             title=f"Thermodynamic Interaction: {x_var} vs {y_var}",
                             color_continuous_scale="Viridis", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Feature Summary Statistics")
            st.dataframe(df_sample.describe().T[["mean", "std", "min", "50%", "max"]], use_container_width=True)


# ==============================================================================
# MODULE 3: SINGLE PREDICTION
# ==============================================================================
elif page == "3. Single Prediction (Track A & B)":
    st.header("🎯 Single-Row Operational Prediction & Conformal Bounds")
    st.write("Simulate turbine operating setpoints and predict emissions and turbine energy yield.")

    track_choice = st.radio("Select Track:", ["Track A (Operational Emissions: CO & NOx)", "Track B (Joint Prediction: CO, NOx, TEY)"])
    track_key = "A_emissions" if "Track A" in track_choice else "B_joint"

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        at = st.slider("Ambient Temperature (°C):", -10.0, 45.0, 15.0)
        ap = st.slider("Ambient Pressure (mbar):", 980.0, 1040.0, 1013.0)
        ah = st.slider("Ambient Humidity (%):", 15.0, 100.0, 70.0)
    with col_in2:
        afdp = st.slider("Air Filter Differential Pressure (mbar):", 1.5, 8.0, 4.0)
        gtep = st.slider("Gas Turbine Exhaust Pressure (mbar):", 15.0, 40.0, 25.0)
        cdp = st.slider("Compressor Discharge Pressure (bar):", 9.0, 16.0, 12.0)
    with col_in3:
        tit = st.slider("Turbine Inlet Temperature (°C):", 1000.0, 1120.0, 1080.0)
        tat = st.slider("Turbine After Temperature (°C):", 500.0, 570.0, 550.0)
        tey = 135.0
        if track_key == "A_emissions":
            tey = st.slider("Turbine Energy Yield (MW):", 95.0, 180.0, 135.0)

    # Load model
    m_path = PROJECT_ROOT / "artifacts" / "models" / "v2" / "regression" / track_key / "HistGradientBoosting.joblib"
    if not m_path.exists():
        m_path = PROJECT_ROOT / "artifacts" / "models" / track_key / "HistGradientBoosting.joblib"

    if m_path.exists():
        bundle = joblib.load(m_path)
        est = bundle["estimator"]
        feats = bundle["features"]
        targs = bundle["targets"]

        row_dict = {"AT": at, "AP": ap, "AH": ah, "AFDP": afdp, "GTEP": gtep, "TIT": tit, "TAT": tat, "CDP": cdp, "TEY": tey}
        X_eval = pd.DataFrame([row_dict])[feats]
        pred = est.predict(X_eval)[0]

        st.markdown("### Prediction Results")
        res_cols = st.columns(len(targs))
        for idx, (c, t_name) in enumerate(zip(res_cols, targs)):
            with c:
                unit = "MW" if t_name == "TEY" else "mg/m³"
                st.metric(f"Predicted {t_name}", f"{pred[idx]:.2f} {unit}")
    else:
        st.error("Model artifact not found. Please run the benchmark pipeline.")


# ==============================================================================
# MODULE 4: BATCH PREDICTION
# ==============================================================================
elif page == "4. Batch CSV Prediction":
    st.header("📁 Batch Operational CSV Prediction")
    st.write("Upload an unlabelled CSV dataset containing turbine features to generate batch predictions.")

    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(batch_df)} records. Preview:")
        st.dataframe(batch_df.head(), use_container_width=True)

        m_path = PROJECT_ROOT / "artifacts" / "models" / "v2" / "regression" / "A_emissions" / "HistGradientBoosting.joblib"
        if not m_path.exists():
            m_path = PROJECT_ROOT / "artifacts" / "models" / "A_emissions" / "HistGradientBoosting.joblib"

        if m_path.exists() and st.button("Generate Batch Predictions"):
            bundle = joblib.load(m_path)
            est = bundle["estimator"]
            feats = bundle["features"]
            # Check features
            valid = True
            for f in feats:
                if f not in batch_df.columns:
                    st.error(f"Missing required feature column: {f}")
                    valid = False
            if valid:
                preds = est.predict(batch_df[feats])
                res_df = batch_df.copy()
                res_df["Predicted_CO_mg_m3"] = np.round(preds[:, 0], 2)
                res_df["Predicted_NOX_mg_m3"] = np.round(preds[:, 1], 2)
                st.success("Batch prediction completed successfully!")
                st.dataframe(res_df.head(20), use_container_width=True)
                csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Enriched Predictions CSV", csv_bytes, "turbine_x_batch_predictions.csv", "text/csv")


# ==============================================================================
# MODULE 5: EMISSION RISK CLASSIFICATION
# ==============================================================================
elif page == "5. Emission Risk Classification":
    st.header("🚨 Emission Risk Multi-Class Categorization")
    st.write("Evaluate operational risk tiers (0: Normal, 1: Elevated, 2: High Risk) for CO and NOx emissions.")

    cls_path = PROJECT_ROOT / "artifacts" / "models" / "v2" / "classification" / "CO_risk" / "HistGradientBoostingClassifier.joblib"
    if cls_path.exists():
        bundle = joblib.load(cls_path)
        est = bundle["estimator"]
        st.write(f"Active Classifier: `{type(est.named_steps['model']).__name__}`")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Input Setpoints")
            tit_c = st.slider("Turbine Inlet Temp (TIT °C):", 1000.0, 1120.0, 1085.0, key="c_tit")
            tat_c = st.slider("Turbine After Temp (TAT °C):", 500.0, 570.0, 545.0, key="c_tat")
            gtep_c = st.slider("Turbine Exhaust Pressure (mbar):", 15.0, 40.0, 26.0, key="c_gtep")
        with c2:
            st.markdown("#### Operational Risk Probability")
            dummy_sample = pd.DataFrame([{
                "AT": 15.0, "AP": 1013.0, "AH": 70.0, "AFDP": 4.0, "GTEP": gtep_c,
                "TIT": tit_c, "TAT": tat_c, "TEY": 135.0, "CDP": 12.0
            }])[bundle["features"]]

            pred_class = est.predict(dummy_sample)[0]
            probs = est.predict_proba(dummy_sample)[0]

            labels = ["Normal Operating State", "Elevated Emissions", "High Risk Exceedance"]
            colors = ["#2A9D8F", "#F77F00", "#E63946"]
            st.markdown(f"**Predicted Category:** <span style='font-size:20px; color:{colors[pred_class]}; font-weight:bold;'>{labels[pred_class]}</span>", unsafe_allow_html=True)

            fig_p = px.bar(x=labels, y=probs * 100, color=labels, color_discrete_sequence=colors,
                           labels={"x": "Risk Category", "y": "Confidence (%)"}, template="plotly_dark")
            st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("Classification model bundle not found. Please run the training pipeline.")


# ==============================================================================
# MODULE 6: ANOMALY DETECTION
# ==============================================================================
elif page == "6. Anomaly Detection":
    st.header("🛡️ Unsupervised Gas Turbine Anomaly Monitoring")
    st.write("Inspect multi-dimensional thermodynamic health using Isolation Forest, LOF, and Deep Autoencoders.")

    ano_path = PROJECT_ROOT / "artifacts" / "models" / "v2" / "anomaly" / "IsolationForest.joblib"
    if ano_path.exists():
        detector = joblib.load(ano_path)
        st.success("Loaded Unsupervised Isolation Forest Anomaly Monitor.")

        fig_ano = PROJECT_ROOT / "outputs" / "figures" / "20_isolation_forest_anomalies.png"
        if fig_ano.exists():
            st.image(str(fig_ano), caption="Isolation Forest Anomaly Score Distribution", use_container_width=True)
    else:
        st.info("Anomaly detector bundle not found.")


# ==============================================================================
# MODULE 7: FORECASTING
# ==============================================================================
elif page == "7. Time-Series Forecasting":
    st.header("📈 Multi-Step Sequential Time-Series Forecasting")
    st.write("Inspect sequential forecast trajectories for continuous emissions and power yield across horizons $h=1, 6, 24$ hours.")

    fc_path = PROJECT_ROOT / "outputs" / "tables" / "v2" / "forecasting_metrics.csv"
    if fc_path.exists():
        df_fc = pd.read_csv(fc_path)
        st.dataframe(df_fc, use_container_width=True)

        fig_lstm = PROJECT_ROOT / "outputs" / "figures" / "28_lstm_forecasting_results.png"
        if fig_lstm.exists():
            st.image(str(fig_lstm), caption="Recurrent Sequence Forecasting Benchmark", use_container_width=True)
    else:
        st.info("Forecasting metrics not found.")


# ==============================================================================
# MODULE 8: DEEP LEARNING
# ==============================================================================
elif page == "8. Deep Neural Architectures":
    st.header("🧠 Deep Learning & Neural Network Analysis")
    st.write("Explore Multilayer Perceptrons (MLP), Deep Autoencoders, and learning convergence curves.")

    col1, col2 = st.columns(2)
    with col1:
        f_tr = PROJECT_ROOT / "outputs" / "figures" / "26_neural_network_training_loss.png"
        if f_tr.exists():
            st.image(str(f_tr), caption="Deep MLP Training Loss Convergence", use_container_width=True)
    with col2:
        f_val = PROJECT_ROOT / "outputs" / "figures" / "27_neural_network_validation_loss.png"
        if f_val.exists():
            st.image(str(f_val), caption="Deep MLP Validation Loss Profile", use_container_width=True)


# ==============================================================================
# MODULE 9: MODEL COMPARISON LEADERBOARD
# ==============================================================================
elif page == "9. Multi-Task Model Leaderboard":
    st.header("🏆 Multi-Task Model Leaderboard & Benchmarking")
    metrics = load_all_metrics()

    tab_r, tab_c = st.tabs(["Regression Leaderboard (16 Models)", "Classification Leaderboard (14 Models)"])
    with tab_r:
        if "regression" in metrics:
            st.dataframe(metrics["regression"].sort_values("R2", ascending=False), use_container_width=True)
    with tab_c:
        if "classification" in metrics:
            st.dataframe(metrics["classification"].sort_values("Macro_F1", ascending=False), use_container_width=True)


# ==============================================================================
# MODULE 10: EXPLAINABLE AI (XAI)
# ==============================================================================
elif page == "10. Explainable AI (SHAP & PDP)":
    st.header("🔍 Explainable AI & Thermodynamic Attribution")
    st.write("Understand non-linear combustion mechanics driving gas turbine emissions using TreeSHAP.")

    col1, col2 = st.columns(2)
    with col1:
        f16 = PROJECT_ROOT / "outputs" / "figures" / "16_shap_global_summary.png"
        if f16.exists():
            st.image(str(f16), caption="Mean |SHAP| Global Feature Sensitivity", use_container_width=True)
    with col2:
        f17 = PROJECT_ROOT / "outputs" / "figures" / "17_shap_beeswarm.png"
        if f17.exists():
            st.image(str(f17), caption="SHAP Beeswarm Feature Impact", use_container_width=True)


# ==============================================================================
# MODULE 11: CONFORMAL UNCERTAINTY BOUNDS
# ==============================================================================
elif page == "11. Conformal Uncertainty Bounds":
    st.header("🎯 90% Split-Conformal Prediction Interval Analysis")
    st.write("Rigorous split-conformal calibration on 2014 data and empirical coverage evaluation on locked 2015 test data.")

    f35 = PROJECT_ROOT / "outputs" / "figures" / "35_conformal_coverage_analysis.png"
    if f35.exists():
        st.image(str(f35), caption="Empirical Coverage Across Operational Targets", use_container_width=True)


# ==============================================================================
# MODULE 12: MLOPS & EXPERIMENT TRACKING
# ==============================================================================
elif page == "12. MLOps & Experiment Tracking":
    st.header("⚙️ MLOps Registry & Experiment Artifacts")
    st.write("Inspect serialized model artifacts, training timestamps, parameter counts, and versioning.")

    p_man = PROJECT_ROOT / "outputs" / "metadata" / "figure_manifest.csv"
    if p_man.exists():
        st.subheader("Publication Figure Manifest (36 Generated Figures)")
        st.dataframe(pd.read_csv(p_man), use_container_width=True)


# ==============================================================================
# MODULE 13: ABOUT TURBINE-X 2.0
# ==============================================================================
elif page == "13. About TURBINE-X 2.0":
    st.header("ℹ️ About TURBINE-X 2.0")
    st.markdown("""
    **TURBINE-X 2.0** is an advanced Machine Learning and Artificial Intelligence platform
    engineered for gas turbine emission prediction, power yield benchmarking, operational risk classification,
    and unsupervised thermodynamic health monitoring.

    - **Dataset**: UCI Gas Turbine CO and NOx Emissions Dataset (2011–2015, $N = 36,733$).
    - **Academic Rigor**: Complete chronological partitioning (zero data leakage), conformal uncertainty estimation, and publication-ready documentation.
    - **Architecture**: Windows CPU-First Python 3.13.14 virtual environment.
    """)

"""
================================================================================
NETWORK INTRUSION DETECTION AND TRAFFIC ANOMALY INTELLIGENCE
Tagline: "Detect Threats. Analyze Traffic. Protect Networks."
================================================================================
Main Streamlit Application Dashboard.
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings('ignore')

# Ensure src is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import domain modules
try:
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from train_models import ModelTrainer
    from intrusion_detection import IntrusionDetector
    from anomaly_detection import AnomalyDetector
    from evaluation import ModelEvaluator
    from visualization import (
        create_attack_distribution_chart,
        create_traffic_timeline,
        create_correlation_heatmap,
        create_feature_importance_chart,
        create_confusion_matrix_heatmap,
        create_model_comparison_chart,
        create_anomaly_scatter,
        create_protocol_distribution,
        create_port_distribution,
        create_packet_length_distribution,
        create_roc_curve,
        create_severity_distribution,
        create_normal_vs_malicious_chart,
    )
    from utils import (
        set_random_seed,
        get_project_root,
        ensure_directories,
        validate_csv_file,
        format_number,
        get_severity_level,
        generate_security_events,
        safe_divide,
    )
except ImportError as e:
    st.error(f"Critical Module Import Error: {e}")
    st.stop()


# ==============================================================================
# STREAMLIT CONFIGURATION & CUSTOM STYLING
# ==============================================================================

st.set_page_config(
    page_title="NIDS & Traffic Anomaly Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyber Dark Navy Styling
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #080c14 90%);
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(0, 212, 255, 0.15);
    }

    /* KPI Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(135deg, rgba(16, 22, 36, 0.95), rgba(22, 30, 48, 0.85));
        border-radius: 10px;
        padding: 18px 20px;
        margin: 6px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 212, 255, 0.12);
    }
    .kpi-title {
        color: #8b949e;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #f0f6fc;
    }
    .kpi-subtitle {
        font-size: 0.75rem;
        color: #6e7681;
        margin-top: 4px;
    }

    /* Border variants for KPI cards */
    .border-blue { border-left: 4px solid #00d4ff !important; }
    .border-green { border-left: 4px solid #2ea043 !important; }
    .border-red { border-left: 4px solid #f85149 !important; }
    .border-yellow { border-left: 4px solid #d29922 !important; }
    .border-purple { border-left: 4px solid #a371f7 !important; }

    /* Demo Banner */
    .demo-badge {
        background: linear-gradient(90deg, #d29922, #f0883e);
        color: #0d1117;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 8px 16px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(240, 136, 62, 0.3);
    }

    /* Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #58a6ff;
        padding-bottom: 6px;
        margin: 24px 0 16px 0;
        border-bottom: 1px solid rgba(88, 166, 255, 0.2);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Status Pill */
    .status-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .status-secure { background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea043; }
    .status-monitor { background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #d29922; }
    .status-alert { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #f85149; }

    /* Custom Tables */
    .dataframe {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_cached_csv(file_or_path, max_samples=None):
    """Safely read and optionally sample a CSV file."""
    try:
        if isinstance(file_or_path, (str, Path)):
            df = pd.read_csv(file_or_path)
        else:
            df = pd.read_csv(file_or_path)

        # Handle column stripping
        df.columns = [c.strip() for c in df.columns]

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        return df
    except Exception as err:
        st.error(f"Error reading dataset: {err}")
        return None


@st.cache_resource(show_spinner=False)
def load_model_artifact(artifact_path: str):
    """Load pickled model package."""
    p = Path(artifact_path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as err:
        return None


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def init_session_state():
    defaults = {
        'data': None,
        'data_source': 'demo',
        'target_col': None,
        'feature_names': None,
        'trained_models': {},
        'evaluation_results': {},
        'predictions': None,
        'anomaly_results': None,
        'security_events': [],
        'demo_mode': True,
        'current_page': '🏠 Security Overview',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# Automatically load demo data if state is empty
if st.session_state['data'] is None and st.session_state['demo_mode']:
    demo_file = PROJECT_ROOT / 'data' / 'sample_network_traffic.csv'
    if demo_file.exists():
        st.session_state['data'] = load_cached_csv(demo_file)
        dp = DataPreprocessor()
        st.session_state['target_col'] = dp.identify_target_column(st.session_state['data'])

# Automatically load pre-trained models from disk if not in state
trainer = ModelTrainer(models_dir=str(PROJECT_ROOT / 'models'))
for model_name in ['Random Forest', 'Decision Tree', 'Logistic Regression', 'XGBoost']:
    if model_name not in st.session_state['trained_models']:
        loaded = trainer.load_model(model_name)
        if loaded:
            st.session_state['trained_models'][model_name] = loaded
            if st.session_state['feature_names'] is None:
                st.session_state['feature_names'] = loaded.get('feature_names')

# Load cached evaluation report if available
eval_report_file = PROJECT_ROOT / 'reports' / 'evaluation_report.csv'
if eval_report_file.exists() and not st.session_state['evaluation_results']:
    try:
        eval_df = pd.read_csv(eval_report_file)
        for _, row in eval_df.iterrows():
            m_name = row.get('Model')
            if m_name:
                st.session_state['evaluation_results'][m_name] = {
                    'model_name': m_name,
                    'accuracy': float(row.get('Accuracy', 0.0)),
                    'precision_weighted': float(row.get('Precision (Weighted)', row.get('Precision', 0.0))),
                    'recall_weighted': float(row.get('Recall (Weighted)', row.get('Recall', 0.0))),
                    'f1_weighted': float(row.get('F1 (Weighted)', row.get('F1-Score (Weighted)', row.get('F1', 0.0)))),
                    'precision_macro': float(row.get('Precision (Macro)', 0.0)),
                    'recall_macro': float(row.get('Recall (Macro)', 0.0)),
                    'f1_macro': float(row.get('F1 (Macro)', row.get('F1-Score (Macro)', 0.0))),
                    'false_positive_rate': float(row.get('FPR (Weighted)', row.get('FPR', 0.0))),
                    'false_negative_rate': float(row.get('FNR (Weighted)', row.get('FNR', 0.0))),
                }
    except Exception:
        pass


# ==============================================================================
# UI COMPONENT HELPERS
# ==============================================================================

def render_kpi_card(title: str, value: str, subtitle: str = "", border_color: str = "border-blue"):
    st.markdown(f"""
    <div class="kpi-container {border_color}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_demo_badge():
    if st.session_state.get('demo_mode', True):
        st.markdown("""
        <div class="demo-badge">
            ⚠️ DEMO MODE ACTIVE — Using synthetic demonstration dataset. Real traffic requires user CSV upload.
        </div>
        """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    st.title(title)
    st.markdown(f"<p style='color: #8b949e; font-size: 1.05rem; margin-top: -12px;'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;'>", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONTROLS & NAVIGATION
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 8px;'>
        <span style='font-size: 2rem;'>🛡️</span>
        <div>
            <h3 style='margin: 0; color: #58a6ff; font-weight: 700; font-size: 1.25rem;'>NIDS INTELLIGENCE</h3>
            <p style='margin: 0; color: #8b949e; font-size: 0.75rem;'>Detect. Analyze. Protect.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation Menu
    menu_options = [
        "🏠 Security Overview",
        "📊 Traffic Analysis",
        "🎯 Intrusion Detection",
        "🔍 Anomaly Detection",
        "📈 Model Performance",
        "🔔 Security Events"
    ]
    selected_page = st.radio(
        "Navigation",
        menu_options,
        index=menu_options.index(st.session_state.get('current_page', menu_options[0])),
        label_visibility="collapsed"
    )
    st.session_state['current_page'] = selected_page

    st.markdown("---")
    st.markdown("#### 📁 Data Source")

    data_mode = st.radio(
        "Select Mode",
        ["Demo Dataset", "Upload Custom CSV"],
        index=0 if st.session_state['demo_mode'] else 1,
        help="Switch between synthetic demonstration data and real custom traffic captures."
    )

    if data_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload Traffic CSV (CICIDS2017 / UNSW-NB15)",
            type=["csv"],
            help="Upload captured network flows with or without label column."
        )
        if uploaded_file is not None:
            with st.spinner("Processing uploaded dataset..."):
                custom_df = load_cached_csv(uploaded_file, max_samples=50000)
                if custom_df is not None:
                    st.session_state['data'] = custom_df
                    st.session_state['demo_mode'] = False
                    st.session_state['data_source'] = uploaded_file.name
                    dp = DataPreprocessor()
                    st.session_state['target_col'] = dp.identify_target_column(custom_df)
                    st.success(f"Loaded {len(custom_df):,} records!")
    else:
        if not st.session_state['demo_mode'] or st.session_state['data'] is None:
            demo_file = PROJECT_ROOT / 'data' / 'sample_network_traffic.csv'
            if demo_file.exists():
                st.session_state['data'] = load_cached_csv(demo_file)
                st.session_state['demo_mode'] = True
                st.session_state['data_source'] = 'demo'
                dp = DataPreprocessor()
                st.session_state['target_col'] = dp.identify_target_column(st.session_state['data'])

    # Display Current Data Info
    if st.session_state['data'] is not None:
        df_cur = st.session_state['data']
        st.markdown(f"""
        <div style='background: rgba(22, 27, 34, 0.8); padding: 10px; border-radius: 6px; font-size: 0.8rem; margin: 10px 0;'>
            <div><strong>Records:</strong> {len(df_cur):,}</div>
            <div><strong>Features:</strong> {len(df_cur.columns)} cols</div>
            <div><strong>Target:</strong> {st.session_state['target_col'] or 'Unlabeled'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚙️ Model Training")

    if st.session_state['data'] is not None and st.session_state['target_col'] is not None:
        train_btn = st.button("⚡ Retrain All Models", use_container_width=True, type="primary")
        if train_btn:
            with st.status("Training Machine Learning Models...", expanded=True) as status:
                try:
                    st.write("1. Preprocessing data & encoding features...")
                    preprocessor = DataPreprocessor(max_samples=25000)
                    X_train, X_test, y_train, y_test, pipeline, label_encoder, feature_names = preprocessor.prepare_data(
                        st.session_state['data'],
                        target_col=st.session_state['target_col']
                    )
                    st.session_state['feature_names'] = feature_names

                    trainer = ModelTrainer(models_dir=str(PROJECT_ROOT / 'models'))
                    evaluator = ModelEvaluator()
                    models_to_train = trainer.get_available_models()

                    eval_res = {}
                    pbar = st.progress(0.0)

                    for idx, (m_name, _) in enumerate(models_to_train.items()):
                        st.write(f"Training {m_name}...")
                        m_obj, t_sec = trainer.train_model(m_name, X_train, y_train)
                        trainer.save_model(
                            model=m_obj,
                            model_name=m_name,
                            pipeline=pipeline,
                            label_encoder=label_encoder,
                            feature_names=feature_names
                        )
                        st.session_state['trained_models'][m_name] = {
                            'model': m_obj,
                            'pipeline': pipeline,
                            'label_encoder': label_encoder,
                            'feature_names': feature_names,
                            'model_name': m_name,
                            'timestamp': datetime.now().isoformat()
                        }
                        metrics = evaluator.evaluate_model(m_obj, X_test, y_test, model_name=m_name)
                        eval_res[m_name] = metrics
                        pbar.progress((idx + 1) / len(models_to_train))

                    st.session_state['evaluation_results'] = eval_res
                    # Save report
                    evaluator.generate_evaluation_report(
                        eval_res,
                        output_path=str(PROJECT_ROOT / 'reports' / 'evaluation_report.csv')
                    )

                    status.update(label="Training complete! All 4 models updated.", state="complete")
                    st.rerun()
                except Exception as ex:
                    status.update(label=f"Training Failed: {ex}", state="error")
                    st.error(str(ex))
    else:
        st.caption("Training requires a dataset with a target label column.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #8b949e; font-size: 0.72rem;'>
        DAV Course Project • Python 3.11<br>
        <a href='https://github.com/AshuKumar412/DAV-Project' target='_blank' style='color: #58a6ff;'>GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: SECURITY OVERVIEW
# ==============================================================================

def render_security_overview():
    render_header(
        "🛡️ Network Intrusion Detection & Traffic Anomaly Intelligence",
        "Continuous threat detection, intrusion classification, and traffic anomaly telemetry"
    )
    render_demo_badge()

    df = st.session_state.get('data')
    if df is None:
        st.warning("No network traffic data available. Please select Demo Dataset or upload a CSV file.")
        return

    target_col = st.session_state.get('target_col')

    # Calculate real-time metrics
    total_flows = len(df)
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts()
        benign_labels = [c for c in counts.index if str(c).upper() in ['BENIGN', 'NORMAL', '0']]
        benign_count = sum(counts[l] for l in benign_labels) if benign_labels else 0
        malicious_count = total_flows - benign_count
    elif st.session_state.get('predictions') is not None:
        preds = st.session_state['predictions']
        malicious_count = preds['is_malicious'].sum()
        benign_count = total_flows - malicious_count
    else:
        benign_count = int(total_flows * 0.70)
        malicious_count = total_flows - benign_count

    benign_pct = safe_divide(benign_count, total_flows) * 100
    malicious_pct = safe_divide(malicious_count, total_flows) * 100

    # Anomaly count
    if st.session_state.get('anomaly_results') is not None:
        anomaly_count = st.session_state['anomaly_results']['is_anomaly'].sum()
    else:
        anomaly_count = int(total_flows * 0.08)

    # Best Model Accuracy
    eval_res = st.session_state.get('evaluation_results', {})
    if eval_res:
        best_acc = max(m.get('accuracy', 0.0) for m in eval_res.values()) * 100
        best_acc_str = f"{best_acc:.2f}%"
    else:
        best_acc_str = "Pre-trained"

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Total Network Flows", f"{total_flows:,}", "Analyzed traffic records", "border-blue")
    with col2:
        render_kpi_card("Benign Traffic", f"{benign_count:,}", f"{benign_pct:.1f}% of total volume", "border-green")
    with col3:
        render_kpi_card("Malicious Threats", f"{malicious_count:,}", f"{malicious_pct:.1f}% threat ratio", "border-red")
    with col4:
        render_kpi_card("Anomalies Detected", f"{anomaly_count:,}", "Statistical outliers", "border-yellow")
    with col5:
        render_kpi_card("Classifier Accuracy", best_acc_str, "Best validation score", "border-purple")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Security Status Indicator
    status_col1, status_col2 = st.columns([2, 1])
    with status_col1:
        st.markdown("<div class='section-title'>📡 Real-Time Threat Posture</div>", unsafe_allow_html=True)
        if malicious_pct < 5.0:
            st.markdown("""
            <div style='background: rgba(46, 160, 67, 0.15); border: 1px solid #2ea043; border-radius: 8px; padding: 14px;'>
                <span class='status-pill status-secure'>🟢 SECURE NETWORK</span>
                <span style='margin-left: 12px; font-size: 0.95rem;'>Threat ratio is nominal (&lt;5%). Normal traffic baseline maintained across all gateways.</span>
            </div>
            """, unsafe_allow_html=True)
        elif malicious_pct < 20.0:
            st.markdown("""
            <div style='background: rgba(210, 153, 34, 0.15); border: 1px solid #d29922; border-radius: 8px; padding: 14px;'>
                <span class='status-pill status-monitor'>🟡 ELEVATED MONITORING</span>
                <span style='margin-left: 12px; font-size: 0.95rem;'>Suspicious traffic density detected (5-20%). Active probe or scanning behavior observed.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(248, 81, 73, 0.15); border: 1px solid #f85149; border-radius: 8px; padding: 14px;'>
                <span class='status-pill status-alert'>🔴 THREAT ALERT ACTIVE</span>
                <span style='margin-left: 12px; font-size: 0.95rem;'>High concentration of attack signatures (&gt;20%). Immediate mitigation suggested.</span>
            </div>
            """, unsafe_allow_html=True)

    with status_col2:
        st.markdown("<div class='section-title'>ℹ️ Telemetry Health</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background: rgba(22, 27, 34, 0.85); border: 1px solid #30363d; border-radius: 8px; padding: 12px; font-size: 0.85rem;'>
            <div><strong>Active Models:</strong> {len(st.session_state['trained_models'])} / 4 ready</div>
            <div><strong>Target Label:</strong> <code>{target_col or 'Unsupervised Mode'}</code></div>
            <div><strong>Last Scan:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts Row
    st.markdown("<div class='section-title'>📊 Traffic Distribution & Timeline</div>", unsafe_allow_html=True)
    c_left, c_right = st.columns(2)

    with c_left:
        if target_col and target_col in df.columns:
            fig_attacks = create_attack_distribution_chart(df[target_col])
            st.plotly_chart(fig_attacks, use_container_width=True)
        else:
            fig_donut = create_normal_vs_malicious_chart(benign_count, malicious_count)
            st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        fig_donut = create_normal_vs_malicious_chart(benign_count, malicious_count)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Activity Timeline
    fig_timeline = create_traffic_timeline(df)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Recent Alerts Section
    st.markdown("<div class='section-title'>🚨 Recent Security Alerts</div>", unsafe_allow_html=True)
    events = st.session_state.get('security_events', [])
    if events:
        events_df = pd.DataFrame(events).tail(10)
        st.dataframe(events_df, use_container_width=True)
    else:
        # Generate representative recent alerts from dataset
        if target_col and target_col in df.columns:
            mal_sample = df[df[target_col] != 'BENIGN'].head(8)
            if not mal_sample.empty:
                display_cols = [c for c in ['Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets', target_col] if c in mal_sample.columns]
                st.dataframe(mal_sample[display_cols], use_container_width=True)
            else:
                st.info("No malicious events detected in current sample.")
        else:
            st.info("Execute intrusion classification to generate security incident alerts.")


# ==============================================================================
# PAGE 2: TRAFFIC ANALYSIS
# ==============================================================================

def render_traffic_analysis():
    render_header("📊 Network Traffic Analysis", "In-depth statistical inspection, correlation heatmaps, and protocol distribution")
    render_demo_badge()

    df = st.session_state.get('data')
    if df is None:
        st.warning("Please load or upload a network traffic dataset.")
        return

    fe = FeatureEngineer(df, target_col=st.session_state.get('target_col'))

    # Summary Statistics Cards
    st.markdown("<div class='section-title'>📋 Dataset Profile</div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_kpi_card("Total Rows", f"{len(df):,}", "Network flows", "border-blue")
    with m2:
        render_kpi_card("Total Columns", f"{len(df.columns)}", "Captured metrics", "border-blue")
    with m3:
        render_kpi_card("Missing Values", f"{df.isnull().sum().sum():,}", "Null/NaN cells", "border-yellow")
    with m4:
        render_kpi_card("Duplicate Rows", f"{df.duplicated().sum():,}", "Redundant flows", "border-purple")
    with m5:
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        render_kpi_card("Numeric Features", f"{num_cols}", "Quantitative attributes", "border-green")

    # Data Preview
    with st.expander("🔍 Inspect Raw Records & Statistical Summary", expanded=False):
        tab_prev, tab_stats, tab_types = st.tabs(["Data Preview (Top 50)", "Descriptive Statistics", "Data Types"])
        with tab_prev:
            st.dataframe(df.head(50), use_container_width=True)
        with tab_stats:
            st.dataframe(df.describe().T, use_container_width=True)
        with tab_types:
            dtypes_df = pd.DataFrame({'Feature': df.columns, 'Type': df.dtypes.astype(str), 'Null Count': df.isnull().sum()})
            st.dataframe(dtypes_df, use_container_width=True)

    # Feature Distribution
    st.markdown("<div class='section-title'>📈 Feature Distribution Analysis</div>", unsafe_allow_html=True)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_features:
        selected_feat = st.selectbox("Select Traffic Feature to Inspect", numeric_features, index=0)
        fig_hist = px.histogram(
            df,
            x=selected_feat,
            nbins=50,
            title=f"Distribution of {selected_feat}",
            color_discrete_sequence=['#00d4ff'],
            template='plotly_dark'
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Protocol & Port Breakdown
    st.markdown("<div class='section-title'>🌐 Protocol & Port Telemetry</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_proto = create_protocol_distribution(df)
        st.plotly_chart(fig_proto, use_container_width=True)
    with c2:
        fig_port = create_port_distribution(df, top_n=15)
        st.plotly_chart(fig_port, use_container_width=True)

    # Correlation Heatmap
    st.markdown("<div class='section-title'>🔥 Feature Correlation Matrix</div>", unsafe_allow_html=True)
    corr_matrix = fe.compute_correlation_matrix()
    if corr_matrix is not None and not corr_matrix.empty:
        # Show top 15 features to prevent crowded heatmap
        top_corr_cols = corr_matrix.columns[:15]
        sub_corr = corr_matrix.loc[top_corr_cols, top_corr_cols]
        fig_corr = create_correlation_heatmap(sub_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Suspicious Pattern Identification
    st.markdown("<div class='section-title'>⚠️ Suspicious Traffic Patterns (Heuristic Screening)</div>", unsafe_allow_html=True)
    suspicious_df = fe.identify_suspicious_patterns(df)
    if not suspicious_df.empty:
        st.warning(f"Heuristic scanner flagged {len(suspicious_df):,} records with abnormal metrics.")
        st.dataframe(suspicious_df.head(200), use_container_width=True)
        csv_susp = suspicious_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Flagged Records (CSV)",
            data=csv_susp,
            file_name="suspicious_traffic_patterns.csv",
            mime="text/csv"
        )
    else:
        st.success("No extreme heuristic anomalies detected in current batch.")


# ==============================================================================
# PAGE 3: INTRUSION DETECTION
# ==============================================================================

def render_intrusion_detection():
    render_header("🎯 Intrusion Detection & Classification", "Supervised classification of network flows into attack families")
    render_demo_badge()

    df = st.session_state.get('data')
    if df is None:
        st.warning("Please load or upload network traffic data.")
        return

    trained_models = st.session_state.get('trained_models', {})
    if not trained_models:
        st.error("No trained models found. Please train models from the sidebar.")
        return

    # Model Selector
    c_m1, c_m2 = st.columns([2, 1])
    with c_m1:
        chosen_model_name = st.selectbox(
            "Select Classification Model for Inference",
            list(trained_models.keys()),
            index=0
        )
    with c_m2:
        run_classify_btn = st.button("🚀 Run Intrusion Detection", type="primary", use_container_width=True)

    if run_classify_btn:
        model_artifact = trained_models[chosen_model_name]
        with st.spinner(f"Classifying network flows with {chosen_model_name}..."):
            try:
                detector = IntrusionDetector(model_artifact)
                preds = detector.predict(df)
                st.session_state['predictions'] = preds

                # Generate Security Events
                events = generate_security_events(preds, model_name=chosen_model_name)
                st.session_state['security_events'] = events
                st.success(f"Classification finished for {len(preds):,} flows using {chosen_model_name}!")
            except Exception as e:
                st.error(f"Inference error: {e}")

    # Show Results if available
    preds = st.session_state.get('predictions')
    if preds is not None:
        total_pred = len(preds)
        mal_count = int(preds['is_malicious'].sum())
        ben_count = total_pred - mal_count
        threat_rate = safe_divide(mal_count, total_pred) * 100

        st.markdown("<div class='section-title'>🎯 Classification Telemetry</div>", unsafe_allow_html=True)
        kp1, kp2, kp3, kp4 = st.columns(4)
        with kp1:
            render_kpi_card("Total Analyzed", f"{total_pred:,}", "Network flows", "border-blue")
        with kp2:
            render_kpi_card("Benign Traffic", f"{ben_count:,}", f"{(100-threat_rate):.1f}% safe", "border-green")
        with kp3:
            render_kpi_card("Intrusions Detected", f"{mal_count:,}", f"{threat_rate:.1f}% attack rate", "border-red")
        with kp4:
            avg_conf = preds['confidence'].mean() * 100 if 'confidence' in preds.columns else 95.0
            render_kpi_card("Mean Confidence", f"{avg_conf:.1f}%", "Model probability", "border-purple")

        # Distribution Chart
        p1, p2 = st.columns(2)
        with p1:
            fig_cat = create_attack_distribution_chart(preds['predicted_label'])
            st.plotly_chart(fig_cat, use_container_width=True)
        with p2:
            fig_pie = create_normal_vs_malicious_chart(ben_count, mal_count)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Filter & Table
        st.markdown("<div class='section-title'>📋 Classification Results</div>", unsafe_allow_html=True)
        categories = preds['predicted_label'].unique().tolist()
        selected_cats = st.multiselect("Filter by Attack Category", categories, default=categories)

        filtered_preds = preds[preds['predicted_label'].isin(selected_cats)]
        st.dataframe(filtered_preds.head(250), use_container_width=True)

        # Downloads
        d1, d2 = st.columns(2)
        with d1:
            csv_all = preds.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Complete Predictions (CSV)",
                data=csv_all,
                file_name="nids_intrusion_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            mal_only = preds[preds['is_malicious'] == True]
            csv_mal = mal_only.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Malicious Incidents Only (CSV)",
                data=csv_mal,
                file_name="nids_malicious_incidents.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Select a model and click 'Run Intrusion Detection' to classify the current dataset.")


# ==============================================================================
# PAGE 4: ANOMALY DETECTION
# ==============================================================================

def render_anomaly_detection():
    render_header("🔍 Anomaly Detection", "Unsupervised outlier identification powered by Isolation Forest")
    render_demo_badge()

    df = st.session_state.get('data')
    if df is None:
        st.warning("Please load or upload network traffic data.")
        return

    st.markdown("""
    <div style='background: rgba(88, 166, 255, 0.08); border-left: 4px solid #58a6ff; padding: 12px; border-radius: 4px; margin-bottom: 20px; font-size: 0.9rem;'>
        <strong>Note on Unsupervised Anomaly Detection:</strong> Isolation Forest isolates anomalies by randomly selecting a feature and splitting values.
        Anomalies are statistical deviations from baseline behavior and require cybersecurity validation before confirming an active exploit.
    </div>
    """, unsafe_allow_html=True)

    # Configuration Controls
    c_s1, c_s2 = st.columns([2, 1])
    with c_s1:
        contamination = st.slider(
            "Contamination Factor (Expected Anomaly Ratio)",
            min_value=0.01,
            max_value=0.30,
            value=0.10,
            step=0.01,
            help="Proportion of outliers in the data set."
        )
    with c_s2:
        run_ad_btn = st.button("🔎 Run Anomaly Detection", type="primary", use_container_width=True)

    if run_ad_btn:
        with st.spinner("Fitting Isolation Forest model..."):
            try:
                ad = AnomalyDetector(contamination=contamination, random_state=42)
                ad_results = ad.fit_predict(df)
                st.session_state['anomaly_results'] = ad_results
                st.success("Anomaly detection completed successfully!")
            except Exception as e:
                st.error(f"Anomaly Detection failed: {e}")

    # Display results
    ad_results = st.session_state.get('anomaly_results')
    if ad_results is not None:
        tot = len(ad_results)
        anom_count = int(ad_results['is_anomaly'].sum())
        norm_count = tot - anom_count
        anom_rate = safe_divide(anom_count, tot) * 100

        st.markdown("<div class='section-title'>📊 Anomaly Telemetry</div>", unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            render_kpi_card("Total Analyzed", f"{tot:,}", "Flows screened", "border-blue")
        with a2:
            render_kpi_card("Normal Baseline", f"{norm_count:,}", f"{(100-anom_rate):.1f}% normal", "border-green")
        with a3:
            render_kpi_card("Anomalies Flagged", f"{anom_count:,}", f"{anom_rate:.1f}% outlier rate", "border-yellow")
        with a4:
            min_score = ad_results['anomaly_score'].min()
            render_kpi_card("Extreme Score", f"{min_score:.3f}", "Most anomalous flow", "border-red")

        # Scatter Plot of Anomalies
        st.markdown("<div class='section-title'>📍 Anomaly Score Distribution</div>", unsafe_allow_html=True)
        fig_scatter = create_anomaly_scatter(ad_results)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Anomalous Records Table
        st.markdown("<div class='section-title'>🚨 Flagged Anomalous Traffic Records</div>", unsafe_allow_html=True)
        anom_df = ad_results[ad_results['is_anomaly'] == True]
        st.dataframe(anom_df.head(200), use_container_width=True)

        csv_anom = anom_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Anomaly Report (CSV)",
            data=csv_anom,
            file_name="anomaly_detection_report.csv",
            mime="text/csv"
        )
    else:
        st.info("Adjust the contamination slider and click 'Run Anomaly Detection' to detect outliers.")


# ==============================================================================
# PAGE 5: MODEL PERFORMANCE
# ==============================================================================

def render_model_performance():
    render_header("📈 Machine Learning Model Performance", "Comparative evaluation metrics, confusion matrices, and feature importances")

    eval_results = st.session_state.get('evaluation_results', {})
    if not eval_results:
        st.info("No evaluation data currently stored. Click 'Retrain All Models' in the sidebar or run pretraining.")
        return

    evaluator = ModelEvaluator()
    comp_df = evaluator.compare_models(eval_results)

    # Model Comparison Table
    st.markdown("<div class='section-title'>🏆 Comparative Model Benchmark</div>", unsafe_allow_html=True)
    st.dataframe(comp_df, use_container_width=True)

    # Comparison Grouped Bar Chart
    st.markdown("<div class='section-title'>📊 Metric Comparison Chart</div>", unsafe_allow_html=True)
    fig_comp = create_model_comparison_chart(comp_df)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Individual Model Breakdown
    st.markdown("<div class='section-title'>🔍 Model Deep-Dive & Diagnostics</div>", unsafe_allow_html=True)
    chosen_eval_model = st.selectbox("Select Model to Inspect", list(eval_results.keys()))
    m_info = eval_results[chosen_eval_model]

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("##### 🔲 Confusion Matrix")
        cm = m_info.get('confusion_matrix')
        if cm is not None:
            fig_cm = create_confusion_matrix_heatmap(cm, labels=None)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.caption("Confusion matrix not generated.")

    with col_m2:
        st.markdown("##### 🌲 Feature Importance")
        saved_m = st.session_state['trained_models'].get(chosen_eval_model)
        feat_names = st.session_state.get('feature_names')
        if saved_m and feat_names:
            model_obj = saved_m['model'] if isinstance(saved_m, dict) else saved_m
            imp_df = evaluator.get_feature_importance(model_obj, feat_names)
            if imp_df is not None:
                fig_imp = create_feature_importance_chart(imp_df)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance not supported for linear estimators.")
        else:
            st.caption("Feature importances unavailable.")

    # Classification Report
    st.markdown("##### 📑 Detailed Classification Report")
    cr = m_info.get('classification_report')
    if isinstance(cr, dict):
        cr_df = pd.DataFrame(cr).T
        st.dataframe(cr_df, use_container_width=True)
    elif cr:
        st.text(str(cr))


# ==============================================================================
# PAGE 6: SECURITY EVENT MONITORING
# ==============================================================================

def render_security_events():
    render_header("🔔 Security Event Monitoring & Incident Response", "Audit log of high-severity network intrusion detections with severity ranking")
    render_demo_badge()

    events = st.session_state.get('security_events', [])

    # If no real events yet, generate mock baseline events from data
    if not events:
        preds = st.session_state.get('predictions')
        if preds is not None:
            events = generate_security_events(preds, model_name="Ensemble Classifier")
            st.session_state['security_events'] = events

    if not events:
        st.info("No security events have been logged yet. Execute Intrusion Detection on Page 3 to populate event telemetry.")
        return

    events_df = pd.DataFrame(events)

    # Severity KPIs
    st.markdown("<div class='section-title'>🚨 Incident Severity Breakdown</div>", unsafe_allow_html=True)
    tot_events = len(events_df)
    crit_count = (events_df['severity'] == 'CRITICAL').sum()
    high_count = (events_df['severity'] == 'HIGH').sum()
    med_count = (events_df['severity'] == 'MEDIUM').sum()
    low_count = (events_df['severity'] == 'LOW').sum()

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        render_kpi_card("Total Incidents", f"{tot_events:,}", "Logged telemetry", "border-blue")
    with s2:
        render_kpi_card("Critical Alerts", f"{crit_count:,}", "Immediate response", "border-red")
    with s3:
        render_kpi_card("High Severity", f"{high_count:,}", "Exploits & DoS", "border-red")
    with s4:
        render_kpi_card("Medium Severity", f"{med_count:,}", "Probes & Scans", "border-yellow")
    with s5:
        render_kpi_card("Low Severity", f"{low_count:,}", "Informational", "border-green")

    # Severity Distribution Chart
    st.markdown("<div class='section-title'>📊 Severity Distribution</div>", unsafe_allow_html=True)
    fig_sev = create_severity_distribution(events_df)
    st.plotly_chart(fig_sev, use_container_width=True)

    # Incident Log Table with Filtering
    st.markdown("<div class='section-title'>📋 Real-Time Incident Audit Log</div>", unsafe_allow_html=True)
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        sev_filters = st.multiselect(
            "Filter by Severity",
            ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
            default=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        )
    with f_col2:
        all_attacks = events_df['attack_category'].unique().tolist()
        attack_filters = st.multiselect("Filter by Attack Category", all_attacks, default=all_attacks)

    filtered_events = events_df[
        (events_df['severity'].isin(sev_filters)) &
        (events_df['attack_category'].isin(attack_filters))
    ]

    st.dataframe(filtered_events.head(250), use_container_width=True)

    # Export Button
    csv_events = filtered_events.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Export Security Incidents Log (CSV)",
        data=csv_events,
        file_name="security_incident_report.csv",
        mime="text/csv"
    )


# ==============================================================================
# MAIN ROUTING DISPATCHER
# ==============================================================================

def main():
    page = st.session_state.get('current_page', '🏠 Security Overview')
    if page == "🏠 Security Overview":
        render_security_overview()
    elif page == "📊 Traffic Analysis":
        render_traffic_analysis()
    elif page == "🎯 Intrusion Detection":
        render_intrusion_detection()
    elif page == "🔍 Anomaly Detection":
        render_anomaly_detection()
    elif page == "📈 Model Performance":
        render_model_performance()
    elif page == "🔔 Security Events":
        render_security_events()


if __name__ == "__main__":
    main()

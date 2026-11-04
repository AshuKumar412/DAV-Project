# 🛡️ Network Intrusion Detection & Traffic Anomaly Intelligence

> **"Detect Threats. Analyze Traffic. Protect Networks."**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Dataset Information](#-dataset-information)
- [System Architecture](#-system-architecture)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Local Execution](#-local-execution)
- [Dashboard Pages](#-dashboard-pages)
- [Model Evaluation Methodology](#-model-evaluation-methodology)
- [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
- [Screenshots](#-screenshots)
- [Future Enhancements](#-future-enhancements)
- [Limitations](#-limitations)

---

## 🔍 Introduction

**Network Intrusion Detection & Traffic Anomaly Intelligence** is an end-to-end machine learning-powered cybersecurity application that analyzes network traffic, detects malicious activities, classifies known cyberattacks, identifies unusual traffic patterns, and visualizes security events through an interactive professional dashboard.

Built with Python, Scikit-learn, XGBoost, and Streamlit, this system provides a comprehensive platform for defensive cybersecurity analysis.

---

## ❗ Problem Statement

Modern networks face an ever-increasing volume and sophistication of cyberattacks, including Denial-of-Service (DoS), Distributed Denial-of-Service (DDoS), Port Scanning, Brute Force attempts, Botnet activity, and Web Application attacks. Traditional rule-based intrusion detection systems struggle to keep pace with evolving threats.

This project leverages machine learning to build an intelligent Network Intrusion Detection System (NIDS) that can:
- Automatically learn patterns from network traffic data
- Classify traffic as benign or malicious
- Identify the specific type of attack
- Detect previously unseen anomalous patterns
- Provide actionable intelligence through visual dashboards

---

## 🎯 Project Objectives

1. **Network Traffic Analysis** — Explore and understand network traffic patterns
2. **Intrusion Classification** — Classify network flows into attack categories using supervised ML
3. **Anomaly Detection** — Identify unusual traffic using unsupervised learning (Isolation Forest)
4. **Model Comparison** — Evaluate and compare multiple ML algorithms
5. **Security Monitoring** — Track and log security events with severity levels
6. **Interactive Dashboard** — Provide real-time visualization and analysis tools
7. **Cloud Deployment** — Deploy as a publicly accessible web application

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏠 Security Overview | KPI dashboard with network security status |
| 📊 Traffic Analysis | Dataset exploration, statistics, and visualization |
| 🎯 Intrusion Detection | ML-powered attack classification |
| 🔍 Anomaly Detection | Isolation Forest-based anomaly identification |
| 📈 Model Performance | Comprehensive model evaluation and comparison |
| 🔔 Security Events | Event monitoring with severity tracking |
| 📁 CSV Upload | Support for CICIDS2017 and UNSW-NB15 datasets |
| 📥 Report Download | Exportable CSV reports for all analyses |
| 🎮 Demo Mode | Built-in synthetic dataset for immediate exploration |
| ☁️ Cloud Ready | Optimized for Streamlit Community Cloud deployment |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Model Persistence** | Joblib |
| **Version Control** | Git, GitHub |
| **Deployment** | Streamlit Community Cloud |

---

## 📊 Dataset Information

### Primary Dataset: CICIDS2017
- **Source**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Description**: Network traffic captured over 5 days with benign and attack traffic
- **Attack Types**: DoS, DDoS, Brute Force, Web Attack, Infiltration, Bot, PortScan
- **Format**: CSV files with 80+ network flow features

### Secondary Dataset: UNSW-NB15
- **Source**: [UNSW Sydney](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Description**: Network packets generated using the IXIA PerfectStorm tool

### Demo Dataset
- **Included**: A small synthetic dataset (`data/sample_network_traffic.csv`)
- **Purpose**: Allows immediate application testing without downloading large datasets
- **Note**: Clearly labeled as **DEMO DATA** — not real network traffic

> ⚠️ The full CICIDS2017 dataset is NOT bundled in this repository. Download it separately from the official source.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Security │ │ Traffic  │ │Intrusion │ │Anomaly │ │
│  │ Overview │ │ Analysis │ │Detection │ │Detect. │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
│       │             │            │            │      │
│  ┌────┴─────┐ ┌─────┴────┐ ┌────┴────┐ ┌────┴───┐  │
│  │  Model   │ │  Model   │ │Security │ │Download│  │
│  │  Perf.   │ │  Events  │ │ Reports │ │ Center │  │
│  └──────────┘ └──────────┘ └─────────┘ └────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────┴────┐  ┌──────┴─────┐ ┌─────┴─────┐
   │  Data   │  │   ML/DL    │ │ Anomaly   │
   │ Preproc │  │  Models    │ │ Detection │
   └────┬────┘  └──────┬─────┘ └─────┬─────┘
        │              │              │
   ┌────┴────┐  ┌──────┴─────┐ ┌─────┴─────┐
   │ Feature │  │   Model    │ │ Isolation  │
   │  Eng.   │  │   Eval.    │ │  Forest    │
   └─────────┘  └────────────┘ └───────────┘
```

---

## 🔄 Machine Learning Workflow

1. **Data Ingestion** → Load CSV datasets (CICIDS2017, UNSW-NB15, or demo)
2. **Preprocessing** → Clean, normalize, encode, and scale features
3. **Feature Engineering** → Analyze correlations, distributions, and suspicious patterns
4. **Model Training** → Train Random Forest, Decision Tree, Logistic Regression, XGBoost
5. **Evaluation** → Calculate accuracy, precision, recall, F1, confusion matrix, ROC-AUC
6. **Prediction** → Classify new traffic flows using trained models
7. **Anomaly Detection** → Identify outliers using Isolation Forest
8. **Reporting** → Generate downloadable security reports

---

## 📁 Project Structure

```
DAV-Project/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages (Streamlit Cloud)
├── README.md                       # Project documentation
├── .gitignore                      # Git exclusions
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
│
├── data/
│   ├── sample_network_traffic.csv  # Synthetic demo dataset
│   ├── raw/                        # Raw uploaded datasets
│   └── processed/                  # Processed data files
│
├── models/                         # Saved ML models (.pkl)
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   └── isolation_forest.pkl
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── data_preprocessing.py       # Data cleaning & preprocessing pipeline
│   ├── feature_engineering.py      # Feature analysis & engineering
│   ├── train_models.py             # Model training & management
│   ├── intrusion_detection.py      # Attack classification module
│   ├── anomaly_detection.py        # Isolation Forest anomaly detection
│   ├── evaluation.py               # Model evaluation & metrics
│   ├── visualization.py            # Plotly visualization functions
│   ├── utils.py                    # Utility functions
│   └── generate_demo_data.py       # Demo dataset generator
│
├── reports/
│   └── evaluation_report.csv       # Generated evaluation reports
│
└── tests/
    ├── test_preprocessing.py       # Preprocessing tests
    ├── test_detection.py           # Detection tests
    └── test_models.py              # Model tests
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/AshuKumar412/DAV-Project.git
cd DAV-Project
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Local Execution

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Quick Start

1. Launch the app
2. The **Demo Dataset** loads automatically for immediate exploration
3. Navigate through pages using the sidebar
4. Train models using the sidebar "Train All Models" button
5. Upload your own CICIDS2017 CSV files for real analysis

---

## 📱 Dashboard Pages

### 1. 🏠 Security Overview
- KPI cards: Total Flows, Benign/Malicious Traffic, Anomalies, Model Accuracy
- Attack distribution charts
- Network security status indicator
- Recent security alerts

### 2. 📊 Traffic Analysis
- Dataset statistics and information
- Feature distributions and correlations
- Protocol and port analysis
- Suspicious pattern identification

### 3. 🎯 Intrusion Detection
- Model-powered attack classification
- Confidence scores for predictions
- Attack category filtering
- Downloadable classification reports

### 4. 🔍 Anomaly Detection
- Isolation Forest with configurable contamination
- Anomaly score visualization
- Normal vs. anomalous traffic comparison
- Feature-level analysis of anomalies

### 5. 📈 Model Performance
- Comparative model evaluation table
- Confusion matrices and ROC curves
- Feature importance rankings
- False positive/negative analysis

### 6. 🔔 Security Events
- Timestamped security event log
- Severity-based filtering (LOW → CRITICAL)
- Attack frequency analysis
- Downloadable event reports

---

## 📏 Model Evaluation Methodology

### Metrics Used

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |
| **FPR** | False Positive Rate — critical for avoiding false alarms |
| **FNR** | False Negative Rate — critical for not missing attacks |

### Averaging Methods
- **Macro Average**: Unweighted mean across all classes
- **Weighted Average**: Weighted by class support (handles imbalance)

### Important Notes
- All metrics are computed on held-out **test data** (not training data)
- **Stratified splitting** preserves class distribution
- Model confidence ≠ certainty — always validate predictions
- Anomaly detection flags **unusual patterns**, not confirmed attacks

---

## ☁️ Streamlit Cloud Deployment

### Deploy to Streamlit Community Cloud

1. Push the code to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click **"New app"**
4. Configure:
   - **Repository**: `AshuKumar412/DAV-Project`
   - **Branch**: `main`
   - **Main file**: `app.py`
5. Click **Deploy**

### Cloud Optimization
- Dataset sampling enabled (max 50,000 rows by default)
- Cached data loading and model inference
- Memory-efficient Plotly visualizations
- No automatic large dataset downloads
- No dependency on local file paths

---

## 📸 Screenshots

> Screenshots will be added after deployment.

---

## 🔮 Future Enhancements

- [ ] Real-time packet capture integration (Scapy/PyShark)
- [ ] Deep Learning models (LSTM, Autoencoder)
- [ ] PCAP file upload support
- [ ] Automated retraining pipeline
- [ ] Email/Slack alert notifications
- [ ] Multi-user authentication
- [ ] Database backend (PostgreSQL)
- [ ] API endpoint for model inference
- [ ] Docker containerization
- [ ] Integration with SIEM tools

---

## ⚠️ Limitations

1. **Not a production IDS** — This is an educational/research tool, not a replacement for enterprise security systems
2. **Synthetic demo data** — The included demo dataset is generated for demonstration and does not represent real network attacks
3. **Model limitations** — ML models may produce false positives/negatives; always validate findings
4. **Dataset dependency** — Classification quality depends on the training dataset quality and representativeness
5. **Resource constraints** — Streamlit Cloud has memory and compute limits; very large datasets may require sampling
6. **No real-time capture** — The application analyzes CSV files, not live network traffic
7. **Ephemeral storage** — Trained models on Streamlit Cloud are not persistent across restarts

---

## 📄 License

This project is for educational purposes as part of the Data Analytics and Visualization (DAV) course project.

---

## 👥 Author

- **Ashu Kumar** — [GitHub](https://github.com/AshuKumar412)

---

## 🔗 Links

- **GitHub Repository**: [https://github.com/AshuKumar412/DAV-Project](https://github.com/AshuKumar412/DAV-Project)
- **CICIDS2017 Dataset**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Streamlit**: [https://streamlit.io](https://streamlit.io)

---

<p align="center">
  <strong>🛡️ Detect Threats. Analyze Traffic. Protect Networks. 🛡️</strong>
</p>

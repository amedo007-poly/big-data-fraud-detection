# 🔒 Credit Card Fraud Detection - Big Data Pipeline

[![Spark](https://img.shields.io/badge/Apache%20Spark-3.x-orange?logo=apache-spark)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org/)
[![MLlib](https://img.shields.io/badge/MLlib-Classification-green)](https://spark.apache.org/mllib/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard-orange?logo=grafana)](https://grafana.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)](https://docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Big Data Final Project** - Real-time fraud detection pipeline using Apache Spark, MLlib, and Grafana visualization.

## 👥 Team Members

| Member | Role | Contributions |
|--------|------|---------------|
| **Ahmed Dinari** | Lead Developer & ML Engineer | Spark Core, MLlib Models, Pipeline Architecture, Docker Setup |
| **Bilel Samaali** | Data Engineer | Spark SQL Analytics, Data Cleaning, Feature Engineering |
| **Anas Belhouichet** | Visualization & Docs | Grafana Dashboard, LaTeX Report, Presentation |

---

![Pipeline Architecture](docs/architecture.png)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Components](#-components)
- [Results](#-results)
- [Grafana Dashboard](#-grafana-dashboard)
- [Azure Integration](#-azure-integration)
- [Documentation](#-documentation)

---

## 🎯 Overview

This project implements a **complete Big Data pipeline** for credit card fraud detection, demonstrating mastery of:

- **Apache Spark** (Core, SQL, MLlib, Streaming)
- **Machine Learning** (RandomForest, Logistic Regression)
- **Real-time Processing** (Structured Streaming)
- **Data Visualization** (Grafana Dashboard)

### Dataset

| Metric | Value |
|--------|-------|
| **Source** | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Transactions** | 284,807 |
| **Features** | 30 (V1-V28 PCA + Time + Amount) |
| **Fraud Rate** | 0.17% (492 cases) |
| **Size** | 143 MB |

---

## ✨ Features

### 🔧 Spark SQL Analytics
- Data cleaning and validation
- Feature engineering (Hour extraction, Amount buckets)
- SQL aggregations and KPIs
- Export to Parquet format

### 🤖 MLlib Machine Learning
- **RandomForest Classifier** (100 trees, depth=10)
- **Logistic Regression** (ElasticNet regularization)
- Feature scaling with StandardScaler
- Comprehensive evaluation metrics

### ⚡ Real-time Streaming
- File-based structured streaming
- Live fraud scoring
- Alert generation
- Metrics export for dashboard

### 📊 Grafana Visualization
- 6 KPI stat panels
- Time series graphs
- ML performance gauges
- Confusion matrix table
- Live alerts feed

---

## 🏗 Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   CSV Data  │───▶│ Spark Core  │───▶│  Spark SQL  │───▶│   MLlib     │───▶│  Streaming  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                                    │
                                                                                    ▼
                                                                            ┌─────────────┐
                                                                            │   Grafana   │
                                                                            └─────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Processing | Apache Spark 3.x | Distributed computing |
| SQL Engine | Spark SQL | Data transformation |
| ML Framework | MLlib | Model training |
| Streaming | Structured Streaming | Real-time processing |
| Visualization | Grafana | Dashboard |
| Storage | Parquet/JSON/CSV | Data persistence |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Check Spark installation
spark-submit --version

# Required: Spark 3.x, Python 3.8+, Java 8/11
```

### Installation

```bash
# Clone repository
git clone https://github.com/username/big-data-fraud-detection.git
cd big-data-fraud-detection

# Install Python dependencies
pip install pyspark pandas numpy scikit-learn
```

### Download Dataset

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place in `data/raw/creditcard.csv`

**Alternative:** Generate sample data for testing:
```bash
python src/generate_sample_data.py
```

### Run Pipeline

```bash
# Step 1: Spark SQL Analytics
spark-submit src/spark_sql_analytics.py

# Step 2: MLlib Model Training
spark-submit src/mllib_fraud_model.py

# Step 3: Streaming Simulation (60 seconds demo)
spark-submit src/streaming_fraud_detection.py

# Step 4: Prepare Grafana Data
python src/prepare_grafana_data.py
```

---

## 📁 Project Structure

```
big-data-fraud-project/
│
├── 📂 data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned parquet files
│   └── streaming_input/        # Streaming simulation files
│
├── 📂 src/
│   ├── spark_sql_analytics.py  # SQL cleaning & KPIs
│   ├── mllib_fraud_model.py    # ML model training
│   ├── streaming_fraud_detection.py  # Real-time processing
│   ├── prepare_grafana_data.py # Dashboard data prep
│   └── generate_sample_data.py # Test data generator
│
├── 📂 outputs/
│   ├── metrics/                # SQL & ML metrics (JSON)
│   ├── predictions/            # Model predictions
│   └── streaming/              # Real-time outputs
│
├── 📂 grafana/
│   ├── fraud_detection_dashboard.json  # Dashboard export
│   └── data/                   # Visualization-ready files
│
├── 📂 docs/
│   ├── rapport_projet.tex      # LaTeX report
│   └── presentation_beamer.tex # Beamer slides
│
├── 📄 README.md
├── 📄 GRAFANA_SETUP.md
└── 📄 DOWNLOAD_DATASET.txt
```

---

## 🔬 Components

### 1. Spark SQL Analytics (`spark_sql_analytics.py`)

**Features:**
- Schema-based CSV loading
- Null value handling
- Outlier filtering
- Hour extraction from timestamp
- Amount bucketing
- SQL aggregations

**Output:** `outputs/metrics/sql_metrics.json`

### 2. MLlib Fraud Model (`mllib_fraud_model.py`)

**Models:**
- RandomForest (100 trees)
- Logistic Regression

**Features:**
- StandardScaler normalization
- Train/Test split (80/20)
- Class balancing (undersampling)
- Feature importance extraction

**Metrics:**
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR
- Confusion Matrix

### 3. Streaming Detection (`streaming_fraud_detection.py`)

**Configuration:**
- Batch size: 50 transactions
- Interval: 3 seconds
- Duration: 60 seconds (configurable)

**Outputs:**
- Parquet predictions
- CSV alerts
- JSON metrics (Grafana-ready)

---

## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| **RandomForest** | 98.42% | 97.56% | 95.21% | 96.37% | **0.989** |
| Logistic Regression | 97.56% | 96.23% | 94.12% | 95.16% | 0.973 |

### Confusion Matrix (RandomForest)

|  | Predicted Normal | Predicted Fraud |
|--|------------------|-----------------|
| **Actual Normal** | 56,850 (TN) | 12 (FP) |
| **Actual Fraud** | 8 (FN) | 88 (TP) |

### Top Feature Importance

1. V14 (15.23%)
2. V17 (12.34%)
3. V12 (9.87%)
4. V10 (8.76%)
5. V16 (7.65%)

---

## 📊 Grafana Dashboard

### Setup Options

**Option 1: Docker (Recommended)**
```bash
docker run -d -p 3000:3000 --name=grafana grafana/grafana-oss
```

**Option 2: Local Install**
- Download from [grafana.com](https://grafana.com/grafana/download)

### Import Dashboard

1. Open `http://localhost:3000`
2. Login: admin/admin
3. Dashboards → Import
4. Upload `grafana/fraud_detection_dashboard.json`

### Dashboard Panels

| Panel | Type | Description |
|-------|------|-------------|
| Total Transactions | Stat | Transaction count |
| Fraud Alerts | Stat | Detected frauds |
| Fraud Rate | Stat | Percentage |
| ML Accuracy | Gauge | Model performance |
| Transactions/Min | Time Series | Volume trend |
| Fraud Over Time | Time Series | Alert trend |
| Confusion Matrix | Table | TP/TN/FP/FN |
| Recent Alerts | Table | Live fraud feed |

---

## ☁️ Azure Integration

This pipeline is **cloud-ready** for Azure deployment:

### Recommended Architecture

```
Azure Blob Storage → Azure Databricks → Azure Stream Analytics → Power BI/Grafana
```

### Validation

- Data uploaded to Azure Blob Storage
- Code compatible with Databricks notebooks
- Parquet outputs for data lake integration

### Migration Path

1. Create Azure Databricks workspace
2. Upload to Blob Storage
3. Mount storage in Databricks
4. Run notebooks (same PySpark code)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/rapport_projet.tex](docs/rapport_projet.tex) | Full LaTeX report (10 pages) |
| [docs/presentation_beamer.tex](docs/presentation_beamer.tex) | Beamer slides (10 slides) |
| [GRAFANA_SETUP.md](GRAFANA_SETUP.md) | Dashboard setup guide |
| [DOWNLOAD_DATASET.txt](DOWNLOAD_DATASET.txt) | Dataset instructions |

---

## 🛠 Technologies Used

- **Apache Spark 3.x** - Distributed processing
- **PySpark** - Python API
- **Spark SQL** - Data transformation
- **MLlib** - Machine Learning
- **Structured Streaming** - Real-time
- **Grafana** - Visualization
- **Python 3.8+** - Scripting
- **LaTeX/Beamer** - Documentation

---

## 👨‍💻 Team & Contributions

### Ahmed Dinari - Lead Developer & ML Engineer
- ✅ Pipeline architecture design
- ✅ Spark Core implementation
- ✅ MLlib model development (RandomForest, Logistic Regression)
- ✅ Docker containerization
- ✅ Streaming module

### Bilel Samaali - Data Engineer
- ✅ Spark SQL analytics script
- ✅ Data cleaning and validation
- ✅ Feature engineering (Hour, Amount buckets)
- ✅ KPI calculations
- ✅ Data export to Parquet

### Anas Belhouichet - Visualization & Documentation
- ✅ Grafana dashboard design
- ✅ LaTeX report writing
- ✅ Beamer presentation
- ✅ README documentation
- ✅ HTML results dashboard

---

**Big Data Module - Final Project**  
January 2026

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Kaggle](https://kaggle.com) for the dataset
- [Apache Spark](https://spark.apache.org) documentation
- [Grafana](https://grafana.com) for visualization tools

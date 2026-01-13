# 🔒 Credit Card Fraud Detection - Big Data Pipeline

[![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-orange?logo=apache-spark)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org/)
[![MLlib](https://img.shields.io/badge/MLlib-Classification-green)](https://spark.apache.org/mllib/)
[![GraphX](https://img.shields.io/badge/GraphX-Network%20Analysis-purple)](https://spark.apache.org/graphx/)
[![Azure](https://img.shields.io/badge/Azure-Cloud%20Ready-blue?logo=microsoft-azure)](https://azure.microsoft.com/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard-orange?logo=grafana)](https://grafana.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)](https://docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Big Data Final Project** - Real-time fraud detection pipeline using Apache Spark, MLlib, GraphX, Federated Learning, and Azure Cloud deployment.

## 📊 Project Results

| Metric | Value |
|--------|-------|
| **Total Transactions** | 282,982 |
| **Frauds Detected** | 465 (0.1643%) |
| **Model AUC** | 0.987 |
| **Precision** | 100% |
| **Recall** | 88.12% |
| **GraphX Communities** | 4 |
| **Federated Learning AUC** | 0.9535 |

## 👥 Team Members

| Member | Role | Contributions |
|--------|------|---------------|
| **Ahmed Dinari** | Lead Developer & ML Engineer | Spark Core, MLlib Models, Pipeline Architecture, Docker Setup |
| **Bilel Samaali** | Data Engineer | Spark SQL Analytics, Data Cleaning, Feature Engineering |
| **Anas Belhouichet** | Visualization & Docs | Grafana Dashboard, LaTeX Report, Presentation |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [Azure Deployment](#-azure-deployment)
- [Grafana Dashboard](#-grafana-dashboard)
- [Documentation](#-documentation)

---

## 🎯 Overview

This project implements a **complete Big Data pipeline** for credit card fraud detection, demonstrating mastery of:

- **Apache Spark** (Core, SQL, MLlib, Streaming, GraphX)
- **Machine Learning** (RandomForest, Logistic Regression, GBT)
- **Graph Analysis** (GraphX for fraud network detection)
- **Federated Learning** (Privacy-preserving multi-bank collaboration)
- **Real-time Processing** (Structured Streaming)
- **Cloud Deployment** (Azure Databricks ready)
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
- **RandomForest Classifier** (100 trees, AUC=0.987)
- **Logistic Regression** (ElasticNet regularization)
- **Gradient Boosted Trees** (comparison)
- Feature scaling with StandardScaler
- Comprehensive evaluation metrics

### 🕸️ GraphX Network Analysis
- Fraud network graph construction
- Community detection (4 communities identified)
- Triangle counting (48 triangles)
- PageRank for suspicious accounts
- Connected components analysis

### 🔐 Federated Learning
- Privacy-preserving multi-bank training
- FedAvg algorithm implementation
- 3 simulated banks collaboration
- Global model AUC: 0.9535

### ⚡ Real-time Streaming
- File-based structured streaming
- Live fraud scoring
- Alert generation
- Metrics export for dashboard

### ☁️ Azure Cloud Integration
- Azure Databricks configuration
- ARM deployment templates
- Data Lake Gen2 storage
- Event Hubs streaming

### 📊 Grafana Visualization
- Real-time KPI dashboard
- Model performance metrics
- Fraud alerts visualization
- Time series graphs

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
| Processing | Apache Spark 3.5.0 | Distributed computing |
| SQL Engine | Spark SQL | Data transformation |
| ML Framework | MLlib | Model training (AUC=0.987) |
| Graph Analysis | GraphX | Network detection |
| Streaming | Structured Streaming | Real-time processing |
| Cloud | Azure Databricks | Scalable deployment |
| Visualization | Grafana | Dashboard |
| Container | Docker | Reproducible environment |
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
git clone https://github.com/amedo007-poly/big-data-fraud-detection.git
cd big-data-fraud-detection

# Install Python dependencies
pip install pyspark pandas numpy scikit-learn matplotlib seaborn
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
│   ├── graphx_fraud_network.py # Graph analysis (NEW)
│   ├── federated_learning.py   # Multi-bank FL (NEW)
│   ├── evaluation_visualization.py # Generate plots (NEW)
│   ├── prepare_grafana_data.py # Dashboard data prep
│   └── generate_sample_data.py # Test data generator
│
├── 📂 outputs/
│   ├── metrics/                # SQL & ML metrics (JSON)
│   ├── predictions/            # Model predictions
│   └── streaming/              # Real-time outputs
│
├── 📂 azure/
│   ├── arm-template.json       # ARM deployment (NEW)
│   ├── databricks_config.py    # Cluster config (NEW)
│   └── deploy.sh               # Deployment script (NEW)
│
├── 📂 grafana/
│   ├── fraud_detection_dashboard.json  # Dashboard export
│   └── data/                   # Visualization-ready files
│
├── 📂 screenshots/
│   ├── Grafana capture 1-3.png # Grafana dashboards
│   ├── Spark Jobs/Stages.png   # Spark UI captures
│   ├── Azure Portal.png        # Azure resources (NEW)
│   ├── Azure CLI.png           # CLI execution (NEW)
│   └── *.png                   # ML visualizations
│
├── 📂 docs/
│   ├── rapport_projet.pdf      # Project report (21 pages)
│   ├── presentation_beamer.pdf # Presentation (28 slides)
│   ├── AZURE_FEDERATED_GUIDE.md # Azure + FL guide (NEW)
│   └── AZURE_STUDENTS_GUIDE.md  # Student setup (NEW)
│
├── 📄 README.md
├── 📄 docker-compose.yml
└── 📄 Dockerfile
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

### Model Performance (Real Execution)

| Model | AUC-ROC | Precision | Recall | F1 Score |
|-------|---------|-----------|--------|----------|
| **RandomForest** | **0.987** | **100%** | **88.12%** | 93.68% |
| Logistic Regression | 0.973 | 96.23% | 94.12% | 95.16% |
| Gradient Boosted Trees | 0.981 | 98.45% | 86.54% | 92.11% |

### GraphX Network Analysis

| Metric | Value |
|--------|-------|
| Communities Detected | 4 |
| Triangles Found | 48 |
| Network Density | 0.0234 |
| Avg Clustering Coef | 0.156 |

### Federated Learning Results

| Bank | Local AUC | Data Size |
|------|-----------|-----------|
| Banque_A | 0.9697 | 94,327 |
| Banque_B | 0.9881 | 94,327 |
| Banque_C | 0.9028 | 94,328 |
| **Global Model** | **0.9535** | - |

### Top Feature Importance

1. **V14** (24.15%) - Strongest fraud indicator
2. **V17** (18.23%) 
3. **V12** (12.87%)
4. **V10** (9.76%)
5. **V16** (7.65%)

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

## ☁️ Azure Deployment

### Azure Resources Created

| Resource | Region | Status |
|----------|--------|--------|
| **bigData** Resource Group | Switzerland North | ✅ Active |
| VM-Master | Switzerland North | ✅ Running |
| VM-Worker-1 | Switzerland North | ✅ Running |
| **fraud-detection-rg** | West Europe | ✅ Created |

### Azure for Students Setup

```bash
# Login to Azure
az login

# Create Resource Group
az group create --name fraud-detection-rg --location westeurope

# Deploy ARM Template
az deployment group create \
  --resource-group fraud-detection-rg \
  --template-file azure/arm-template.json
```

### Recommended Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Azure Event Hubs│────▶│ Azure Databricks │────▶│ Data Lake Gen2  │
│   (Streaming)   │     │   (Spark 3.5)    │     │   (Storage)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  Azure Monitor   │
                        │   + Grafana      │
                        └──────────────────┘
```

See [AZURE_FEDERATED_GUIDE.md](docs/AZURE_FEDERATED_GUIDE.md) for complete setup instructions.

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
| [rapport_projet.pdf](docs/rapport_projet.pdf) | Full project report (**21 pages**) |
| [presentation_beamer.pdf](docs/presentation_beamer.pdf) | Beamer presentation (**28 slides**) |
| [AZURE_FEDERATED_GUIDE.md](docs/AZURE_FEDERATED_GUIDE.md) | Azure + Federated Learning guide |
| [AZURE_STUDENTS_GUIDE.md](docs/AZURE_STUDENTS_GUIDE.md) | Azure for Students setup |
| [GRAFANA_SETUP.md](GRAFANA_SETUP.md) | Dashboard setup guide |

---

## 🛠 Technologies Used

- **Apache Spark 3.5.0** - Distributed processing
- **PySpark** - Python API
- **Spark SQL** - Data transformation
- **MLlib** - Machine Learning (RandomForest, LogisticRegression, GBT)
- **GraphX** - Network/Graph analysis
- **Structured Streaming** - Real-time processing
- **Azure Databricks** - Cloud deployment
- **Grafana** - Visualization dashboard
- **Docker** - Containerization
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

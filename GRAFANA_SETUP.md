================================================================================
GRAFANA SETUP GUIDE - Credit Card Fraud Detection Dashboard
================================================================================

This guide explains how to set up Grafana to visualize your fraud detection
pipeline results.

================================================================================
OPTION 1: DOCKER (RECOMMENDED - 5 minutes)
================================================================================

STEP 1: Start Grafana with Docker
----------------------------------
docker run -d -p 3000:3000 --name=grafana grafana/grafana-oss

STEP 2: Access Grafana
----------------------------------
Open: http://localhost:3000
Login: admin / admin (change password when prompted)

STEP 3: Add Data Source
----------------------------------
1. Go to Configuration → Data Sources
2. Click "Add data source"
3. Choose one of:
   
   A) For CSV files: Install "CSV" plugin
      - Settings → Plugins → Search "CSV"
      - Configure path to: grafana/data/

   B) For JSON files: Install "JSON API" plugin
      - Settings → Plugins → Search "JSON"

   C) For simple demo: Use "TestData" data source (built-in)

STEP 4: Import Dashboard
----------------------------------
1. Go to Dashboards → Import
2. Click "Upload JSON file"
3. Select: grafana/fraud_detection_dashboard.json
4. Click Import

================================================================================
OPTION 2: GRAFANA CLOUD (FREE TIER)
================================================================================

1. Sign up at: https://grafana.com/products/cloud/
2. Create a free instance
3. Upload the dashboard JSON
4. Use Grafana Cloud's file upload or API

================================================================================
OPTION 3: LOCAL INSTALL (Windows)
================================================================================

STEP 1: Download Grafana
----------------------------------
https://grafana.com/grafana/download?platform=windows

STEP 2: Install and Start
----------------------------------
- Run the MSI installer
- Start service: services.msc → Grafana → Start

STEP 3: Access at http://localhost:3000

================================================================================
CONNECTING YOUR DATA
================================================================================

After running the Spark pipeline, your data will be in:

  outputs/metrics/
    ├── sql_metrics.json          # SQL analytics results
    ├── ml_metrics_randomforest.json   # Model metrics
    ├── hourly_stats.csv          # Time series data
    └── streaming/                # Real-time metrics

  grafana/data/
    ├── overview_metrics.json     # Dashboard KPIs
    ├── hourly_transactions.csv   # Time series
    ├── amount_distribution.csv   # Bar charts
    ├── feature_importance.csv    # Feature analysis
    ├── confusion_matrix.csv      # ML evaluation
    └── recent_alerts.csv         # Live alerts table

================================================================================
DASHBOARD PANELS EXPLAINED
================================================================================

ROW 1: OVERVIEW METRICS
-----------------------
• Total Transactions - Total count processed
• Fraud Alerts - Number of detected frauds
• Fraud Rate % - Percentage of fraud
• Avg Transaction Amount - Mean amount
• Model Accuracy - ML model accuracy
• AUC-ROC Score - Model performance metric

ROW 2: REAL-TIME STREAMING
--------------------------
• Transactions Per Minute - Live volume graph
• Fraud Detections Over Time - Fraud trend line

ROW 3: ML MODEL PERFORMANCE
---------------------------
• Precision Gauge - Positive predictive value
• Recall Gauge - True positive rate
• F1 Score Gauge - Harmonic mean
• Confusion Matrix Table
• Transaction Distribution Pie

ROW 4: TRANSACTION ANALYSIS
---------------------------
• Amount Distribution by Bucket
• Transactions by Hour
• Fraud Rate by Amount Bucket

ROW 5: FEATURE IMPORTANCE
-------------------------
• Top 10 Features Bar Chart
• Recent Fraud Alerts Table

================================================================================
SCREENSHOTS FOR REPORT
================================================================================

Capture these screenshots for your report/presentation:

1. Full dashboard overview (zoom out)
2. KPI metrics row close-up
3. Time series graphs
4. ML performance gauges
5. Confusion matrix table
6. Feature importance chart
7. Live alerts table

================================================================================
QUICK DEMO WITHOUT FULL SETUP
================================================================================

If you don't have time for full Grafana setup:

1. Run: python src/prepare_grafana_data.py
2. This generates all visualization-ready CSV/JSON files
3. You can show these in the report with:
   - Python matplotlib plots
   - Excel charts
   - Screenshots of the JSON data

The professor will see that the data is Grafana-compatible!

================================================================================

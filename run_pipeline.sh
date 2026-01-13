#!/bin/bash
# ============================================================================
# BIG DATA FRAUD DETECTION - COMPLETE PIPELINE RUNNER
# ============================================================================

echo ""
echo "============================================================================"
echo "   CREDIT CARD FRAUD DETECTION - BIG DATA PIPELINE"
echo "============================================================================"
echo ""

# Check if data exists
if [ ! -f "data/raw/creditcard.csv" ]; then
    echo "[WARNING] Dataset not found!"
    echo "[INFO] Generating sample data for testing..."
    python3 src/generate_sample_data.py
    echo ""
fi

echo "[STEP 1/4] Running Spark SQL Analytics..."
echo "----------------------------------------------------------------------------"
spark-submit src/spark_sql_analytics.py || { echo "[ERROR] Spark SQL failed"; exit 1; }
echo ""

echo "[STEP 2/4] Training MLlib Models..."
echo "----------------------------------------------------------------------------"
spark-submit src/mllib_fraud_model.py || { echo "[ERROR] MLlib failed"; exit 1; }
echo ""

echo "[STEP 3/4] Preparing Grafana Data..."
echo "----------------------------------------------------------------------------"
python3 src/prepare_grafana_data.py
echo ""

echo "[STEP 4/4] Streaming Simulation (60 seconds)..."
echo "----------------------------------------------------------------------------"
read -p "Press Enter to start streaming demo, or Ctrl+C to skip..."
spark-submit src/streaming_fraud_detection.py

echo ""
echo "============================================================================"
echo "   PIPELINE COMPLETED SUCCESSFULLY!"
echo "============================================================================"
echo ""
echo "Outputs:"
echo "  - Metrics: outputs/metrics/"
echo "  - Predictions: outputs/predictions/"
echo "  - Grafana Data: grafana/data/"
echo ""
echo "Next steps:"
echo "  1. Start Grafana: docker run -d -p 3000:3000 grafana/grafana-oss"
echo "  2. Open http://localhost:3000"
echo "  3. Import dashboard from: grafana/fraud_detection_dashboard.json"
echo ""
echo "============================================================================"

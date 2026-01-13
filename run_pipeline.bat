@echo off
REM ============================================================================
REM BIG DATA FRAUD DETECTION - COMPLETE PIPELINE RUNNER
REM ============================================================================
REM Run this script to execute the entire pipeline
REM ============================================================================

echo.
echo ============================================================================
echo    CREDIT CARD FRAUD DETECTION - BIG DATA PIPELINE
echo ============================================================================
echo.

REM Check if data exists
if not exist "data\raw\creditcard.csv" (
    echo [WARNING] Dataset not found!
    echo [INFO] Generating sample data for testing...
    python src\generate_sample_data.py
    echo.
)

echo [STEP 1/4] Running Spark SQL Analytics...
echo ----------------------------------------------------------------------------
spark-submit src\spark_sql_analytics.py
if %errorlevel% neq 0 (
    echo [ERROR] Spark SQL failed. Check your Spark installation.
    pause
    exit /b 1
)
echo.

echo [STEP 2/4] Training MLlib Models...
echo ----------------------------------------------------------------------------
spark-submit src\mllib_fraud_model.py
if %errorlevel% neq 0 (
    echo [ERROR] MLlib training failed.
    pause
    exit /b 1
)
echo.

echo [STEP 3/4] Preparing Grafana Data...
echo ----------------------------------------------------------------------------
python src\prepare_grafana_data.py
echo.

echo [STEP 4/4] Streaming Simulation (Optional - 60 seconds)
echo ----------------------------------------------------------------------------
echo Press any key to start streaming demo, or Ctrl+C to skip...
pause >nul
spark-submit src\streaming_fraud_detection.py

echo.
echo ============================================================================
echo    PIPELINE COMPLETED SUCCESSFULLY!
echo ============================================================================
echo.
echo Outputs:
echo   - Metrics: outputs\metrics\
echo   - Predictions: outputs\predictions\
echo   - Grafana Data: grafana\data\
echo.
echo Next steps:
echo   1. Start Grafana: docker run -d -p 3000:3000 grafana/grafana-oss
echo   2. Open http://localhost:3000
echo   3. Import dashboard from: grafana\fraud_detection_dashboard.json
echo.
echo ============================================================================
pause

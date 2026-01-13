"""
================================================================================
SPARK STREAMING - Real-Time Fraud Detection Simulation
================================================================================
Big Data Final Project - Module Clôture
Technologies: Spark Structured Streaming, File-based Streaming

This script simulates:
1. Real-time transaction arrival via folder monitoring
2. Live ML predictions on new transactions
3. Alert generation for detected fraud
4. Metrics export for Grafana dashboard
================================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, lit, when, count, sum as spark_sum,
    window, avg, max as spark_max, min as spark_min
)
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, TimestampType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml import PipelineModel
import os
import json
import time
import shutil
from datetime import datetime
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAMING_INPUT = os.path.join(BASE_PATH, "data", "streaming_input")
STREAMING_OUTPUT = os.path.join(BASE_PATH, "outputs", "streaming")
METRICS_OUTPUT = os.path.join(BASE_PATH, "outputs", "metrics")
DATA_RAW = os.path.join(BASE_PATH, "data", "raw", "creditcard.csv")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

TRANSACTION_SCHEMA = StructType([
    StructField("Time", DoubleType(), True),
] + [
    StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)
] + [
    StructField("Amount", DoubleType(), True),
    StructField("Class", IntegerType(), True),
])

# ============================================================================
# SPARK SESSION
# ============================================================================

def create_spark_session():
    """Initialize Spark Session for Streaming"""
    spark = SparkSession.builder \
        .appName("FraudDetection-Streaming") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.streaming.checkpointLocation", 
                os.path.join(BASE_PATH, "checkpoints")) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("=" * 70)
    print("SPARK STREAMING SESSION INITIALIZED")
    print(f"  App: {spark.sparkContext.appName}")
    print(f"  Version: {spark.version}")
    print("=" * 70)
    return spark

# ============================================================================
# DATA SIMULATION - Generates "live" transactions
# ============================================================================

class TransactionSimulator:
    """Simulates real-time transaction arrival by writing CSV batches"""
    
    def __init__(self, source_file, output_dir, batch_size=100, interval=5):
        self.source_file = source_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.interval = interval
        self.running = False
        self.batch_count = 0
        self.total_sent = 0
        self.fraud_sent = 0
        
    def start(self):
        """Start the simulation in a background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        print(f"\n[SIMULATOR] Started - Batch size: {self.batch_size}, Interval: {self.interval}s")
        
    def stop(self):
        """Stop the simulation"""
        self.running = False
        print(f"\n[SIMULATOR] Stopped - Total batches: {self.batch_count}, Total transactions: {self.total_sent}")
        
    def _run_simulation(self):
        """Internal simulation loop"""
        import csv
        import random
        
        # Read all transactions
        with open(self.source_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            all_rows = list(reader)
        
        # Ensure output directory exists and is empty
        os.makedirs(self.output_dir, exist_ok=True)
        for f in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, f))
        
        while self.running and all_rows:
            # Take a batch
            batch = []
            for _ in range(min(self.batch_size, len(all_rows))):
                if all_rows:
                    row = all_rows.pop(random.randint(0, len(all_rows)-1))
                    batch.append(row)
                    if row[-1] == '1':  # Fraud
                        self.fraud_sent += 1
            
            # Write batch to new CSV file
            self.batch_count += 1
            self.total_sent += len(batch)
            batch_file = os.path.join(
                self.output_dir, 
                f"batch_{self.batch_count:04d}_{datetime.now().strftime('%H%M%S')}.csv"
            )
            
            with open(batch_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(batch)
            
            print(f"  [BATCH {self.batch_count}] Sent {len(batch)} transactions (Total: {self.total_sent}, Frauds: {self.fraud_sent})")
            
            time.sleep(self.interval)
        
        self.running = False

# ============================================================================
# STREAMING PIPELINE
# ============================================================================

def create_streaming_pipeline(spark):
    """Create the Spark Structured Streaming pipeline"""
    print("\n[STEP 1] Creating streaming pipeline...")
    
    # Read stream from CSV files
    stream_df = spark.readStream \
        .schema(TRANSACTION_SCHEMA) \
        .option("header", "true") \
        .option("maxFilesPerTrigger", 1) \
        .csv(STREAMING_INPUT)
    
    print(f"  ✓ Streaming source: {STREAMING_INPUT}")
    
    # Add processing timestamp
    stream_df = stream_df.withColumn("processing_time", current_timestamp())
    
    return stream_df

def prepare_features_streaming(df):
    """Prepare features for ML prediction (stateless)"""
    # Simple vector assembly without fitting (for streaming)
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features",
        handleInvalid="skip"
    )
    return assembler.transform(df)

def apply_rule_based_detection(df):
    """
    Apply rule-based fraud detection for demonstration
    (Since loading a pre-trained model requires model persistence setup)
    """
    # Rule-based scoring (mimics ML model behavior)
    # These rules are based on typical fraud patterns
    df_scored = df.withColumn(
        "fraud_score",
        # V1 < -3 is often fraud
        when(col("V1") < -3, 0.7).otherwise(0.0) +
        # V3 < -4 is suspicious
        when(col("V3") < -4, 0.15).otherwise(0.0) +
        # V4 > 3 is suspicious
        when(col("V4") > 3, 0.1).otherwise(0.0) +
        # High amount is suspicious
        when(col("Amount") > 1000, 0.15).otherwise(0.0) +
        # V7 < -5 is suspicious
        when(col("V7") < -5, 0.1).otherwise(0.0) +
        # V10 < -5 is suspicious
        when(col("V10") < -5, 0.1).otherwise(0.0)
    ).withColumn(
        "prediction",
        when(col("fraud_score") > 0.5, 1).otherwise(0)
    ).withColumn(
        "is_alert",
        when(col("fraud_score") > 0.5, lit("FRAUD_ALERT")).otherwise(lit("NORMAL"))
    )
    
    return df_scored

# ============================================================================
# OUTPUT SINKS
# ============================================================================

def write_to_console(df, output_mode="append"):
    """Write stream output to console (for debugging)"""
    return df.writeStream \
        .outputMode(output_mode) \
        .format("console") \
        .option("truncate", False) \
        .start()

def write_to_parquet(df, output_path):
    """Write stream output to parquet files"""
    os.makedirs(output_path, exist_ok=True)
    checkpoint_path = os.path.join(BASE_PATH, "checkpoints", "parquet")
    
    return df.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", output_path) \
        .option("checkpointLocation", checkpoint_path) \
        .start()

def write_alerts_to_csv(df, output_path):
    """Write fraud alerts to CSV for Grafana"""
    os.makedirs(output_path, exist_ok=True)
    checkpoint_path = os.path.join(BASE_PATH, "checkpoints", "alerts")
    
    # Filter only fraud alerts
    alerts = df.filter(col("prediction") == 1)
    
    return alerts.writeStream \
        .outputMode("append") \
        .format("csv") \
        .option("path", output_path) \
        .option("checkpointLocation", checkpoint_path) \
        .option("header", True) \
        .start()

def write_metrics_json(df, metrics_path):
    """
    Write aggregated metrics to JSON using foreachBatch
    This creates files Grafana can read
    """
    os.makedirs(metrics_path, exist_ok=True)
    
    def process_batch(batch_df, batch_id):
        if batch_df.count() > 0:
            # Calculate batch metrics
            metrics = batch_df.agg(
                count("*").alias("total_transactions"),
                spark_sum(when(col("prediction") == 1, 1).otherwise(0)).alias("fraud_detected"),
                spark_sum(when(col("Class") == 1, 1).otherwise(0)).alias("actual_fraud"),
                avg("Amount").alias("avg_amount"),
                spark_max("fraud_score").alias("max_fraud_score")
            ).collect()[0]
            
            # Create metrics dict
            metrics_dict = {
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "total_transactions": metrics["total_transactions"],
                "fraud_detected": int(metrics["fraud_detected"]) if metrics["fraud_detected"] else 0,
                "actual_fraud": int(metrics["actual_fraud"]) if metrics["actual_fraud"] else 0,
                "avg_amount": round(metrics["avg_amount"], 2) if metrics["avg_amount"] else 0,
                "max_fraud_score": round(metrics["max_fraud_score"], 4) if metrics["max_fraud_score"] else 0,
                "fraud_rate": round(
                    (metrics["fraud_detected"] / metrics["total_transactions"] * 100), 4
                ) if metrics["total_transactions"] > 0 else 0
            }
            
            # Write to JSON file
            json_file = os.path.join(metrics_path, f"streaming_metrics_{batch_id:04d}.json")
            with open(json_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            # Also append to cumulative CSV for Grafana time series
            csv_file = os.path.join(metrics_path, "streaming_metrics_timeseries.csv")
            write_header = not os.path.exists(csv_file)
            with open(csv_file, 'a') as f:
                if write_header:
                    f.write("timestamp,batch_id,total_transactions,fraud_detected,fraud_rate,avg_amount\n")
                f.write(f"{metrics_dict['timestamp']},{batch_id},{metrics_dict['total_transactions']},"
                       f"{metrics_dict['fraud_detected']},{metrics_dict['fraud_rate']},{metrics_dict['avg_amount']}\n")
            
            print(f"\n  [METRICS] Batch {batch_id}: {metrics_dict['total_transactions']} txns, "
                  f"{metrics_dict['fraud_detected']} fraud ({metrics_dict['fraud_rate']:.2f}%)")
    
    checkpoint_path = os.path.join(BASE_PATH, "checkpoints", "metrics")
    
    return df.writeStream \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", checkpoint_path) \
        .start()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main Streaming Pipeline"""
    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD DETECTION - STREAMING PIPELINE")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Clean up previous runs
    for folder in ["checkpoints", "outputs/streaming"]:
        path = os.path.join(BASE_PATH, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
    
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Start transaction simulator
        print("\n[STEP 1] Starting transaction simulator...")
        simulator = TransactionSimulator(
            source_file=DATA_RAW,
            output_dir=STREAMING_INPUT,
            batch_size=50,  # 50 transactions per batch
            interval=3      # Every 3 seconds
        )
        simulator.start()
        
        # Wait for first batch
        print("  Waiting for first batch...")
        time.sleep(5)
        
        # Create streaming pipeline
        stream_df = create_streaming_pipeline(spark)
        
        # Apply fraud detection
        print("\n[STEP 2] Applying fraud detection rules...")
        scored_df = apply_rule_based_detection(stream_df)
        
        # Select output columns
        output_df = scored_df.select(
            "Time", "Amount", "Class", "fraud_score", 
            "prediction", "is_alert", "processing_time"
        )
        
        # Start output streams
        print("\n[STEP 3] Starting output streams...")
        
        # Stream 1: Metrics to JSON/CSV for Grafana
        metrics_query = write_metrics_json(scored_df, os.path.join(METRICS_OUTPUT, "streaming"))
        print("  ✓ Metrics stream started")
        
        # Stream 2: All predictions to parquet
        parquet_query = write_to_parquet(output_df, os.path.join(STREAMING_OUTPUT, "predictions"))
        print("  ✓ Parquet stream started")
        
        # Stream 3: Alerts to CSV
        alerts_query = write_alerts_to_csv(output_df, os.path.join(STREAMING_OUTPUT, "alerts"))
        print("  ✓ Alerts stream started")
        
        print("\n" + "=" * 70)
        print("STREAMING ACTIVE - Press Ctrl+C to stop")
        print("=" * 70)
        print("\nMonitoring transactions...")
        
        # Run for 60 seconds (demo mode)
        start_time = time.time()
        duration = 60  # Run for 60 seconds
        
        while time.time() - start_time < duration:
            time.sleep(1)
            remaining = int(duration - (time.time() - start_time))
            if remaining % 10 == 0 and remaining > 0:
                print(f"  ... {remaining}s remaining")
        
        # Stop
        print("\n[STOPPING] Graceful shutdown...")
        simulator.stop()
        
        metrics_query.stop()
        parquet_query.stop()
        alerts_query.stop()
        
        # Print final summary
        print("\n" + "=" * 70)
        print("STREAMING PIPELINE COMPLETED")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  • Total batches processed: {simulator.batch_count}")
        print(f"  • Total transactions: {simulator.total_sent}")
        print(f"  • Fraud transactions sent: {simulator.fraud_sent}")
        print(f"\nOutputs:")
        print(f"  • Metrics: {os.path.join(METRICS_OUTPUT, 'streaming')}")
        print(f"  • Predictions: {os.path.join(STREAMING_OUTPUT, 'predictions')}")
        print(f"  • Alerts: {os.path.join(STREAMING_OUTPUT, 'alerts')}")
        print("\n" + "=" * 70)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping streams...")
        simulator.stop()
        
    finally:
        spark.stop()
        print("\nSpark session stopped.")

if __name__ == "__main__":
    main()

"""
================================================================================
SPARK SQL - Data Cleaning and Analytics for Credit Card Fraud Detection
================================================================================
Big Data Final Project - Module Clôture
Technologies: Apache Spark, Spark SQL, PySpark

This script performs:
1. Data loading from CSV
2. Data cleaning and validation
3. SQL analytics and KPIs
4. Export to Parquet for downstream processing
================================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum, avg, min, max, stddev,
    when, hour, round as spark_round, lit,
    percentile_approx, desc, asc
)
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
import os
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_PATH, "data", "raw", "creditcard.csv")
DATA_PROCESSED = os.path.join(BASE_PATH, "data", "processed")
METRICS_OUTPUT = os.path.join(BASE_PATH, "outputs", "metrics")

# ============================================================================
# SPARK SESSION INITIALIZATION
# ============================================================================

def create_spark_session():
    """Initialize Spark Session with optimized configuration"""
    spark = SparkSession.builder \
        .appName("FraudDetection-SparkSQL") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("=" * 70)
    print("SPARK SESSION INITIALIZED")
    print(f"  App Name: {spark.sparkContext.appName}")
    print(f"  Master: {spark.sparkContext.master}")
    print(f"  Spark Version: {spark.version}")
    print("=" * 70)
    return spark

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(spark):
    """Load credit card transaction data from CSV"""
    print("\n[STEP 1] Loading data from CSV...")
    
    # Define schema for better performance
    schema = StructType([
        StructField("Time", DoubleType(), True),
    ] + [
        StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)
    ] + [
        StructField("Amount", DoubleType(), True),
        StructField("Class", IntegerType(), True),
    ])
    
    df = spark.read.csv(DATA_RAW, header=True, schema=schema)
    
    record_count = df.count()
    print(f"  ✓ Loaded {record_count:,} transactions")
    print(f"  ✓ Columns: {len(df.columns)}")
    
    return df

# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_data(df):
    """Clean and validate transaction data"""
    print("\n[STEP 2] Cleaning data...")
    
    initial_count = df.count()
    
    # Remove null values
    df_clean = df.dropna()
    
    # Remove invalid amounts (negative or zero)
    df_clean = df_clean.filter(col("Amount") > 0)
    
    # Remove extreme outliers in Amount (> 99.9 percentile for non-fraud)
    amount_threshold = df_clean.filter(col("Class") == 0) \
        .approxQuantile("Amount", [0.999], 0.01)[0]
    
    # Keep fraud transactions regardless of amount
    df_clean = df_clean.filter(
        (col("Amount") <= amount_threshold) | (col("Class") == 1)
    )
    
    # Add derived features
    df_clean = df_clean.withColumn(
        "Hour", (col("Time") / 3600).cast("integer") % 24
    ).withColumn(
        "Amount_Log", when(col("Amount") > 0, 
                          spark_round(col("Amount"), 2)).otherwise(0)
    ).withColumn(
        "Is_High_Amount", when(col("Amount") > 500, 1).otherwise(0)
    )
    
    final_count = df_clean.count()
    removed = initial_count - final_count
    
    print(f"  ✓ Initial records: {initial_count:,}")
    print(f"  ✓ After cleaning: {final_count:,}")
    print(f"  ✓ Removed: {removed:,} ({removed/initial_count*100:.2f}%)")
    
    return df_clean

# ============================================================================
# SQL ANALYTICS
# ============================================================================

def sql_analytics(spark, df):
    """Perform SQL analytics and generate KPIs"""
    print("\n[STEP 3] Running SQL Analytics...")
    
    # Register as temp view for SQL queries
    df.createOrReplaceTempView("transactions")
    
    metrics = {}
    
    # ----- KPI 1: Overall Statistics -----
    print("\n  [KPI 1] Overall Statistics...")
    overall = spark.sql("""
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as fraud_count,
            SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) as normal_count,
            ROUND(AVG(Amount), 2) as avg_amount,
            ROUND(MIN(Amount), 2) as min_amount,
            ROUND(MAX(Amount), 2) as max_amount,
            ROUND(STDDEV(Amount), 2) as stddev_amount
        FROM transactions
    """).collect()[0]
    
    metrics['overall'] = {
        'total_transactions': overall['total_transactions'],
        'fraud_count': overall['fraud_count'],
        'normal_count': overall['normal_count'],
        'fraud_rate': round(overall['fraud_count'] / overall['total_transactions'] * 100, 4),
        'avg_amount': overall['avg_amount'],
        'min_amount': overall['min_amount'],
        'max_amount': overall['max_amount'],
        'stddev_amount': overall['stddev_amount']
    }
    print(f"    Total: {metrics['overall']['total_transactions']:,}")
    print(f"    Fraud Rate: {metrics['overall']['fraud_rate']:.4f}%")
    
    # ----- KPI 2: Fraud vs Normal Amount Comparison -----
    print("\n  [KPI 2] Fraud vs Normal Comparison...")
    comparison = spark.sql("""
        SELECT 
            Class,
            COUNT(*) as count,
            ROUND(AVG(Amount), 2) as avg_amount,
            ROUND(PERCENTILE(Amount, 0.5), 2) as median_amount,
            ROUND(MAX(Amount), 2) as max_amount
        FROM transactions
        GROUP BY Class
        ORDER BY Class
    """).collect()
    
    metrics['class_comparison'] = [
        {
            'class': row['Class'],
            'class_name': 'Fraud' if row['Class'] == 1 else 'Normal',
            'count': row['count'],
            'avg_amount': row['avg_amount'],
            'median_amount': row['median_amount'],
            'max_amount': row['max_amount']
        }
        for row in comparison
    ]
    
    # ----- KPI 3: Hourly Distribution -----
    print("\n  [KPI 3] Hourly Distribution...")
    hourly = spark.sql("""
        SELECT 
            Hour,
            COUNT(*) as total_count,
            SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as fraud_count,
            ROUND(SUM(Amount), 2) as total_amount,
            ROUND(AVG(Amount), 2) as avg_amount
        FROM transactions
        GROUP BY Hour
        ORDER BY Hour
    """).collect()
    
    metrics['hourly_distribution'] = [
        {
            'hour': row['Hour'],
            'total_count': row['total_count'],
            'fraud_count': row['fraud_count'],
            'total_amount': row['total_amount'],
            'avg_amount': row['avg_amount']
        }
        for row in hourly
    ]
    
    # ----- KPI 4: Amount Buckets -----
    print("\n  [KPI 4] Amount Distribution Buckets...")
    buckets = spark.sql("""
        SELECT 
            CASE 
                WHEN Amount < 10 THEN '0-10'
                WHEN Amount < 50 THEN '10-50'
                WHEN Amount < 100 THEN '50-100'
                WHEN Amount < 500 THEN '100-500'
                WHEN Amount < 1000 THEN '500-1000'
                ELSE '1000+'
            END as amount_bucket,
            COUNT(*) as count,
            SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as fraud_count,
            ROUND(SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) as fraud_rate
        FROM transactions
        GROUP BY 
            CASE 
                WHEN Amount < 10 THEN '0-10'
                WHEN Amount < 50 THEN '10-50'
                WHEN Amount < 100 THEN '50-100'
                WHEN Amount < 500 THEN '100-500'
                WHEN Amount < 1000 THEN '500-1000'
                ELSE '1000+'
            END
        ORDER BY 
            CASE amount_bucket
                WHEN '0-10' THEN 1
                WHEN '10-50' THEN 2
                WHEN '50-100' THEN 3
                WHEN '100-500' THEN 4
                WHEN '500-1000' THEN 5
                ELSE 6
            END
    """).collect()
    
    metrics['amount_buckets'] = [
        {
            'bucket': row['amount_bucket'],
            'count': row['count'],
            'fraud_count': row['fraud_count'],
            'fraud_rate': row['fraud_rate']
        }
        for row in buckets
    ]
    
    # ----- KPI 5: Top Features Correlation with Fraud -----
    print("\n  [KPI 5] Feature Analysis...")
    feature_stats = spark.sql("""
        SELECT 
            ROUND(AVG(CASE WHEN Class = 1 THEN V1 END), 4) as V1_fraud_avg,
            ROUND(AVG(CASE WHEN Class = 0 THEN V1 END), 4) as V1_normal_avg,
            ROUND(AVG(CASE WHEN Class = 1 THEN V3 END), 4) as V3_fraud_avg,
            ROUND(AVG(CASE WHEN Class = 0 THEN V3 END), 4) as V3_normal_avg,
            ROUND(AVG(CASE WHEN Class = 1 THEN V4 END), 4) as V4_fraud_avg,
            ROUND(AVG(CASE WHEN Class = 0 THEN V4 END), 4) as V4_normal_avg,
            ROUND(AVG(CASE WHEN Class = 1 THEN V7 END), 4) as V7_fraud_avg,
            ROUND(AVG(CASE WHEN Class = 0 THEN V7 END), 4) as V7_normal_avg
        FROM transactions
    """).collect()[0]
    
    metrics['feature_analysis'] = {
        'V1': {'fraud_avg': feature_stats['V1_fraud_avg'], 'normal_avg': feature_stats['V1_normal_avg']},
        'V3': {'fraud_avg': feature_stats['V3_fraud_avg'], 'normal_avg': feature_stats['V3_normal_avg']},
        'V4': {'fraud_avg': feature_stats['V4_fraud_avg'], 'normal_avg': feature_stats['V4_normal_avg']},
        'V7': {'fraud_avg': feature_stats['V7_fraud_avg'], 'normal_avg': feature_stats['V7_normal_avg']},
    }
    
    print("  ✓ SQL Analytics completed!")
    return metrics

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_data(spark, df, metrics):
    """Export processed data and metrics"""
    print("\n[STEP 4] Exporting data...")
    
    # Ensure directories exist
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    os.makedirs(METRICS_OUTPUT, exist_ok=True)
    
    # Export cleaned data to Parquet
    parquet_path = os.path.join(DATA_PROCESSED, "transactions_clean")
    df.write.mode("overwrite").parquet(parquet_path)
    print(f"  ✓ Parquet saved: {parquet_path}")
    
    # Helper function to convert Decimal to float for JSON serialization
    def convert_decimals(obj):
        from decimal import Decimal
        if isinstance(obj, dict):
            return {k: convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_decimals(item) for item in obj]
        elif isinstance(obj, Decimal):
            return float(obj)
        return obj
    
    # Export metrics to JSON (for Grafana)
    metrics_file = os.path.join(METRICS_OUTPUT, "sql_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(convert_decimals(metrics), f, indent=2)
    print(f"  ✓ Metrics JSON saved: {metrics_file}")
    
    # Export hourly data as CSV (for Grafana time series)
    hourly_df = spark.createDataFrame(metrics['hourly_distribution'])
    hourly_csv = os.path.join(METRICS_OUTPUT, "hourly_stats.csv")
    hourly_df.toPandas().to_csv(hourly_csv, index=False)
    print(f"  ✓ Hourly CSV saved: {hourly_csv}")
    
    # Export amount buckets as CSV
    buckets_df = spark.createDataFrame(metrics['amount_buckets'])
    buckets_csv = os.path.join(METRICS_OUTPUT, "amount_buckets.csv")
    buckets_df.toPandas().to_csv(buckets_csv, index=False)
    print(f"  ✓ Amount buckets CSV saved: {buckets_csv}")
    
    return parquet_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD DETECTION - SPARK SQL PIPELINE")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Load data
        df = load_data(spark)
        
        # Clean data
        df_clean = clean_data(df)
        
        # Run SQL Analytics
        metrics = sql_analytics(spark, df_clean)
        
        # Export results
        parquet_path = export_data(spark, df_clean, metrics)
        
        # Print summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nKey Metrics:")
        print(f"  • Total Transactions: {metrics['overall']['total_transactions']:,}")
        print(f"  • Fraud Cases: {metrics['overall']['fraud_count']:,}")
        print(f"  • Fraud Rate: {metrics['overall']['fraud_rate']:.4f}%")
        print(f"  • Average Amount: ${metrics['overall']['avg_amount']:.2f}")
        print(f"\nOutputs:")
        print(f"  • Cleaned Parquet: {parquet_path}")
        print(f"  • Metrics: {METRICS_OUTPUT}")
        print("\n" + "=" * 70)
        
        return df_clean, metrics
        
    finally:
        spark.stop()
        print("\nSpark session stopped.")

if __name__ == "__main__":
    main()

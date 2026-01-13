"""
Script to keep Spark UI alive for screenshots
Run this and access http://localhost:4040
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, sum as spark_sum
import time

# Create Spark session
spark = SparkSession.builder \
    .appName("FraudDetection-Demo") \
    .master("local[*]") \
    .config("spark.ui.port", "4040") \
    .getOrCreate()

print("=" * 60)
print("SPARK UI DISPONIBLE SUR: http://localhost:4040")
print("=" * 60)

# Load and process data to generate jobs for UI
print("\n[1] Loading data...")
df = spark.read.csv("/app/data/raw/creditcard.csv", header=True, inferSchema=True)
df.cache()
print(f"    Loaded {df.count()} transactions")

# Run some analytics to show jobs in UI
print("\n[2] Running analytics (check Jobs tab)...")

# Job 1: Basic statistics
stats = df.describe()
stats.show()

# Job 2: Fraud analysis
print("\n[3] Fraud distribution (check SQL tab)...")
df.createOrReplaceTempView("transactions")

fraud_stats = spark.sql("""
    SELECT 
        Class,
        COUNT(*) as count,
        ROUND(AVG(Amount), 2) as avg_amount,
        ROUND(MAX(Amount), 2) as max_amount
    FROM transactions
    GROUP BY Class
""")
fraud_stats.show()

# Job 3: Hourly analysis
print("\n[4] Hourly analysis...")
hourly = spark.sql("""
    SELECT 
        CAST(Time / 3600 AS INT) % 24 as hour,
        COUNT(*) as transactions,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as frauds
    FROM transactions
    GROUP BY CAST(Time / 3600 AS INT) % 24
    ORDER BY hour
""")
hourly.show(24)

print("\n" + "=" * 60)
print("SPARK UI READY FOR SCREENSHOTS!")
print("=" * 60)
print("""
SCREENSHOTS TO TAKE:

1. JOBS TAB (http://localhost:4040/jobs/)
   - Shows all completed Spark jobs
   - Good for showing distributed processing

2. STAGES TAB (http://localhost:4040/stages/)
   - Shows execution stages
   - DAG visualization

3. STORAGE TAB (http://localhost:4040/storage/)
   - Shows cached RDDs/DataFrames

4. SQL TAB (http://localhost:4040/SQL/)
   - Shows SQL queries executed
   - Query execution plans

5. EXECUTORS TAB (http://localhost:4040/executors/)
   - Shows executor metrics
   - Memory usage

Press Ctrl+C to stop when done with screenshots.
""")

# Keep session alive
print("\nKeeping Spark session alive... (Ctrl+C to stop)")
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\nStopping Spark session...")
    spark.stop()
    print("Done!")

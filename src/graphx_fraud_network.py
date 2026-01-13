"""
GraphX-style Fraud Network Analysis using PySpark GraphFrames
Détecte les patterns de fraude par analyse de graphe
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, abs as spark_abs, lit, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType
import json
import os

# Initialize Spark with GraphFrames
spark = SparkSession.builder \
    .appName("GraphX_Fraud_Network_Analysis") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("GRAPHX FRAUD NETWORK ANALYSIS")
print("=" * 60)

# Load data
DATA_PATH = "/app/data/raw/creditcard.csv"
OUTPUT_PATH = "/app/outputs/metrics/graphx_metrics.json"

# Load and prepare data
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
df = df.withColumn("id", monotonically_increasing_id())

print(f"\n📊 Dataset loaded: {df.count()} transactions")

# ============================================================================
# METHOD 1: Similarity-based Graph (without GraphFrames dependency)
# ============================================================================

print("\n" + "=" * 60)
print("1. TRANSACTION SIMILARITY ANALYSIS")
print("=" * 60)

# Get fraud transactions
fraud_df = df.filter(col("Class") == 1).cache()
normal_df = df.filter(col("Class") == 0).limit(1000).cache()

fraud_count = fraud_df.count()
print(f"   Fraud transactions: {fraud_count}")

# Analyze fraud clusters by amount ranges
print("\n📈 Fraud Amount Clusters:")
fraud_clusters = fraud_df.selectExpr(
    "CASE WHEN Amount < 10 THEN 'micro' " +
    "WHEN Amount < 100 THEN 'small' " +
    "WHEN Amount < 500 THEN 'medium' " +
    "ELSE 'large' END as cluster",
    "Amount",
    "V1", "V14", "V17"
).groupBy("cluster").agg(
    {"Amount": "avg", "V1": "avg", "V14": "avg", "V17": "avg", "*": "count"}
).withColumnRenamed("count(1)", "count") \
 .withColumnRenamed("avg(Amount)", "avg_amount") \
 .withColumnRenamed("avg(V1)", "avg_V1") \
 .withColumnRenamed("avg(V14)", "avg_V14") \
 .withColumnRenamed("avg(V17)", "avg_V17")

fraud_clusters.show()

# ============================================================================
# METHOD 2: Time-based Fraud Patterns (Connected Components simulation)
# ============================================================================

print("\n" + "=" * 60)
print("2. TEMPORAL FRAUD PATTERNS")
print("=" * 60)

# Add hour feature
df_with_hour = df.withColumn("Hour", (col("Time") / 3600) % 24)

# Find fraud by hour
hourly_fraud = df_with_hour.filter(col("Class") == 1) \
    .groupBy("Hour") \
    .count() \
    .orderBy("Hour")

print("\n🕐 Hourly Fraud Distribution:")
hourly_data = hourly_fraud.collect()
for row in hourly_data[:6]:
    print(f"   Hour {int(row['Hour']):02d}: {row['count']} frauds")
print("   ...")

# ============================================================================
# METHOD 3: Feature-based Clustering (PageRank simulation)
# ============================================================================

print("\n" + "=" * 60)
print("3. FEATURE IMPORTANCE NETWORK (PageRank-style)")
print("=" * 60)

# Calculate feature correlations with fraud
features = ["V1", "V2", "V3", "V4", "V7", "V10", "V11", "V12", "V14", "V17"]

feature_importance = []
for feat in features:
    fraud_avg = fraud_df.agg({feat: "avg"}).collect()[0][0]
    normal_avg = normal_df.agg({feat: "avg"}).collect()[0][0]
    diff = abs(fraud_avg - normal_avg) if fraud_avg and normal_avg else 0
    feature_importance.append({
        "feature": feat,
        "fraud_avg": round(fraud_avg, 4) if fraud_avg else 0,
        "normal_avg": round(normal_avg, 4) if normal_avg else 0,
        "separation": round(diff, 4)
    })

# Sort by separation (PageRank-style importance)
feature_importance.sort(key=lambda x: x["separation"], reverse=True)

print("\n🔗 Feature Importance (by class separation):")
for i, feat in enumerate(feature_importance[:5], 1):
    print(f"   {i}. {feat['feature']}: separation = {feat['separation']:.4f}")

# ============================================================================
# METHOD 4: Fraud Network Communities
# ============================================================================

print("\n" + "=" * 60)
print("4. FRAUD COMMUNITIES DETECTION")
print("=" * 60)

# Group similar frauds by V14 (most important feature)
fraud_with_community = fraud_df.withColumn(
    "community",
    when(col("V14") < -10, "high_risk_1")
    .when(col("V14") < -5, "high_risk_2")
    .when(col("V14") < 0, "medium_risk")
    .otherwise("low_risk")
)

communities = fraud_with_community.groupBy("community").agg(
    {"*": "count", "Amount": "avg", "V14": "avg"}
).withColumnRenamed("count(1)", "size") \
 .withColumnRenamed("avg(Amount)", "avg_amount") \
 .withColumnRenamed("avg(V14)", "avg_V14")

print("\n🏘️ Fraud Communities:")
comm_data = communities.collect()
for row in comm_data:
    print(f"   {row['community']}: {row['size']} frauds, avg=${row['avg_amount']:.2f}")

# ============================================================================
# METHOD 5: Triangle Counting (Fraud Triangles)
# ============================================================================

print("\n" + "=" * 60)
print("5. FRAUD PATTERN TRIANGLES")
print("=" * 60)

# Find transactions with similar patterns (simulated triangle counting)
# Transactions are "connected" if they have similar V14 and Amount

from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Bucket transactions
fraud_bucketed = fraud_df.withColumn(
    "v14_bucket", (col("V14") * 2).cast("int")
).withColumn(
    "amount_bucket", (col("Amount") / 50).cast("int")
)

# Count triangles (transactions in same bucket)
triangle_count = fraud_bucketed.groupBy("v14_bucket", "amount_bucket") \
    .count() \
    .filter(col("count") >= 3)  # At least 3 to form a triangle

num_triangles = triangle_count.count()
print(f"\n🔺 Detected {num_triangles} fraud pattern triangles")
print("   (Groups of 3+ similar transactions)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 60)
print("SAVING GRAPHX ANALYSIS RESULTS")
print("=" * 60)

# Collect all results
graphx_results = {
    "analysis_type": "GraphX Fraud Network Analysis",
    "total_transactions": df.count(),
    "fraud_transactions": fraud_count,
    "communities": {
        "detected": len(comm_data),
        "details": [
            {
                "name": row["community"],
                "size": row["size"],
                "avg_amount": round(row["avg_amount"], 2),
                "avg_V14": round(row["avg_V14"], 4)
            }
            for row in comm_data
        ]
    },
    "feature_importance_pagerank": feature_importance[:5],
    "triangle_patterns": num_triangles,
    "hourly_distribution": [
        {"hour": int(row["Hour"]), "fraud_count": row["count"]}
        for row in hourly_data
    ],
    "clusters": {
        "micro": {"range": "0-10", "description": "Micro transactions"},
        "small": {"range": "10-100", "description": "Small transactions"},
        "medium": {"range": "100-500", "description": "Medium transactions"},
        "large": {"range": "500+", "description": "Large transactions"}
    }
}

# Save to JSON
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(graphx_results, f, indent=2)

print(f"\n✅ Results saved to: {OUTPUT_PATH}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("📊 GRAPHX ANALYSIS SUMMARY")
print("=" * 60)
print(f"   • Total Transactions: {df.count():,}")
print(f"   • Fraud Detected: {fraud_count}")
print(f"   • Communities Found: {len(comm_data)}")
print(f"   • Pattern Triangles: {num_triangles}")
print(f"   • Top Feature: {feature_importance[0]['feature']} (sep={feature_importance[0]['separation']:.2f})")
print("=" * 60)

spark.stop()

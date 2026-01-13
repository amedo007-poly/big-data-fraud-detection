"""
================================================================================
MLlib - Machine Learning Fraud Detection Model
================================================================================
Big Data Final Project - Module Clôture
Technologies: Apache Spark MLlib, RandomForest, Logistic Regression

This script performs:
1. Feature engineering and preparation
2. Train/Test split with stratification
3. Model training (RandomForest + Logistic Regression)
4. Model evaluation (Precision, Recall, F1, AUC-ROC)
5. Feature importance analysis
6. Export predictions and metrics for Grafana
================================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, rand
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
import os
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_PATH, "data", "raw", "creditcard.csv")
DATA_PROCESSED = os.path.join(BASE_PATH, "data", "processed")
METRICS_OUTPUT = os.path.join(BASE_PATH, "outputs", "metrics")
PREDICTIONS_OUTPUT = os.path.join(BASE_PATH, "outputs", "predictions")

# Feature columns (V1-V28 + Amount)
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL_COL = "Class"

# ============================================================================
# SPARK SESSION
# ============================================================================

def create_spark_session():
    """Initialize Spark Session for ML"""
    spark = SparkSession.builder \
        .appName("FraudDetection-MLlib") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("=" * 70)
    print("SPARK MLlib SESSION INITIALIZED")
    print(f"  App: {spark.sparkContext.appName}")
    print(f"  Version: {spark.version}")
    print("=" * 70)
    return spark

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare_data(spark):
    """Load data and prepare features for ML"""
    print("\n[STEP 1] Loading and preparing data...")
    
    # Load from CSV
    df = spark.read.csv(DATA_RAW, header=True, inferSchema=True)
    print(f"  ✓ Loaded {df.count():,} records")
    
    # Remove nulls
    df = df.dropna()
    
    # Show class distribution
    class_dist = df.groupBy("Class").count().collect()
    for row in class_dist:
        print(f"  Class {row['Class']}: {row['count']:,} records")
    
    return df

def create_balanced_dataset(df, ratio=1.0):
    """
    Create balanced dataset by undersampling majority class
    This improves model performance on imbalanced data
    """
    print("\n[STEP 2] Balancing dataset...")
    
    # Separate classes
    fraud = df.filter(col("Class") == 1)
    normal = df.filter(col("Class") == 0)
    
    fraud_count = fraud.count()
    normal_count = normal.count()
    
    print(f"  Original - Fraud: {fraud_count:,}, Normal: {normal_count:,}")
    
    # Undersample normal transactions
    sample_fraction = (fraud_count * ratio) / normal_count
    if sample_fraction < 1.0:
        normal_sampled = normal.sample(False, sample_fraction, seed=42)
    else:
        normal_sampled = normal
    
    # Combine
    df_balanced = fraud.union(normal_sampled)
    
    # Shuffle
    df_balanced = df_balanced.orderBy(rand(seed=42))
    
    final_fraud = df_balanced.filter(col("Class") == 1).count()
    final_normal = df_balanced.filter(col("Class") == 0).count()
    
    print(f"  Balanced - Fraud: {final_fraud:,}, Normal: {final_normal:,}")
    print(f"  Total: {df_balanced.count():,}")
    
    return df_balanced

def prepare_features(df):
    """Assemble and scale features"""
    print("\n[STEP 3] Preparing ML features...")
    
    # Assemble features into vector
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features_raw"
    )
    
    # Scale features
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[assembler, scaler])
    
    # Fit and transform
    model = pipeline.fit(df)
    df_features = model.transform(df)
    
    print(f"  ✓ Features assembled: {len(FEATURE_COLS)} columns → 1 vector")
    print(f"  ✓ Features scaled with StandardScaler")
    
    return df_features, model

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_random_forest(train_data):
    """Train RandomForest Classifier"""
    print("\n[MODEL] Training RandomForest Classifier...")
    
    rf = RandomForestClassifier(
        labelCol=LABEL_COL,
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        minInstancesPerNode=5,
        seed=42,
        featureSubsetStrategy="sqrt"
    )
    
    model = rf.fit(train_data)
    print("  ✓ RandomForest trained with 100 trees")
    
    return model

def train_logistic_regression(train_data):
    """Train Logistic Regression Classifier"""
    print("\n[MODEL] Training Logistic Regression...")
    
    lr = LogisticRegression(
        labelCol=LABEL_COL,
        featuresCol="features",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.8
    )
    
    model = lr.fit(train_data)
    print("  ✓ Logistic Regression trained")
    
    return model

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(predictions, model_name):
    """Comprehensive model evaluation"""
    print(f"\n[EVAL] Evaluating {model_name}...")
    
    metrics = {}
    
    # Binary Classification Metrics
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction"
    )
    
    # AUC-ROC
    auc_roc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
    metrics['auc_roc'] = round(auc_roc, 4)
    
    # AUC-PR (important for imbalanced data)
    auc_pr = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})
    metrics['auc_pr'] = round(auc_pr, 4)
    
    # Multiclass Metrics
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction"
    )
    
    # Accuracy
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    metrics['accuracy'] = round(accuracy, 4)
    
    # Precision
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    metrics['precision'] = round(precision, 4)
    
    # Recall
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
    metrics['recall'] = round(recall, 4)
    
    # F1 Score
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    metrics['f1_score'] = round(f1, 4)
    
    # Confusion Matrix
    pred_and_labels = predictions.select("prediction", LABEL_COL).rdd \
        .map(lambda row: (float(row["prediction"]), float(row[LABEL_COL])))
    
    mllibMetrics = MulticlassMetrics(pred_and_labels)
    confusion = mllibMetrics.confusionMatrix().toArray()
    
    tn, fp, fn, tp = int(confusion[0][0]), int(confusion[0][1]), int(confusion[1][0]), int(confusion[1][1])
    
    metrics['confusion_matrix'] = {
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp
    }
    
    # Calculate specific fraud metrics
    if (tp + fp) > 0:
        fraud_precision = tp / (tp + fp)
    else:
        fraud_precision = 0
    
    if (tp + fn) > 0:
        fraud_recall = tp / (tp + fn)
    else:
        fraud_recall = 0
    
    metrics['fraud_precision'] = round(fraud_precision, 4)
    metrics['fraud_recall'] = round(fraud_recall, 4)
    
    # Print results
    print(f"\n  {'='*50}")
    print(f"  {model_name} EVALUATION RESULTS")
    print(f"  {'='*50}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:            {metrics['auc_pr']:.4f}")
    print(f"  {'='*50}")
    print(f"  FRAUD DETECTION SPECIFIC:")
    print(f"  Fraud Precision:   {metrics['fraud_precision']:.4f}")
    print(f"  Fraud Recall:      {metrics['fraud_recall']:.4f}")
    print(f"  {'='*50}")
    print(f"  CONFUSION MATRIX:")
    print(f"           Predicted")
    print(f"           Normal  Fraud")
    print(f"  Actual")
    print(f"  Normal   {tn:6d}  {fp:5d}")
    print(f"  Fraud    {fn:6d}  {tp:5d}")
    print(f"  {'='*50}")
    
    return metrics

def get_feature_importance(rf_model, feature_names):
    """Extract feature importance from RandomForest"""
    print("\n[ANALYSIS] Feature Importance...")
    
    importances = rf_model.featureImportances.toArray()
    
    # Pair features with importance
    feature_imp = list(zip(feature_names, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    
    print("\n  Top 10 Most Important Features:")
    print("  " + "-" * 35)
    for i, (name, imp) in enumerate(feature_imp[:10], 1):
        print(f"  {i:2d}. {name:8s}: {imp:.4f}")
    
    return dict(feature_imp)

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results(predictions, metrics, feature_importance, model_name):
    """Export predictions and metrics for Grafana"""
    print(f"\n[EXPORT] Saving {model_name} results...")
    
    os.makedirs(METRICS_OUTPUT, exist_ok=True)
    os.makedirs(PREDICTIONS_OUTPUT, exist_ok=True)
    
    # Save metrics JSON
    metrics_file = os.path.join(METRICS_OUTPUT, f"ml_metrics_{model_name.lower().replace(' ', '_')}.json")
    export_data = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'feature_importance': feature_importance
    }
    with open(metrics_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_file}")
    
    # Save predictions sample as CSV
    predictions_sample = predictions.select(
        "Time", "Amount", LABEL_COL, "prediction", "probability"
    ).limit(10000)
    
    # Convert probability to fraud probability using UDF
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType
    from pyspark.ml.linalg import DenseVector, SparseVector
    
    def extract_prob(v):
        try:
            if isinstance(v, DenseVector):
                return float(v[1])
            elif isinstance(v, SparseVector):
                return float(v.toArray()[1])
            else:
                return 0.0
        except:
            return 0.0
    
    extract_prob_udf = udf(extract_prob, DoubleType())
    
    predictions_df = predictions_sample.withColumn(
        "fraud_probability",
        extract_prob_udf(col("probability"))
    ).drop("probability")
    
    predictions_csv = os.path.join(PREDICTIONS_OUTPUT, f"predictions_{model_name.lower().replace(' ', '_')}.csv")
    predictions_df.toPandas().to_csv(predictions_csv, index=False)
    print(f"  ✓ Predictions saved: {predictions_csv}")
    
    # Save predictions as parquet
    parquet_path = os.path.join(PREDICTIONS_OUTPUT, f"predictions_{model_name.lower().replace(' ', '_')}_parquet")
    predictions_df.write.mode("overwrite").parquet(parquet_path)
    print(f"  ✓ Parquet saved: {parquet_path}")
    
    return metrics_file, predictions_csv

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main ML Pipeline"""
    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD DETECTION - MLlib PIPELINE")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Load data
        df = load_and_prepare_data(spark)
        
        # Balance dataset
        df_balanced = create_balanced_dataset(df, ratio=2.0)
        
        # Prepare features
        df_features, feature_pipeline = prepare_features(df_balanced)
        
        # Split data
        print("\n[STEP 4] Splitting data (80% train, 20% test)...")
        train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)
        print(f"  Train: {train_data.count():,} records")
        print(f"  Test: {test_data.count():,} records")
        
        # ========== RANDOM FOREST ==========
        rf_model = train_random_forest(train_data)
        rf_predictions = rf_model.transform(test_data)
        rf_metrics = evaluate_model(rf_predictions, "RandomForest")
        rf_importance = get_feature_importance(rf_model, FEATURE_COLS)
        export_results(rf_predictions, rf_metrics, rf_importance, "RandomForest")
        
        # ========== LOGISTIC REGRESSION ==========
        lr_model = train_logistic_regression(train_data)
        lr_predictions = lr_model.transform(test_data)
        lr_metrics = evaluate_model(lr_predictions, "LogisticRegression")
        lr_importance = dict(zip(FEATURE_COLS, [abs(c) for c in lr_model.coefficients.toArray()]))
        export_results(lr_predictions, lr_metrics, lr_importance, "LogisticRegression")
        
        # Model comparison
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\n{'Metric':<20} {'RandomForest':<15} {'LogisticReg':<15}")
        print("-" * 50)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'fraud_recall']:
            rf_val = rf_metrics.get(metric, 0)
            lr_val = lr_metrics.get(metric, 0)
            winner = "←" if rf_val > lr_val else "→" if lr_val > rf_val else "="
            print(f"{metric:<20} {rf_val:<15.4f} {lr_val:<15.4f} {winner}")
        
        # Final summary
        best_model = "RandomForest" if rf_metrics['auc_roc'] > lr_metrics['auc_roc'] else "LogisticRegression"
        print(f"\n✓ Best Model: {best_model}")
        print(f"✓ Fraud Recall (most important): RF={rf_metrics['fraud_recall']:.4f}, LR={lr_metrics['fraud_recall']:.4f}")
        print("\n" + "=" * 70)
        print("MLlib PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return rf_model, lr_model, rf_metrics, lr_metrics
        
    finally:
        spark.stop()
        print("\nSpark session stopped.")

if __name__ == "__main__":
    main()

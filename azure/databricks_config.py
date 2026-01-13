# Azure Databricks Fraud Detection - Cluster Configuration
# This file configures the Spark cluster for production deployment

cluster_config = {
    "cluster_name": "fraud-detection-cluster",
    "spark_version": "13.3.x-scala2.12",
    "node_type_id": "Standard_DS3_v2",
    "num_workers": 2,
    "autoscale": {
        "min_workers": 1,
        "max_workers": 4
    },
    "spark_conf": {
        "spark.sql.shuffle.partitions": "200",
        "spark.streaming.backpressure.enabled": "true",
        "spark.streaming.kafka.maxRatePerPartition": "1000",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true"
    },
    "spark_env_vars": {
        "PYSPARK_PYTHON": "/databricks/python3/bin/python3"
    },
    "custom_tags": {
        "project": "fraud-detection",
        "team": "data-science",
        "cost-center": "analytics"
    }
}

# Databricks Job Configuration for Scheduled Runs
job_config = {
    "name": "Fraud Detection Pipeline",
    "tasks": [
        {
            "task_key": "data_ingestion",
            "description": "Load and clean transaction data",
            "notebook_task": {
                "notebook_path": "/Repos/fraud-detection/notebooks/01_data_ingestion"
            },
            "new_cluster": cluster_config
        },
        {
            "task_key": "feature_engineering",
            "description": "Generate ML features",
            "depends_on": [{"task_key": "data_ingestion"}],
            "notebook_task": {
                "notebook_path": "/Repos/fraud-detection/notebooks/02_feature_engineering"
            },
            "existing_cluster_id": "${cluster_id}"
        },
        {
            "task_key": "model_training",
            "description": "Train RandomForest model",
            "depends_on": [{"task_key": "feature_engineering"}],
            "notebook_task": {
                "notebook_path": "/Repos/fraud-detection/notebooks/03_model_training"
            },
            "existing_cluster_id": "${cluster_id}"
        },
        {
            "task_key": "model_evaluation",
            "description": "Evaluate and log metrics",
            "depends_on": [{"task_key": "model_training"}],
            "notebook_task": {
                "notebook_path": "/Repos/fraud-detection/notebooks/04_model_evaluation"
            },
            "existing_cluster_id": "${cluster_id}"
        }
    ],
    "schedule": {
        "quartz_cron_expression": "0 0 6 * * ?",
        "timezone_id": "Europe/Paris",
        "pause_status": "UNPAUSED"
    },
    "email_notifications": {
        "on_failure": ["ahmed.dinari@email.com"],
        "on_success": ["ahmed.dinari@email.com"]
    }
}

# MLflow Configuration
mlflow_config = {
    "experiment_name": "/fraud-detection-experiment",
    "model_registry_name": "fraud-detection-model",
    "tracking_uri": "databricks"
}

# Streaming Configuration for Event Hubs
streaming_config = {
    "eventhubs.connectionString": "${EVENT_HUB_CONNECTION_STRING}",
    "eventhubs.consumerGroup": "spark-consumer",
    "eventhubs.startingPosition": '{"offset": "-1", "seqNo": -1, "enqueuedTime": null, "isInclusive": true}',
    "maxEventsPerTrigger": "1000"
}

# Cost Estimation (Monthly)
cost_estimate = {
    "databricks_workspace": {
        "tier": "Standard",
        "dbu_per_hour": 0.4,
        "hours_per_month": 720,
        "cost_usd": 150
    },
    "data_lake_storage": {
        "capacity_gb": 100,
        "cost_per_gb": 0.05,
        "cost_usd": 5
    },
    "event_hubs": {
        "tier": "Standard",
        "throughput_units": 1,
        "cost_usd": 25
    },
    "total_monthly_usd": 180
}

if __name__ == "__main__":
    import json
    print("=" * 60)
    print("AZURE DATABRICKS CONFIGURATION")
    print("=" * 60)
    print("\n📊 Cluster Configuration:")
    print(json.dumps(cluster_config, indent=2))
    print("\n💰 Monthly Cost Estimate:")
    for service, details in cost_estimate.items():
        if service != "total_monthly_usd":
            print(f"   {service}: ${details.get('cost_usd', details)}")
    print(f"\n   TOTAL: ${cost_estimate['total_monthly_usd']}/month")

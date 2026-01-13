# 🚀 Guide d'Intégration Azure & Federated Learning

## Table des Matières
1. [Azure Databricks - Déploiement](#1-azure-databricks---déploiement)
2. [Federated Learning - Implémentation](#2-federated-learning---implémentation)
3. [Architecture Complète](#3-architecture-complète)

---

## 1. Azure Databricks - Déploiement

### 1.1 Prérequis

```bash
# Installer Azure CLI
# Windows: winget install Microsoft.AzureCLI
# Ou télécharger: https://aka.ms/installazurecliwindows

# Se connecter à Azure
az login

# Vérifier la souscription
az account show
```

### 1.2 Créer les Ressources Azure

```bash
# Variables
RESOURCE_GROUP="fraud-detection-rg"
LOCATION="westeurope"
DATABRICKS_WORKSPACE="fraud-databricks-ws"
STORAGE_ACCOUNT="frauddatastorage"

# 1. Créer le Resource Group
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2. Créer le Storage Account (Data Lake Gen2)
az storage account create \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --kind StorageV2 \
    --hierarchical-namespace true

# 3. Créer le conteneur pour les données
az storage container create \
    --name fraud-data \
    --account-name $STORAGE_ACCOUNT

# 4. Créer Azure Databricks Workspace
az databricks workspace create \
    --resource-group $RESOURCE_GROUP \
    --name $DATABRICKS_WORKSPACE \
    --location $LOCATION \
    --sku standard
```

### 1.3 Uploader les Données vers Azure

```bash
# Obtenir la clé du storage
STORAGE_KEY=$(az storage account keys list \
    --account-name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --query '[0].value' -o tsv)

# Uploader le dataset
az storage blob upload \
    --account-name $STORAGE_ACCOUNT \
    --account-key $STORAGE_KEY \
    --container-name fraud-data \
    --file data/raw/creditcard.csv \
    --name raw/creditcard.csv
```

### 1.4 Configurer Databricks

1. **Accéder au workspace**: Portal Azure → Databricks → Launch Workspace

2. **Créer un Cluster**:
```python
# Dans Databricks, créer un notebook et configurer:
cluster_config = {
    "cluster_name": "fraud-detection-cluster",
    "spark_version": "13.3.x-scala2.12",
    "node_type_id": "Standard_DS3_v2",
    "autoscale": {
        "min_workers": 2,
        "max_workers": 8
    },
    "spark_conf": {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true"
    }
}
```

3. **Monter le Storage**:
```python
# Dans un notebook Databricks
storage_account = "frauddatastorage"
container = "fraud-data"
storage_key = dbutils.secrets.get(scope="fraud-scope", key="storage-key")

dbutils.fs.mount(
    source=f"wasbs://{container}@{storage_account}.blob.core.windows.net",
    mount_point="/mnt/fraud-data",
    extra_configs={f"fs.azure.account.key.{storage_account}.blob.core.windows.net": storage_key}
)

# Vérifier
display(dbutils.fs.ls("/mnt/fraud-data/raw/"))
```

### 1.5 Exécuter le Pipeline sur Databricks

```python
# Notebook: fraud_detection_pipeline.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow

# MLflow tracking
mlflow.set_experiment("/fraud-detection")

# Charger les données depuis Azure Storage
df = spark.read.csv("/mnt/fraud-data/raw/creditcard.csv", header=True, inferSchema=True)

print(f"Total transactions: {df.count()}")
print(f"Fraudes: {df.filter(df.Class == 1).count()}")

# Feature Engineering
feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

# Train/Test Split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Pipeline avec MLflow tracking
with mlflow.start_run(run_name="RandomForest_Azure"):
    # Transformer
    train_assembled = assembler.transform(train)
    scaler_model = scaler.fit(train_assembled)
    train_scaled = scaler_model.transform(train_assembled)
    
    # Entraîner
    rf = RandomForestClassifier(
        labelCol="Class",
        featuresCol="features",
        numTrees=100,
        maxDepth=10
    )
    model = rf.fit(train_scaled)
    
    # Évaluer
    test_assembled = assembler.transform(test)
    test_scaled = scaler_model.transform(test_assembled)
    predictions = model.transform(test_scaled)
    
    evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    
    # Log metrics
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("auc", auc)
    mlflow.spark.log_model(model, "model")
    
    print(f"AUC-ROC: {auc:.4f}")

# Sauvegarder le modèle
model.write().overwrite().save("/mnt/fraud-data/models/random_forest_v1")
```

### 1.6 Streaming avec Azure Event Hubs

```python
# Configuration Event Hubs
ehConf = {
    'eventhubs.connectionString': 
        sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(
            "Endpoint=sb://fraud-eventhub.servicebus.windows.net/;SharedAccessKeyName=listen;SharedAccessKey=xxx;EntityPath=transactions"
        )
}

# Lire le stream
stream_df = spark.readStream \
    .format("eventhubs") \
    .options(**ehConf) \
    .load()

# Appliquer le modèle en temps réel
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, DoubleType

schema = StructType([...])  # Définir le schéma

parsed_stream = stream_df \
    .select(from_json(col("body").cast("string"), schema).alias("data")) \
    .select("data.*")

# Scoring
scored_stream = model.transform(parsed_stream)

# Écrire les alertes
scored_stream.filter(col("prediction") == 1) \
    .writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/mnt/fraud-data/checkpoints/alerts") \
    .start("/mnt/fraud-data/alerts")
```

---

## 2. Federated Learning - Implémentation

### 2.1 Architecture Federated Learning

```
┌─────────────────────────────────────────────────────────────┐
│                    AGGREGATEUR CENTRAL                       │
│                  (Azure Functions / VM)                      │
│                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │ Global Model│  │  FedAvg     │  │  Model      │        │
│   │  Weights    │  │  Algorithm  │  │  Registry   │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   BANQUE A    │   │   BANQUE B    │   │   BANQUE C    │
│   (Client 1)  │   │   (Client 2)  │   │   (Client 3)  │
│               │   │               │   │               │
│ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────┐ │
│ │Local Data │ │   │ │Local Data │ │   │ │Local Data │ │
│ │(Private)  │ │   │ │(Private)  │ │   │ │(Private)  │ │
│ └───────────┘ │   │ └───────────┘ │   │ └───────────┘ │
│       │       │   │       │       │   │       │       │
│       ▼       │   │       ▼       │   │       ▼       │
│ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────┐ │
│ │Local Model│ │   │ │Local Model│ │   │ │Local Model│ │
│ │Training   │ │   │ │Training   │ │   │ │Training   │ │
│ └───────────┘ │   │ └───────────┘ │   │ └───────────┘ │
│       │       │   │       │       │   │       │       │
│       ▼       │   │       ▼       │   │       ▼       │
│  Gradients    │   │  Gradients    │   │  Gradients    │
│  (encrypted)  │   │  (encrypted)  │   │  (encrypted)  │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    ▼ Upload Gradients ▼
```

### 2.2 Implémentation avec PySpark + Flower

Créer le fichier `src/federated_learning.py`:

```python
"""
Federated Learning pour Détection de Fraude
Utilise Flower (flwr) pour l'orchestration
"""

import flwr as fl
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from typing import Dict, List, Tuple
import json


# ============================================================================
# CLIENT FEDERATED (Chaque banque)
# ============================================================================

class FraudDetectionClient(fl.client.NumPyClient):
    """Client Federated Learning pour une banque"""
    
    def __init__(self, client_id: str, data_path: str):
        self.client_id = client_id
        self.spark = SparkSession.builder \
            .appName(f"FederatedClient_{client_id}") \
            .getOrCreate()
        
        # Charger les données locales (privées)
        self.df = self.spark.read.csv(data_path, header=True, inferSchema=True)
        self.feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        
        # Préparer les features
        self._prepare_data()
        
        # Modèle local
        self.model = None
        
    def _prepare_data(self):
        """Prépare les données pour l'entraînement"""
        assembler = VectorAssembler(
            inputCols=self.feature_cols, 
            outputCol="features_raw"
        )
        self.df = assembler.transform(self.df)
        
        scaler = StandardScaler(inputCol="features_raw", outputCol="features")
        scaler_model = scaler.fit(self.df)
        self.df = scaler_model.transform(self.df)
        
        # Split train/test
        self.train_df, self.test_df = self.df.randomSplit([0.8, 0.2], seed=42)
        
    def get_parameters(self, config) -> List[np.ndarray]:
        """Retourne les paramètres du modèle local"""
        if self.model is None:
            # Initialiser avec des poids aléatoires
            return [np.random.randn(29).astype(np.float32)]
        
        # Extraire les coefficients du modèle Spark
        coefficients = self.model.coefficients.toArray()
        intercept = np.array([self.model.intercept])
        return [coefficients, intercept]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Met à jour les paramètres avec le modèle global"""
        # Les paramètres seront utilisés pour initialiser le prochain entraînement
        self.global_weights = parameters
        
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Entraîne le modèle sur les données locales"""
        print(f"[Client {self.client_id}] Entraînement local...")
        
        # Mettre à jour avec les poids globaux
        self.set_parameters(parameters)
        
        # Entraîner le modèle local
        lr = LogisticRegression(
            labelCol="Class",
            featuresCol="features",
            maxIter=10,  # Moins d'itérations pour federated
            regParam=0.01
        )
        self.model = lr.fit(self.train_df)
        
        # Retourner les nouveaux paramètres
        num_samples = self.train_df.count()
        return self.get_parameters(config), num_samples, {}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Évalue le modèle sur les données locales"""
        self.set_parameters(parameters)
        
        if self.model is None:
            return 0.0, 0, {"auc": 0.0}
        
        # Prédictions
        predictions = self.model.transform(self.test_df)
        
        # Évaluation
        evaluator = BinaryClassificationEvaluator(
            labelCol="Class",
            metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)
        
        num_samples = self.test_df.count()
        return float(1 - auc), num_samples, {"auc": float(auc)}


# ============================================================================
# SERVEUR FEDERATED (Aggregateur Central)
# ============================================================================

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Calcule la moyenne pondérée des métriques"""
    total_samples = sum([num_samples for num_samples, _ in metrics])
    
    weighted_auc = sum([
        num_samples * m["auc"] for num_samples, m in metrics
    ]) / total_samples
    
    return {"auc": weighted_auc}


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """Stratégie FedAvg personnalisée pour la détection de fraude"""
    
    def __init__(self, min_clients: int = 2, min_fit_clients: int = 2):
        super().__init__(
            min_available_clients=min_clients,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_fit_clients,
            evaluate_metrics_aggregation_fn=weighted_average
        )
        
    def aggregate_fit(self, server_round, results, failures):
        """Agrège les modèles avec FedAvg"""
        print(f"\n[Server] Round {server_round}: Agrégation de {len(results)} modèles")
        
        # Appeler l'agrégation standard FedAvg
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated is not None:
            print(f"[Server] Modèle global mis à jour")
            
        return aggregated


# ============================================================================
# SIMULATION LOCALE (3 Banques)
# ============================================================================

def simulate_federated_learning():
    """
    Simule le Federated Learning avec 3 clients (banques)
    Pour une vraie implémentation, chaque client serait sur une machine séparée
    """
    from pyspark.sql import SparkSession
    import pandas as pd
    
    print("=" * 60)
    print("FEDERATED LEARNING - SIMULATION")
    print("=" * 60)
    
    # Créer une session Spark
    spark = SparkSession.builder \
        .appName("FederatedSimulation") \
        .master("local[*]") \
        .getOrCreate()
    
    # Charger le dataset complet
    df = spark.read.csv("/app/data/raw/creditcard.csv", header=True, inferSchema=True)
    total = df.count()
    print(f"\nDataset total: {total} transactions")
    
    # Simuler 3 banques avec des partitions différentes
    # En réalité, chaque banque aurait ses propres données
    df_with_id = df.withColumn("row_id", F.monotonically_increasing_id())
    
    bank_a = df_with_id.filter(F.col("row_id") % 3 == 0)
    bank_b = df_with_id.filter(F.col("row_id") % 3 == 1)
    bank_c = df_with_id.filter(F.col("row_id") % 3 == 2)
    
    print(f"\nBanque A: {bank_a.count()} transactions")
    print(f"Banque B: {bank_b.count()} transactions")
    print(f"Banque C: {bank_c.count()} transactions")
    
    # Feature Engineering
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")
    
    def prepare_bank_data(bank_df, bank_name):
        assembled = assembler.transform(bank_df)
        scaler_model = scaler.fit(assembled)
        scaled = scaler_model.transform(assembled)
        train, test = scaled.randomSplit([0.8, 0.2], seed=42)
        print(f"  {bank_name}: Train={train.count()}, Test={test.count()}")
        return train, test, scaler_model
    
    print("\nPréparation des données par banque:")
    train_a, test_a, _ = prepare_bank_data(bank_a, "Banque A")
    train_b, test_b, _ = prepare_bank_data(bank_b, "Banque B")
    train_c, test_c, _ = prepare_bank_data(bank_c, "Banque C")
    
    # ========================================
    # FEDERATED LEARNING SIMULATION
    # ========================================
    
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    import numpy as np
    
    NUM_ROUNDS = 5
    
    # Initialiser les poids globaux
    global_weights = None
    
    evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
    
    print("\n" + "=" * 60)
    print("DÉMARRAGE FEDERATED LEARNING")
    print("=" * 60)
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_num} {'='*20}")
        
        local_weights = []
        local_samples = []
        local_aucs = []
        
        # Chaque banque entraîne localement
        for bank_name, train_df, test_df in [
            ("Banque A", train_a, test_a),
            ("Banque B", train_b, test_b),
            ("Banque C", train_c, test_c)
        ]:
            # Entraînement local
            lr = LogisticRegression(
                labelCol="Class",
                featuresCol="features",
                maxIter=10,
                regParam=0.01
            )
            model = lr.fit(train_df)
            
            # Extraire les poids
            weights = model.coefficients.toArray()
            intercept = model.intercept
            
            # Évaluer localement
            predictions = model.transform(test_df)
            auc = evaluator.evaluate(predictions)
            
            num_samples = train_df.count()
            local_weights.append((weights, intercept, num_samples))
            local_samples.append(num_samples)
            local_aucs.append(auc)
            
            print(f"  {bank_name}: AUC={auc:.4f}, Samples={num_samples}")
        
        # ========================================
        # FEDAVG: Agrégation centrale
        # ========================================
        total_samples = sum(local_samples)
        
        # Moyenne pondérée des poids
        avg_weights = np.zeros_like(local_weights[0][0])
        avg_intercept = 0.0
        
        for weights, intercept, n_samples in local_weights:
            weight_factor = n_samples / total_samples
            avg_weights += weight_factor * weights
            avg_intercept += weight_factor * intercept
        
        global_weights = (avg_weights, avg_intercept)
        
        # Calculer l'AUC global (moyenne pondérée)
        global_auc = sum(auc * n / total_samples 
                       for auc, n in zip(local_aucs, local_samples))
        
        print(f"\n  [AGGREGATEUR] AUC Global: {global_auc:.4f}")
    
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING TERMINÉ")
    print(f"AUC Final: {global_auc:.4f}")
    print("=" * 60)
    
    # Sauvegarder les résultats
    results = {
        "algorithm": "FedAvg",
        "num_rounds": NUM_ROUNDS,
        "num_clients": 3,
        "final_auc": float(global_auc),
        "clients": [
            {"name": "Banque A", "samples": int(local_samples[0]), "auc": float(local_aucs[0])},
            {"name": "Banque B", "samples": int(local_samples[1]), "auc": float(local_aucs[1])},
            {"name": "Banque C", "samples": int(local_samples[2]), "auc": float(local_aucs[2])}
        ]
    }
    
    with open("/app/outputs/metrics/federated_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRésultats sauvegardés: /app/outputs/metrics/federated_results.json")
    
    return results


if __name__ == "__main__":
    from pyspark.sql import functions as F
    simulate_federated_learning()
```

### 2.3 Exécuter la Simulation

```bash
# Dans le conteneur Docker
docker exec -it fraud-spark spark-submit \
    --master local[*] \
    /app/src/federated_learning.py
```

### 2.4 Déploiement Réel avec Azure

Pour un déploiement en production:

```
┌─────────────────────────────────────────────────────────────────┐
│                     AZURE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Azure Functions (Aggregateur)               │   │
│  │  - Reçoit les gradients chiffrés                        │   │
│  │  - Exécute FedAvg                                       │   │
│  │  - Distribue le modèle global                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│              ┌───────────────┼───────────────┐                 │
│              │               │               │                  │
│              ▼               ▼               ▼                  │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│  │  Azure Event  │ │  Azure Event  │ │  Azure Event  │        │
│  │  Hubs (A)     │ │  Hubs (B)     │ │  Hubs (C)     │        │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘        │
│          │                 │                 │                  │
│          ▼                 ▼                 ▼                  │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│  │ Databricks A  │ │ Databricks B  │ │ Databricks C  │        │
│  │ (Banque A)    │ │ (Banque B)    │ │ (Banque C)    │        │
│  │               │ │               │ │               │        │
│  │ Private Data  │ │ Private Data  │ │ Private Data  │        │
│  └───────────────┘ └───────────────┘ └───────────────┘        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Azure Key Vault                             │   │
│  │  - Clés de chiffrement                                  │   │
│  │  - Secrets pour communication sécurisée                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture Complète

### 3.1 Commandes de Déploiement Rapide

```bash
# 1. Créer toutes les ressources Azure
cd azure/
chmod +x deploy.sh
./deploy.sh

# 2. Configurer Databricks
az databricks workspace show --name fraud-databricks-ws --resource-group fraud-detection-rg

# 3. Déployer le modèle
databricks jobs create --json-file jobs/fraud_training_job.json
databricks jobs run-now --job-id <JOB_ID>
```

### 3.2 Coûts Estimés

| Service | Usage | Coût/Mois |
|---------|-------|-----------|
| Azure Databricks | Standard, 2-8 workers | ~$120 |
| Data Lake Gen2 | 100 GB | ~$5 |
| Event Hubs | Basic, 1M events | ~$15 |
| Azure Functions | Consumption | ~$5 |
| Key Vault | 10K operations | ~$3 |
| **Total** | | **~$150/mois** |

### 3.3 Checklist de Déploiement

- [ ] Créer Resource Group Azure
- [ ] Déployer Storage Account (Data Lake Gen2)
- [ ] Créer Azure Databricks Workspace
- [ ] Configurer Event Hubs pour streaming
- [ ] Uploader les données initiales
- [ ] Créer et lancer le cluster Databricks
- [ ] Déployer le pipeline MLlib
- [ ] Configurer le monitoring (Azure Monitor)
- [ ] (Optionnel) Déployer Federated Learning

---

## 📚 Ressources

- [Azure Databricks Documentation](https://docs.microsoft.com/azure/databricks/)
- [Flower Federated Learning](https://flower.dev/)
- [PySpark MLlib](https://spark.apache.org/docs/latest/ml-guide.html)
- [Azure Event Hubs + Spark](https://docs.microsoft.com/azure/event-hubs/event-hubs-spark-connector)

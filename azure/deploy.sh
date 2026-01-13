#!/bin/bash
# Azure Deployment Script for Fraud Detection Pipeline
# Run this script to deploy all Azure resources

set -e

echo "=========================================="
echo "AZURE FRAUD DETECTION DEPLOYMENT"
echo "=========================================="

# Configuration
RESOURCE_GROUP="fraud-detection-rg"
LOCATION="westeurope"
WORKSPACE_NAME="fraud-detection-workspace"
STORAGE_ACCOUNT="frauddetectiondl$(date +%s | tail -c 5)"
EVENT_HUB_NS="fraud-streaming-hub"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "\n${BLUE}[1/5] Creating Resource Group...${NC}"
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --tags project=fraud-detection environment=production

echo -e "\n${BLUE}[2/5] Deploying ARM Template...${NC}"
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file arm-template.json \
    --parameters \
        workspaceName=$WORKSPACE_NAME \
        storageAccountName=$STORAGE_ACCOUNT \
        eventHubNamespace=$EVENT_HUB_NS \
        location=$LOCATION

echo -e "\n${BLUE}[3/5] Getting Databricks Workspace URL...${NC}"
WORKSPACE_URL=$(az databricks workspace show \
    --resource-group $RESOURCE_GROUP \
    --name $WORKSPACE_NAME \
    --query "workspaceUrl" -o tsv)
echo "Workspace URL: https://$WORKSPACE_URL"

echo -e "\n${BLUE}[4/5] Getting Storage Account Key...${NC}"
STORAGE_KEY=$(az storage account keys list \
    --resource-group $RESOURCE_GROUP \
    --account-name $STORAGE_ACCOUNT \
    --query "[0].value" -o tsv)
echo "Storage account configured."

echo -e "\n${BLUE}[5/5] Getting Event Hub Connection String...${NC}"
EH_CONNECTION=$(az eventhubs namespace authorization-rule keys list \
    --resource-group $RESOURCE_GROUP \
    --namespace-name $EVENT_HUB_NS \
    --name RootManageSharedAccessKey \
    --query "primaryConnectionString" -o tsv)
echo "Event Hub configured."

echo -e "\n${GREEN}=========================================="
echo "DEPLOYMENT COMPLETE!"
echo "==========================================${NC}"
echo ""
echo "Resources deployed:"
echo "  ✅ Resource Group: $RESOURCE_GROUP"
echo "  ✅ Databricks Workspace: https://$WORKSPACE_URL"
echo "  ✅ Storage Account: $STORAGE_ACCOUNT"
echo "  ✅ Event Hub: $EVENT_HUB_NS"
echo ""
echo "Next steps:"
echo "  1. Access Databricks: https://$WORKSPACE_URL"
echo "  2. Create cluster with config from databricks_config.py"
echo "  3. Upload notebooks from src/ folder"
echo "  4. Configure Event Hub streaming"
echo ""
echo "Estimated monthly cost: ~\$180"

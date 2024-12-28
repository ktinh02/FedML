#!/bin/bash

# Initialize conda
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Activate conda environment
conda activate fedml

# Function to handle FedML login
fedml_login() {
    if [ "$FEDML_PROVIDER" = "true" ]; then
        if [ -z "$INFERENCE_GATEWAY_PORT" ] && [ -z "$INFERENCE_PROXY_PORT" ] && [ -z "$FEDML_CONNECTION_TYPE" ]; then
            fedml login -p "$FEDML_API_KEY" -v "$FEDML_ENV"
        else
            fedml login -p "$FEDML_API_KEY" -v "$FEDML_ENV" -mgp "$INFERENCE_GATEWAY_PORT" -wpp "$INFERENCE_PROXY_PORT" -wct "$FEDML_CONNECTION_TYPE"
        fi
    else
        if [ -z "$INFERENCE_GATEWAY_PORT" ] && [ -z "$INFERENCE_PROXY_PORT" ] && [ -z "$FEDML_CONNECTION_TYPE" ]; then
            fedml login "$FEDML_API_KEY" -v "$FEDML_ENV"
        else
            fedml login "$FEDML_API_KEY" -v "$FEDML_ENV" -mgp "$INFERENCE_GATEWAY_PORT" -wpp "$INFERENCE_PROXY_PORT" -wct "$FEDML_CONNECTION_TYPE"
        fi
    fi
}

# Print all environment variables
echo "=== Environment Variables ==="
printenv | sort
echo "==========================="

# Create devices.id file if FEDML_DEVICE_ID is provided
if [ -n "$FEDML_DEVICE_ID" ]; then
    mkdir -p "/root/.fedml/fedml-client/fedml/data/runner_infos"
    echo "$FEDML_DEVICE_ID" > "/root/.fedml/fedml-client/fedml/data/runner_infos/devices.id"
fi

# Start Redis server
redis-server --daemonize yes

# Login to FedML
fedml_login

# Keep container running
tail -f /dev/null
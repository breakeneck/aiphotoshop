#!/bin/bash

set -e

# Configuration
CONDA_ENV_NAME="aiphotoshop"
PYTHON_VERSION="3.11"

echo "=== Deploying AI Photo Shop ==="

# Pull latest code
echo "Fetching latest code from origin/master..."
git fetch origin master
git reset --hard origin/master

# Initialize conda for current shell
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Check if conda environment exists, create if not
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Creating conda environment: ${CONDA_ENV_NAME}..."
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV_NAME}..."
conda activate ${CONDA_ENV_NAME}

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting application..."
python main.py

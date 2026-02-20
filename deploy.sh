#!/bin/bash

set -e

# Configuration
CONDA_ENV_PATH="./env"
PYTHON_VERSION="3.10"

echo "=== Deploying AI Photo Shop ==="

# Pull latest code
echo "Fetching latest code from origin/master..."
git fetch origin master
git reset --hard origin/master

# Initialize conda for current shell
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Check if conda environment exists, create if not
if [ ! -d "${CONDA_ENV_PATH}" ]; then
    echo "Creating conda environment in: ${CONDA_ENV_PATH}..."
    conda create --prefix ${CONDA_ENV_PATH} python=${PYTHON_VERSION} -y
fi

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV_PATH}..."
conda activate ${CONDA_ENV_PATH}

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting application..."
python main.py

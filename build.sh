#!/bin/bash

# Upgrade pip to the latest version
pip install --upgrade pip

# Install all required dependencies for our project
pip install torch transformers accelerate fastapi uvicorn huggingface_hub

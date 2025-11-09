"""
Backend Configuration
Environment variables and settings for NeuroFlow backend
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"
MODELS_DIR = ML_DIR / "models"
DATA_DIR = ML_DIR / "data"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# CORS settings
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ML Model settings
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", 1))
FRAME_PROCESSING_FPS = int(os.getenv("FRAME_PROCESSING_FPS", 1))

# WebSocket settings
WS_HEARTBEAT_INTERVAL = int(os.getenv("WS_HEARTBEAT_INTERVAL", 30))
WS_MAX_MESSAGE_SIZE = int(os.getenv("WS_MAX_MESSAGE_SIZE", 10 * 1024 * 1024))  # 10MB

# Model paths
EMOTION_CNN_PATH = MODELS_DIR / "emotion_cnn.pth"
FOCUS_RNN_PATH = MODELS_DIR / "focus_rnn.pth"
DRL_AGENT_PATH = MODELS_DIR / "drl_agent.pth"
INTERVENTIONS_PATH = DATA_DIR / "interventions.json"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Session settings
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", 120))

# Performance settings
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", 10))

print(f"✓ Configuration loaded")
print(f"  - Models directory: {MODELS_DIR}")
print(f"  - Data directory: {DATA_DIR}")
print(f"  - GPU enabled: {USE_GPU}")
print(f"  - Debug mode: {DEBUG}")
"""
Configuration file for Dynamic Dilemma Zone Modeling System

Contains all hyperparameters, paths, and model settings.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "LSTM" / "outputs"
MODEL_DIR = BASE_DIR / "LSTM" / "models"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Data configuration
CSV_COLUMNS = [
    "frame_index",
    "tracker_id",
    "vehicle_type",
    "class_id",
    "x",
    "y",
    "distance",
    "time_s",
    "speed_kmh",
    "speed_ms",
    "traffic_light_status",
    "yellow_light",
    "distance_to_stop_line",
    "distance_to_front_vehicle",
    "traffic_density",
    "ttc",
    "yellow_light_decision"
]

# Feature configuration
FEATURE_COLUMNS = [
    "speed_ms",
    "distance_to_stop_line",
    "ttc",
    "distance_to_front_vehicle",
    "traffic_density",
    "class_id"
]

FEATURE_DIM = len(FEATURE_COLUMNS)  # 6 features

# Sequence configuration
SEQUENCE_LENGTH = 12  # Number of frames in sequence (8-15 range, default 12)
MIN_SEQUENCE_LENGTH = 8
MAX_SEQUENCE_LENGTH = 15
FRAMES_BEFORE_YELLOW = SEQUENCE_LENGTH  # Extract last N frames before yellow onset

# Model configuration
MODEL_TYPE = "LSTM"  # Options: "LSTM" or "CNN"
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

# CNN configuration (alternative to LSTM)
CNN_NUM_FILTERS = 64
CNN_KERNEL_SIZE = 3
CNN_STRIDE = 1

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# Optimizer configuration
OPTIMIZER = "Adam"
WEIGHT_DECAY = 1e-5

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "ReduceLROnPlateau"  # Options: "ReduceLROnPlateau", "StepLR"
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# Train/Validation split
TRAIN_VAL_SPLIT = 0.8
SPLIT_BY_DATE = True  # Split by date instead of random

# Data normalization
NORMALIZE_FEATURES = True
NORMALIZATION_METHOD = "standard"  # Options: "standard", "minmax"

# Evaluation metrics
METRICS = ["AUC", "F1", "Accuracy", "Precision", "Recall"]

# Dilemma Zone configuration
DZ_SPEED_RANGE = (0, 25)  # m/s
DZ_DISTANCE_RANGE = (0, 60)  # meters
DZ_GRID_RESOLUTION = 50  # Number of grid points per dimension
DZ_CONTOUR_LEVELS = [0.3, 0.5, 0.7]  # Probability levels for contour lines
DZ_BOUNDARY = (0.45, 0.55)  # Dilemma zone defined as P ∈ [0.45, 0.55]

# Vehicle type mapping for per-class DZ modeling
VEHICLE_TYPES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Explainability configuration
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP analysis
SHAP_BACKGROUND_SIZE = 50  # Size of background dataset for SHAP

# Visualization configuration
FIGURE_SIZE = (12, 8)
DPI = 300
PLOT_STYLE = "seaborn-v0_8"

# Output paths
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
SHAP_DIR = OUTPUT_DIR / "shap_analysis"
DZ_DIR = OUTPUT_DIR / "dilemma_zones"
MODEL_CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Create output subdirectories
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)
DZ_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42


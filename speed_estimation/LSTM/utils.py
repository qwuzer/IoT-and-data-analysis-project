"""
Utility functions for data loading, validation, normalization, and sequence processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings

from .config import (
    CSV_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    MAX_SEQUENCE_LENGTH,
    NORMALIZE_FEATURES,
    NORMALIZATION_METHOD,
    RANDOM_SEED
)

warnings.filterwarnings('ignore')


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data and validate required columns exist.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Map column names from inference_refactored.py format to expected format
    column_mapping = {
        'distance_to_stop_line_m': 'distance_to_stop_line',
        'distance_to_front_vehicle_m': 'distance_to_front_vehicle',
        'ttc_s': 'ttc',
    }
    
    # Rename columns if they exist
    df = df.rename(columns=column_mapping)
    
    # Check for required columns (allow some flexibility)
    required_cols = ['frame_index', 'tracker_id', 'yellow_light_decision']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def load_multiple_csv_files(csv_paths: List[str], make_tracker_ids_unique: bool = True) -> pd.DataFrame:
    """
    Load and combine multiple CSV files into a single DataFrame.
    
    Args:
        csv_paths: List of paths to CSV files
        make_tracker_ids_unique: If True, make tracker_ids unique across files by adding offset
        
    Returns:
        Combined DataFrame with data from all CSV files
    """
    if not csv_paths:
        raise ValueError("No CSV file paths provided")
    
    dataframes = []
    max_tracker_id = 0
    
    for idx, csv_path in enumerate(csv_paths):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"Warning: CSV file not found, skipping: {csv_path}")
            continue
        
        try:
            df = load_csv_data(str(csv_path))
            
            # Make tracker_ids unique across files if requested
            if make_tracker_ids_unique:
                # Add a large offset to tracker_ids to ensure uniqueness
                # Use file index * 100000 to provide plenty of room
                offset = idx * 100000
                df['tracker_id'] = df['tracker_id'] + offset
                max_tracker_id = max(max_tracker_id, df['tracker_id'].max())
            
            # Add source file identifier for tracking
            df['source_file'] = csv_path.stem
            
            dataframes.append(df)
            print(f"Loaded {len(df)} rows from {csv_path.name} ({len(df['tracker_id'].unique())} unique vehicles)")
            
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Total unique vehicles: {len(combined_df['tracker_id'].unique())}")
    print(f"  Source files: {len(combined_df['source_file'].unique())}")
    
    return combined_df


def load_csv_files_from_directory(directory: str, pattern: str = "*_speed_log*.csv", make_tracker_ids_unique: bool = True) -> pd.DataFrame:
    """
    Load all CSV files matching a pattern from a directory.
    
    Args:
        directory: Path to directory containing CSV files
        pattern: Glob pattern to match CSV files (default: "*_speed_log*.csv")
        make_tracker_ids_unique: If True, make tracker_ids unique across files
        
    Returns:
        Combined DataFrame with data from all matching CSV files
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    csv_files = sorted(directory.glob(pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found matching pattern '{pattern}' in {directory}")
    
    print(f"Found {len(csv_files)} CSV files matching pattern '{pattern}':")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    return load_multiple_csv_files([str(f) for f in csv_files], make_tracker_ids_unique=make_tracker_ids_unique)


def filter_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows where yellow_light_decision is valid (0 or 1, or 'go'/'stop').
    Only yellow-onset frames contain labels.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame with valid labels
    """
    # Handle both numeric and string formats
    # Convert string 'go'/'stop' to 0/1 if needed
    df = df.copy()
    
    if df['yellow_light_decision'].dtype == 'object':
        # String format: convert 'go' -> 0, 'stop' -> 1
        df['yellow_light_decision'] = df['yellow_light_decision'].map({
            'go': 0, 'GO': 0, 'Go': 0,
            'stop': 1, 'STOP': 1, 'Stop': 1
        })
    
    # Filter rows where yellow_light_decision is 0 (GO) or 1 (STOP)
    valid_df = df[df['yellow_light_decision'].isin([0, 1])].copy()
    
    return valid_df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate data quality and completeness.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data is valid, raises ValueError otherwise
    """
    # Check for required columns
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Check for NaN values in critical columns (yellow_light_decision can be NaN for non-labeled frames)
    critical_cols = ['tracker_id', 'frame_index']
    for col in critical_cols:
        if df[col].isna().any():
            raise ValueError(f"NaN values found in critical column: {col}")
    
    # Check that tracker_id and frame_index are numeric
    if not pd.api.types.is_numeric_dtype(df['tracker_id']):
        raise ValueError("tracker_id must be numeric")
    if not pd.api.types.is_numeric_dtype(df['frame_index']):
        raise ValueError("frame_index must be numeric")
    
    return True


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    min_val: Optional[np.ndarray] = None,
    max_val: Optional[np.ndarray] = None,
    fit: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize features using standard scaling or min-max scaling.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        mean: Mean values for standard scaling (if fit=False)
        std: Std values for standard scaling (if fit=False)
        min_val: Min values for min-max scaling (if fit=False)
        max_val: Max values for min-max scaling (if fit=False)
        fit: If True, compute normalization parameters from data
        
    Returns:
        Normalized features and normalization parameters dictionary
    """
    if not NORMALIZE_FEATURES:
        return features, {}
    
    features = np.array(features, dtype=np.float32)
    params = {}
    
    if NORMALIZATION_METHOD == "standard":
        if fit:
            mean = np.nanmean(features, axis=0)
            std = np.nanstd(features, axis=0)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            params = {'mean': mean, 'std': std}
        else:
            if mean is None or std is None:
                raise ValueError("mean and std must be provided when fit=False")
            params = {'mean': mean, 'std': std}
        
        normalized = (features - params['mean']) / params['std']
        
    elif NORMALIZATION_METHOD == "minmax":
        if fit:
            min_val = np.nanmin(features, axis=0)
            max_val = np.nanmax(features, axis=0)
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1.0, range_val)
            params = {'min': min_val, 'max': max_val, 'range': range_val}
        else:
            if min_val is None or max_val is None:
                raise ValueError("min_val and max_val must be provided when fit=False")
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1.0, range_val)
            params = {'min': min_val, 'max': max_val, 'range': range_val}
        
        normalized = (features - params['min']) / params['range']
    else:
        raise ValueError(f"Unknown normalization method: {NORMALIZATION_METHOD}")
    
    # Replace NaN and inf with 0
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return normalized, params


def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate sequence to target length.
    
    Args:
        sequence: Input sequence of shape (current_length, feature_dim)
        target_length: Desired sequence length
        
    Returns:
        Padded or truncated sequence of shape (target_length, feature_dim)
    """
    current_length = sequence.shape[0]
    
    if current_length == target_length:
        return sequence
    
    if current_length < target_length:
        # Pad with zeros (or repeat last frame)
        padding = np.zeros((target_length - current_length, sequence.shape[1]))
        # Option: pad with last frame instead of zeros
        # padding = np.repeat(sequence[-1:], target_length - current_length, axis=0)
        padded = np.vstack([sequence, padding])
        return padded
    else:
        # Truncate: take last target_length frames
        truncated = sequence[-target_length:]
        return truncated


def split_by_date(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by date (frame_index) instead of random split.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio of data for training
        
    Returns:
        Tuple of (train_df, val_df)
    """
    # Sort by frame_index
    df_sorted = df.sort_values('frame_index').copy()
    
    # Calculate split point
    total_frames = df_sorted['frame_index'].nunique()
    split_frame = int(total_frames * train_ratio)
    split_frame_value = df_sorted['frame_index'].unique()[split_frame]
    
    # Split
    train_df = df_sorted[df_sorted['frame_index'] <= split_frame_value].copy()
    val_df = df_sorted[df_sorted['frame_index'] > split_frame_value].copy()
    
    return train_df, val_df


def get_feature_array(df: pd.DataFrame) -> np.ndarray:
    """
    Extract feature array from DataFrame.
    
    Args:
        df: DataFrame with feature columns
        
    Returns:
        Feature array of shape (n_samples, feature_dim)
    """
    # Create a copy to avoid modifying original
    feature_df = pd.DataFrame(index=df.index)
    
    # Extract features, handling missing columns
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            feature_df[col] = df[col]
        else:
            # Column missing - fill with default value
            if col == 'class_id':
                feature_df[col] = 2  # Default to car
            else:
                feature_df[col] = 0.0
            print(f"Warning: Feature column '{col}' not found in data, using default value")
    
    # Fill NaN values
    for col in FEATURE_COLUMNS:
        if feature_df[col].isna().any():
            if col == 'class_id':
                feature_df[col] = feature_df[col].fillna(2)  # Default to car
            else:
                feature_df[col] = feature_df[col].fillna(0.0)
    
    features = feature_df[FEATURE_COLUMNS].values.astype(np.float32)
    
    return features


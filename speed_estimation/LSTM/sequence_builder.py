"""
Sequence Builder Module

Builds vehicle sequences from CSV data by extracting last N frames before yellow onset.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from .config import (
    SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    MAX_SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
    FEATURE_DIM,
    NORMALIZATION_METHOD
)
from .utils import (
    load_csv_data,
    filter_valid_labels,
    validate_data,
    get_feature_array,
    pad_or_truncate_sequence,
    normalize_features
)


def identify_yellow_onset_frames(df: pd.DataFrame) -> Dict[int, int]:
    """
    Identify the frame index where yellow light starts globally, then map to each vehicle.
    Uses global yellow onset as reference point for all vehicles.
    
    Args:
        df: DataFrame with traffic light data
        
    Returns:
        Dictionary mapping tracker_id to frame_index of yellow onset (global yellow start)
    """
    # Find the global yellow onset frame (first frame where yellow starts in entire video)
    global_yellow_frames = df[
        (df['yellow_light'] == True) | 
        (df['traffic_light_status'] == 'YELLOW')
    ]['frame_index']
    
    if global_yellow_frames.empty:
        return {}
    
    global_yellow_onset = int(global_yellow_frames.min())
    
    # For each vehicle, use the global yellow onset as reference
    # This ensures we can extract sequences for vehicles that appear before yellow
    yellow_onset = {}
    for tracker_id in df['tracker_id'].unique():
        yellow_onset[tracker_id] = global_yellow_onset
    
    return yellow_onset


def extract_sequence_for_vehicle(
    vehicle_data: pd.DataFrame,
    yellow_onset_frame: int,
    sequence_length: int = SEQUENCE_LENGTH
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Extract sequence of last N frames before yellow onset for a single vehicle.
    
    Args:
        vehicle_data: DataFrame for a single vehicle (ALL frames), sorted by frame_index
        yellow_onset_frame: Frame index where yellow light starts
        sequence_length: Number of frames to extract
        
    Returns:
        Tuple of (feature_sequence, label) or None if insufficient data
        - feature_sequence: shape (sequence_length, feature_dim)
        - label: 0 (GO) or 1 (STOP)
    """
    # Get the label from any frame where yellow_light_decision is set
    label_data = vehicle_data[vehicle_data['yellow_light_decision'].notna()].copy()
    
    if label_data.empty:
        return None
    
    # Convert label to int (handle both string and numeric)
    label_val = label_data.iloc[0]['yellow_light_decision']
    if isinstance(label_val, str):
        label = 1 if label_val.lower() == 'stop' else 0
    else:
        label = int(label_val)
    
    # Get all frames before yellow onset
    before_yellow = vehicle_data[vehicle_data['frame_index'] < yellow_onset_frame].copy()
    
    # If not enough frames before yellow, use frames up to (but not including) yellow onset
    # This handles cases where vehicle appears right at yellow onset
    if len(before_yellow) == 0:
        # No frames before yellow - try using frames up to and including yellow onset
        # (but exclude the actual yellow frame to avoid data leakage)
        up_to_yellow = vehicle_data[vehicle_data['frame_index'] <= yellow_onset_frame].copy()
        if len(up_to_yellow) <= 1:
            # Only yellow frame or less - can't build sequence
            return None
        # Use all frames except the last one (yellow frame)
        sequence_data = up_to_yellow.iloc[:-1].copy()
    elif len(before_yellow) < MIN_SEQUENCE_LENGTH:
        # Use all available frames before yellow, even if less than MIN_SEQUENCE_LENGTH
        # We'll pad later if needed
        sequence_data = before_yellow.copy()
    else:
        # Take last sequence_length frames before yellow
        sequence_data = before_yellow.tail(sequence_length).copy()
    
    # Extract features
    features = get_feature_array(sequence_data)
    
    # Pad or truncate to exact sequence_length
    features = pad_or_truncate_sequence(features, sequence_length)
    
    return (features, label)


def build_sequences_from_dataframe(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    normalize: bool = True,
    fit_normalizer: bool = True,
    normalizer_params: Optional[Dict] = None
) -> Tuple[List[np.ndarray], List[int], Optional[Dict]]:
    """
    Build sequences from DataFrame for all vehicles.
    
    Args:
        df: DataFrame with vehicle data (should include ALL frames, not just labeled ones)
        sequence_length: Number of frames per sequence
        normalize: Whether to normalize features
        fit_normalizer: Whether to fit normalizer on this data (True for train, False for val)
        
    Returns:
        Tuple of:
        - sequences: List of feature sequences, each of shape (sequence_length, feature_dim)
        - labels: List of labels (0=GO, 1=STOP)
        - normalizer_params: Dictionary with normalization parameters (if normalize=True)
    """
    # First, identify vehicles with valid labels
    df_valid_labels = filter_valid_labels(df)
    
    if len(df_valid_labels) == 0:
        raise ValueError("No valid labels found in data")
    
    # Validate data structure
    validate_data(df)
    
    # Identify yellow onset frames from full dataset
    yellow_onset = identify_yellow_onset_frames(df)
    
    sequences = []
    labels = []
    all_features = []
    
    # Group by tracker_id and extract sequences
    # Only process vehicles that have a yellow_light_decision
    vehicles_with_decisions = df_valid_labels['tracker_id'].unique()
    
    for tracker_id in vehicles_with_decisions:
        # Use ALL frames for this vehicle (not just labeled ones) to get history
        vehicle_all_frames = df[df['tracker_id'] == tracker_id].sort_values('frame_index').copy()
        
        # Get labeled frames to find decision
        vehicle_labeled = df_valid_labels[df_valid_labels['tracker_id'] == tracker_id].sort_values('frame_index').copy()
        
        if vehicle_labeled.empty:
            continue
        
        # Use yellow onset frame if available, otherwise use first frame with decision
        decision_frame = vehicle_labeled['frame_index'].min()
        yellow_onset_frame = yellow_onset.get(tracker_id, decision_frame)
        
        result = extract_sequence_for_vehicle(
            vehicle_all_frames,  # Use all frames for sequence extraction
            yellow_onset_frame,
            sequence_length
        )
        
        if result is not None:
            features, label = result
            sequences.append(features)
            labels.append(label)
            all_features.append(features)
    
    if not sequences:
        # Provide more informative error message
        num_vehicles_checked = len(vehicles_with_decisions)
        raise ValueError(
            f"No valid sequences extracted from data. "
            f"Checked {num_vehicles_checked} vehicles with decisions. "
            f"This might be due to insufficient frames before yellow onset. "
            f"Consider reducing MIN_SEQUENCE_LENGTH or checking data quality."
        )
    
    # Normalize features if requested
    normalizer_params_out = None
    if normalize:
        # Flatten all sequences for normalization
        all_features_flat = np.vstack(all_features)
        
        # Normalize
        if fit_normalizer:
            normalized_flat, normalizer_params_out = normalize_features(
                all_features_flat,
                fit=True
            )
        else:
            # For validation/test, use provided normalizer_params
            if normalizer_params is None:
                raise ValueError("normalizer_params must be provided when fit_normalizer=False")
            
            # Use the normalize_features function already imported
            if NORMALIZATION_METHOD == "standard":
                normalized_flat, _ = normalize_features(
                    all_features_flat,
                    fit=False,
                    mean=normalizer_params.get('mean'),
                    std=normalizer_params.get('std')
                )
            elif NORMALIZATION_METHOD == "minmax":
                normalized_flat, _ = normalize_features(
                    all_features_flat,
                    fit=False,
                    min_val=normalizer_params.get('min'),
                    max_val=normalizer_params.get('max')
                )
            else:
                raise ValueError(f"Unknown normalization method: {NORMALIZATION_METHOD}")
        
        # Reshape back to sequences
        sequences = [
            normalized_flat[i*sequence_length:(i+1)*sequence_length]
            for i in range(len(sequences))
        ]
    
    return sequences, labels, normalizer_params_out


def build_sequences_from_csv(
    csv_path: str,
    sequence_length: int = SEQUENCE_LENGTH,
    normalize: bool = True,
    fit_normalizer: bool = True
) -> Tuple[List[np.ndarray], List[int], Optional[Dict]]:
    """
    Build sequences from CSV file.
    
    Args:
        csv_path: Path to CSV file
        sequence_length: Number of frames per sequence
        normalize: Whether to normalize features
        fit_normalizer: Whether to fit normalizer on this data
        
    Returns:
        Tuple of (sequences, labels, normalizer_params)
    """
    # Load data
    df = load_csv_data(csv_path)
    
    # Build sequences
    return build_sequences_from_dataframe(
        df,
        sequence_length,
        normalize,
        fit_normalizer
    )


def get_sequence_statistics(sequences: List[np.ndarray], labels: List[int]) -> Dict:
    """
    Get statistics about the sequences.
    
    Args:
        sequences: List of sequences
        labels: List of labels
        
    Returns:
        Dictionary with statistics
    """
    labels_array = np.array(labels)
    
    stats = {
        'total_sequences': len(sequences),
        'sequence_length': sequences[0].shape[0] if sequences else 0,
        'feature_dim': sequences[0].shape[1] if sequences else 0,
        'stop_count': int(np.sum(labels_array == 1)),
        'go_count': int(np.sum(labels_array == 0)),
        'stop_ratio': float(np.mean(labels_array == 1)),
        'go_ratio': float(np.mean(labels_array == 0))
    }
    
    return stats


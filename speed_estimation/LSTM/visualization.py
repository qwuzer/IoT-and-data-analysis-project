"""
Visualization and Analytics Tools

Provides comprehensive visualization tools for model analysis, predictions, and dilemma zone insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

from .config import (
    VISUALIZATION_DIR,
    FIGURE_SIZE,
    DPI,
    PLOT_STYLE,
    FEATURE_COLUMNS
)

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use(PLOT_STYLE)


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path = None
):
    """
    Plot training history (loss curves).
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Path = None
):
    """
    Plot distribution of predictions by true class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1]))
    
    # Distribution for GO class
    go_probs = y_pred_proba[y_true == 0]
    axes[0].hist(go_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0].set_xlabel('P(STOP)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Prediction Distribution: GO Class')
    axes[0].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution for STOP class
    stop_probs = y_pred_proba[y_true == 1]
    axes[1].hist(stop_probs, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('P(STOP)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Prediction Distribution: STOP Class')
    axes[1].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Prediction distribution plot saved to: {save_path}")
    else:
        plt.show()


def plot_feature_distributions(
    sequences: List[np.ndarray],
    feature_names: List[str] = FEATURE_COLUMNS,
    save_path: Path = None
):
    """
    Plot distributions of features across sequences.
    
    Args:
        sequences: List of sequences
        feature_names: Names of features
        save_path: Path to save plot
    """
    # Flatten sequences
    all_features = np.vstack(sequences)
    
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1] * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature_name in enumerate(feature_names):
        feature_data = all_features[:, i]
        axes[i].hist(feature_data, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution: {feature_name}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Feature distribution plot saved to: {save_path}")
    else:
        plt.show()


def plot_correlation_matrix(
    sequences: List[np.ndarray],
    feature_names: List[str] = FEATURE_COLUMNS,
    save_path: Path = None
):
    """
    Plot correlation matrix of features.
    
    Args:
        sequences: List of sequences
        feature_names: Names of features
        save_path: Path to save plot
    """
    # Flatten sequences and compute correlation
    all_features = np.vstack(sequences)
    df = pd.DataFrame(all_features, columns=feature_names)
    corr_matrix = df.corr()
    
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Correlation matrix plot saved to: {save_path}")
    else:
        plt.show()


def plot_speed_vs_distance_scatter(
    sequences: List[np.ndarray],
    labels: List[int],
    feature_names: List[str] = FEATURE_COLUMNS,
    save_path: Path = None
):
    """
    Plot speed vs distance scatter plot colored by prediction.
    
    Args:
        sequences: List of sequences
        labels: True labels
        feature_names: Names of features
        save_path: Path to save plot
    """
    # Extract last timestep features (most relevant for decision)
    last_features = np.array([seq[-1] for seq in sequences])
    
    speed_idx = feature_names.index('speed_ms') if 'speed_ms' in feature_names else 0
    distance_idx = feature_names.index('distance_to_stop_line') if 'distance_to_stop_line' in feature_names else 1
    
    speeds = last_features[:, speed_idx]
    distances = last_features[:, distance_idx]
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot by class
    go_mask = np.array(labels) == 0
    stop_mask = np.array(labels) == 1
    
    plt.scatter(speeds[go_mask], distances[go_mask], alpha=0.6, label='GO', color='green', s=50)
    plt.scatter(speeds[stop_mask], distances[stop_mask], alpha=0.6, label='STOP', color='red', s=50)
    
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Distance to Stop Line (m)')
    plt.title('Speed vs Distance: Decision Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Speed vs distance scatter plot saved to: {save_path}")
    else:
        plt.show()


def plot_temporal_feature_evolution(
    sequences: List[np.ndarray],
    feature_name: str,
    feature_names: List[str] = FEATURE_COLUMNS,
    save_path: Path = None
):
    """
    Plot how a feature evolves over time in sequences.
    
    Args:
        sequences: List of sequences
        feature_name: Name of feature to plot
        feature_names: Names of features
        save_path: Path to save plot
    """
    if feature_name not in feature_names:
        raise ValueError(f"Feature {feature_name} not found in feature_names")
    
    feature_idx = feature_names.index(feature_name)
    
    # Extract feature values over time
    feature_evolution = np.array([seq[:, feature_idx] for seq in sequences])
    
    # Compute mean and std
    mean_evolution = feature_evolution.mean(axis=0)
    std_evolution = feature_evolution.std(axis=0)
    
    plt.figure(figsize=FIGURE_SIZE)
    
    time_steps = range(len(mean_evolution))
    plt.plot(time_steps, mean_evolution, 'b-', linewidth=2, label='Mean')
    plt.fill_between(
        time_steps,
        mean_evolution - std_evolution,
        mean_evolution + std_evolution,
        alpha=0.3,
        label='±1 Std Dev'
    )
    
    plt.xlabel('Time Step (frames before yellow)')
    plt.ylabel(feature_name)
    plt.title(f'Temporal Evolution: {feature_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Temporal evolution plot saved to: {save_path}")
    else:
        plt.show()


def generate_comprehensive_report(
    sequences: List[np.ndarray],
    labels: List[int],
    y_pred_proba: np.ndarray,
    train_losses: List[float] = None,
    val_losses: List[float] = None,
    output_dir: Path = None
):
    """
    Generate comprehensive visualization report.
    
    Args:
        sequences: List of sequences
        labels: True labels
        y_pred_proba: Predicted probabilities
        train_losses: Training losses (optional)
        val_losses: Validation losses (optional)
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = VISUALIZATION_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating comprehensive visualization report...")
    print(f"Output directory: {output_dir}")
    
    # Training history
    if train_losses and val_losses:
        plot_training_history(
            train_losses, val_losses,
            save_path=output_dir / 'training_history.png'
        )
    
    # Prediction distribution
    plot_prediction_distribution(
        np.array(labels), y_pred_proba,
        save_path=output_dir / 'prediction_distribution.png'
    )
    
    # Feature distributions
    plot_feature_distributions(
        sequences,
        save_path=output_dir / 'feature_distributions.png'
    )
    
    # Correlation matrix
    plot_correlation_matrix(
        sequences,
        save_path=output_dir / 'correlation_matrix.png'
    )
    
    # Speed vs distance
    plot_speed_vs_distance_scatter(
        sequences, labels,
        save_path=output_dir / 'speed_vs_distance.png'
    )
    
    # Temporal evolution for key features
    key_features = ['speed_ms', 'distance_to_stop_line', 'ttc']
    for feat_name in key_features:
        if feat_name in FEATURE_COLUMNS:
            plot_temporal_feature_evolution(
                sequences, feat_name,
                save_path=output_dir / f'temporal_evolution_{feat_name}.png'
            )
    
    print(f"\nComprehensive report generated! Saved to: {output_dir}")


def main():
    """
    Command-line interface for visualization tools.
    """
    import argparse
    from .evaluate_model import load_model, predict
    from .sequence_builder import build_sequences_from_csv
    
    parser = argparse.ArgumentParser(description='Generate Visualization Report')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--plot_type',
        type=str,
        default='all',
        choices=['all', 'training', 'predictions', 'features', 'correlation', 'scatter', 'temporal'],
        help='Type of plot to generate'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, normalizer_params, _ = load_model(args.checkpoint_path, device)
    
    # Load sequences - need to use build_sequences_from_dataframe with normalizer params
    from .utils import load_csv_data
    from .sequence_builder import build_sequences_from_dataframe
    from .config import SEQUENCE_LENGTH
    
    df = load_csv_data(args.csv_path)
    sequences, labels, _ = build_sequences_from_dataframe(
        df,
        sequence_length=SEQUENCE_LENGTH,
        normalize=True,
        fit_normalizer=False,
        normalizer_params=normalizer_params
    )
    
    # Get predictions
    y_pred_proba = predict(model, sequences, device)
    
    # Load training history if available
    checkpoint_dir = Path(args.checkpoint_path).parent
    metadata_path = checkpoint_dir / 'training_metadata.json'
    
    train_losses = None
    val_losses = None
    
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            train_losses = metadata.get('train_losses', [])
            val_losses = metadata.get('val_losses', [])
    
    # Generate visualizations
    if args.plot_type == 'all':
        generate_comprehensive_report(
            sequences, labels, y_pred_proba,
            train_losses, val_losses,
            output_dir=args.output_dir
        )
    else:
        output_dir = Path(args.output_dir) if args.output_dir else VISUALIZATION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.plot_type == 'training' and train_losses and val_losses:
            plot_training_history(train_losses, val_losses, output_dir / 'training_history.png')
        elif args.plot_type == 'predictions':
            plot_prediction_distribution(np.array(labels), y_pred_proba, output_dir / 'prediction_distribution.png')
        elif args.plot_type == 'features':
            plot_feature_distributions(sequences, save_path=output_dir / 'feature_distributions.png')
        elif args.plot_type == 'correlation':
            plot_correlation_matrix(sequences, save_path=output_dir / 'correlation_matrix.png')
        elif args.plot_type == 'scatter':
            plot_speed_vs_distance_scatter(sequences, labels, save_path=output_dir / 'speed_vs_distance.png')
        elif args.plot_type == 'temporal':
            for feat_name in ['speed_ms', 'distance_to_stop_line', 'ttc']:
                if feat_name in FEATURE_COLUMNS:
                    plot_temporal_feature_evolution(sequences, feat_name, save_path=output_dir / f'temporal_evolution_{feat_name}.png')


if __name__ == '__main__':
    main()


"""
Explainability Module for Dilemma Zone Model

Provides SHAP analysis, feature importance, temporal attribution, and partial dependence plots.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from .config import (
    SHAP_SAMPLE_SIZE,
    SHAP_BACKGROUND_SIZE,
    SHAP_DIR,
    VISUALIZATION_DIR,
    FEATURE_COLUMNS,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    FIGURE_SIZE,
    DPI,
    PLOT_STYLE
)
from .model_architecture import DilemmaZoneModel

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use(PLOT_STYLE)


class ModelWrapper:
    """
    Wrapper for PyTorch model to work with SHAP.
    """
    
    def __init__(self, model: DilemmaZoneModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Predict function for SHAP.
        
        Args:
            x: Input array of shape (n_samples, sequence_length, feature_dim)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x)
        
        return predictions.cpu().numpy().flatten()


def compute_shap_values(
    model: DilemmaZoneModel,
    background_data: np.ndarray,
    test_data: np.ndarray,
    device: torch.device,
    explainer_type: str = "deep"
):
    """
    Compute SHAP values for model predictions.
    
    Args:
        model: Trained model
        background_data: Background dataset for SHAP (shape: n_background, seq_len, feat_dim)
        test_data: Test data to explain (shape: n_test, seq_len, feat_dim)
        device: Device to run on
        explainer_type: Type of SHAP explainer ("deep" for DeepExplainer, "kernel" for KernelExplainer)
        
    Returns:
        Tuple of (SHAP values, explainer)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    model_wrapper = ModelWrapper(model, device)
    
    if explainer_type == "deep":
        # Use DeepExplainer for neural networks
        # Convert numpy arrays to torch tensors
        background_tensor = torch.FloatTensor(background_data).to(device)
        explainer = shap.DeepExplainer(model, background_tensor)
    elif explainer_type == "kernel":
        # Use KernelExplainer (slower but more general)
        explainer = shap.KernelExplainer(
            model_wrapper,
            background_data[:SHAP_BACKGROUND_SIZE]  # Limit background size
        )
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    # Convert test data to tensor if using DeepExplainer
    if explainer_type == "deep":
        test_tensor = torch.FloatTensor(test_data).to(device)
        shap_values = explainer.shap_values(test_tensor)
    else:
        shap_values = explainer.shap_values(test_data)
    
    return shap_values, explainer


def get_feature_importance(shap_values: shap.Explanation) -> Dict[str, float]:
    """
    Get feature importance from SHAP values.
    
    Args:
        shap_values: SHAP explanation object
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Calculate mean absolute SHAP values per feature
    if isinstance(shap_values, list):
        # Binary classification: use first class (STOP)
        shap_values = shap_values[0]
    
    # Average over samples and time steps
    feature_importance = np.abs(shap_values.values).mean(axis=(0, 1))  # (feature_dim,)
    
    importance_dict = {
        feature_name: float(importance)
        for feature_name, importance in zip(FEATURE_COLUMNS, feature_importance)
    }
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict


def plot_shap_summary(shap_values: shap.Explanation, save_path: Path, max_display: int = 10):
    """
    Plot SHAP summary plot.
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    shap.summary_plot(shap_values, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to: {save_path}")


def plot_shap_force_plot(
    shap_values: shap.Explanation,
    test_data: np.ndarray,
    sample_idx: int,
    save_path: Path
):
    """
    Plot SHAP force plot for a single prediction.
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Get SHAP values for this sample
    shap_values_sample = shap_values[sample_idx]
    test_sample = test_data[sample_idx]
    
    # Create explanation object for this sample
    explanation = shap.Explanation(
        values=shap_values_sample.values,
        base_values=shap_values_sample.base_values,
        data=test_sample,
        feature_names=FEATURE_COLUMNS
    )
    
    # Plot force plot
    shap.plots.force(explanation, show=False, matplotlib=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"SHAP force plot saved to: {save_path}")


def plot_temporal_attribution(
    shap_values: shap.Explanation,
    save_path: Path
):
    """
    Plot temporal feature attribution (SHAP values per timestep).
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Average SHAP values over samples, get per timestep
    # shap_values shape: (n_samples, seq_len, feat_dim)
    temporal_importance = np.abs(shap_values.values).mean(axis=0)  # (seq_len, feat_dim)
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot heatmap
    sns.heatmap(
        temporal_importance.T,
        xticklabels=[f"t-{SEQUENCE_LENGTH-i}" for i in range(SEQUENCE_LENGTH)],
        yticklabels=FEATURE_COLUMNS,
        cmap='viridis',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Mean |SHAP Value|'}
    )
    
    plt.xlabel('Time Step (frames before yellow)')
    plt.ylabel('Feature')
    plt.title('Temporal Feature Attribution')
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Temporal attribution plot saved to: {save_path}")


def plot_partial_dependence(
    model: DilemmaZoneModel,
    background_data: np.ndarray,
    feature_idx: int,
    feature_name: str,
    feature_range: Tuple[float, float],
    n_points: int = 50,
    save_path: Path = None
):
    """
    Plot partial dependence plot for a single feature.
    
    Args:
        model: Trained model
        background_data: Background data for averaging
        feature_idx: Index of feature to vary
        feature_name: Name of feature
        feature_range: (min, max) range for feature values
        n_points: Number of points to evaluate
        save_path: Path to save plot
    """
    device = next(model.parameters()).device
    
    # Create range of values for this feature
    feature_values = np.linspace(feature_range[0], feature_range[1], n_points)
    
    # Use mean of background data as baseline
    baseline = background_data.mean(axis=0, keepdims=True)  # (1, seq_len, feat_dim)
    baseline = np.repeat(baseline, n_points, axis=0)  # (n_points, seq_len, feat_dim)
    
    # Vary the feature of interest
    # Apply variation to all timesteps (or just last timestep)
    baseline[:, -1, feature_idx] = feature_values  # Vary last timestep
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        baseline_tensor = torch.FloatTensor(baseline).to(device)
        predictions = model(baseline_tensor).cpu().numpy().flatten()
    
    # Plot
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(feature_values, predictions, linewidth=2)
    plt.xlabel(feature_name)
    plt.ylabel('P(STOP)')
    plt.title(f'Partial Dependence: {feature_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Partial dependence plot saved to: {save_path}")
    else:
        plt.show()


def generate_explainability_report(
    model: DilemmaZoneModel,
    background_sequences: List[np.ndarray],
    test_sequences: List[np.ndarray],
    test_labels: List[int],
    device: torch.device,
    output_dir: Path = None,
    explainer_type: str = "deep"
):
    """
    Generate comprehensive explainability report.
    
    Args:
        model: Trained model
        background_sequences: Background sequences for SHAP
        test_sequences: Test sequences to explain
        test_labels: Test labels
        device: Device to run on
        output_dir: Directory to save outputs
        explainer_type: Type of SHAP explainer
    """
    if output_dir is None:
        output_dir = SHAP_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating explainability report...")
    print(f"Output directory: {output_dir}")
    
    # Convert to numpy arrays
    background_array = np.array(background_sequences)
    test_array = np.array(test_sequences)
    
    # Limit sizes for efficiency
    if len(background_array) > SHAP_BACKGROUND_SIZE:
        background_array = background_array[:SHAP_BACKGROUND_SIZE]
    if len(test_array) > SHAP_SAMPLE_SIZE:
        test_array = test_array[:SHAP_SAMPLE_SIZE]
    
    print(f"Background samples: {len(background_array)}")
    print(f"Test samples: {len(test_array)}")
    
    # Compute SHAP values
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not available. Skipping SHAP analysis.")
        return
    
    print("Computing SHAP values...")
    shap_values, explainer = compute_shap_values(
        model, background_array, test_array, device, explainer_type
    )
    
    # Get feature importance
    print("Calculating feature importance...")
    feature_importance = get_feature_importance(shap_values)
    
    print("\nFeature Importance (from SHAP):")
    print("-" * 50)
    for feature, importance in feature_importance.items():
        print(f"  {feature}: {importance:.4f}")
    
    # Save feature importance
    import json
    importance_path = output_dir / 'feature_importance.json'
    with open(importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print(f"\nFeature importance saved to: {importance_path}")
    
    # Generate plots
    print("\nGenerating SHAP plots...")
    
    # Summary plot
    plot_shap_summary(shap_values, output_dir / 'shap_summary.png')
    
    # Temporal attribution
    plot_temporal_attribution(shap_values, output_dir / 'temporal_attribution.png')
    
    # Force plots for a few samples
    num_force_plots = min(3, len(test_array))
    for i in range(num_force_plots):
        plot_shap_force_plot(
            shap_values, test_array, i,
            output_dir / f'shap_force_plot_sample_{i}.png'
        )
    
    # Partial dependence plots
    print("\nGenerating partial dependence plots...")
    
    # Get feature ranges from background data
    feature_ranges = []
    for feat_idx in range(FEATURE_DIM):
        feat_values = background_array[:, :, feat_idx].flatten()
        feat_min = float(np.percentile(feat_values, 5))
        feat_max = float(np.percentile(feat_values, 95))
        feature_ranges.append((feat_min, feat_max))
    
    # Plot for key features
    key_features = ['speed_ms', 'distance_to_stop_line', 'ttc']
    for feat_name in key_features:
        if feat_name in FEATURE_COLUMNS:
            feat_idx = FEATURE_COLUMNS.index(feat_name)
            plot_partial_dependence(
                model, background_array, feat_idx, feat_name,
                feature_ranges[feat_idx],
                save_path=output_dir / f'partial_dependence_{feat_name}.png'
            )
    
    print(f"\nExplainability report complete! Saved to: {output_dir}")


def main():
    """
    Command-line interface for explainability analysis.
    """
    import argparse
    from .evaluate_model import load_model
    from .sequence_builder import build_sequences_from_csv
    
    parser = argparse.ArgumentParser(description='Generate Explainability Report')
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
        help='Path to CSV file for analysis'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for explainability results'
    )
    parser.add_argument(
        '--explainer_type',
        type=str,
        default='deep',
        choices=['deep', 'kernel'],
        help='Type of SHAP explainer'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, normalizer_params, model_config = load_model(args.checkpoint_path, device)
    
    # Load and build sequences
    from .utils import load_csv_data
    from .sequence_builder import build_sequences_from_dataframe
    
    df = load_csv_data(args.csv_path)
    sequences, labels, _ = build_sequences_from_dataframe(
        df,
        sequence_length=SEQUENCE_LENGTH,
        normalize=True,
        fit_normalizer=False,
        normalizer_params=normalizer_params
    )
    
    # Split into background and test
    n_background = min(SHAP_BACKGROUND_SIZE, len(sequences) // 2)
    background_sequences = sequences[:n_background]
    test_sequences = sequences[n_background:n_background + SHAP_SAMPLE_SIZE]
    test_labels = labels[n_background:n_background + SHAP_SAMPLE_SIZE]
    
    # Generate report
    generate_explainability_report(
        model, background_sequences, test_sequences, test_labels,
        device, args.output_dir, args.explainer_type
    )


if __name__ == '__main__':
    main()


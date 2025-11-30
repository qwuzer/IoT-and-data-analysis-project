"""
Model Evaluation Script

Evaluates trained model and computes metrics: AUC, F1, calibration curves, ROC, Precision-Recall.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
import warnings

from .config import (
    MODEL_DIR,
    MODEL_CHECKPOINT_DIR,
    OUTPUT_DIR,
    VISUALIZATION_DIR,
    METRICS,
    RANDOM_SEED
)
from .sequence_builder import build_sequences_from_csv, build_sequences_from_dataframe
from .train_model import SequenceDataset
from .utils import load_csv_data, filter_valid_labels, split_by_date
from .config import TRAIN_VAL_SPLIT, SPLIT_BY_DATE, SEQUENCE_LENGTH
from .model_architecture import create_model, DilemmaZoneModel

warnings.filterwarnings('ignore')

# Set random seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, normalizer_params, model_config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'lstm')
    input_dim = model_config.get('input_dim', 6)
    sequence_length = model_config.get('sequence_length', 12)
    
    # Create model
    model = create_model(
        model_type=model_type,
        input_dim=input_dim,
        sequence_length=sequence_length
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get normalization parameters
    normalizer_params = checkpoint.get('normalizer_params', None)
    
    return model, normalizer_params, model_config


def predict(model, sequences, device, batch_size=32):
    """
    Get predictions from model.
    
    Args:
        model: Trained model
        sequences: List of sequences
        device: Device to run on
        batch_size: Batch size for inference
        
    Returns:
        Numpy array of probabilities
    """
    dataset = SequenceDataset(sequences, [0] * len(sequences))  # Dummy labels
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch_sequences, _ in loader:
            batch_sequences = batch_sequences.to(device)
            probs = model(batch_sequences)
            predictions.append(probs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0).flatten()


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics


def plot_roc_curve(y_true, y_pred_proba, save_path):
    """
    Plot ROC curve.
    """
    if len(np.unique(y_true)) <= 1:
        print("Warning: Cannot plot ROC curve with only one class")
        return
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def plot_precision_recall_curve(y_true, y_pred_proba, save_path):
    """
    Plot Precision-Recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = np.trapz(precision, recall)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Precision-Recall curve saved to: {save_path}")


def plot_calibration_curve(y_true, y_pred_proba, save_path, n_bins=10):
    """
    Plot calibration curve.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Calibration curve saved to: {save_path}")


def evaluate(
    checkpoint_path: str,
    csv_path: str = None,
    sequences: list = None,
    labels: list = None,
    normalizer_params: dict = None,
    output_dir: str = None
):
    """
    Evaluate model on test data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        csv_path: Path to CSV file (if sequences not provided)
        sequences: Pre-built sequences (optional)
        labels: Pre-built labels (optional)
        normalizer_params: Normalization parameters (if using CSV)
        output_dir: Directory to save evaluation results
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR / "evaluation"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model, checkpoint_norm_params, model_config = load_model(checkpoint_path, device)
    print(f"Model type: {model_config.get('model_type', 'unknown')}")
    
    # Load or use provided data
    if sequences is None or labels is None:
        if csv_path is None:
            raise ValueError("Either provide csv_path or sequences/labels")
        
        print(f"Loading data from: {csv_path}")
        df = load_csv_data(csv_path)
        
        # Use validation split or all data
        if SPLIT_BY_DATE:
            train_df, val_df = split_by_date(df, train_ratio=TRAIN_VAL_SPLIT)
            eval_df = val_df
        else:
            from sklearn.model_selection import train_test_split
            df_valid = filter_valid_labels(df)
            _, eval_df = train_test_split(
                df_valid,
                test_size=1 - TRAIN_VAL_SPLIT,
                random_state=RANDOM_SEED
            )
        
        print(f"Evaluation samples: {len(eval_df)}")
        
        # Use normalizer params from checkpoint if available
        norm_params = normalizer_params or checkpoint_norm_params
        
        print("Building evaluation sequences...")
        try:
            sequences, labels, _ = build_sequences_from_dataframe(
                eval_df,
                sequence_length=SEQUENCE_LENGTH,
                normalize=True,
                fit_normalizer=False,
                normalizer_params=norm_params
            )
        except ValueError as e:
            print(f"Warning: Could not build evaluation sequences from validation set: {e}")
            print("Using all available data for evaluation (small dataset).")
            # Build sequences from all data
            sequences, labels, _ = build_sequences_from_dataframe(
                df,
                sequence_length=SEQUENCE_LENGTH,
                normalize=True,
                fit_normalizer=False,
                normalizer_params=norm_params
            )
    
    print(f"Evaluating on {len(sequences)} sequences")
    
    # Get predictions
    print("Generating predictions...")
    y_pred_proba = predict(model, sequences, device)
    y_true = np.array(labels)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred_proba)
    
    # Print metrics
    print("\n" + "=" * 80)
    print("Evaluation Metrics")
    print("=" * 80)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("=" * 80)
    
    # Confusion matrix
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    # Handle case where only one class is present
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        print(f"Only one class present: {'STOP' if unique_classes[0] == 1 else 'GO'}")
    else:
        print(classification_report(y_true, y_pred, target_names=['GO', 'STOP'], labels=[0, 1]))
    
    # Plot curves
    print("\nGenerating plots...")
    plot_roc_curve(y_true, y_pred_proba, output_dir / 'roc_curve.png')
    plot_precision_recall_curve(y_true, y_pred_proba, output_dir / 'precision_recall_curve.png')
    plot_calibration_curve(y_true, y_pred_proba, output_dir / 'calibration_curve.png')
    
    # Save metrics
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(sequences),
        'model_config': model_config
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    print(f"  - Metrics: {results_path}")
    print(f"  - ROC curve: {output_dir / 'roc_curve.png'}")
    print(f"  - Precision-Recall curve: {output_dir / 'precision_recall_curve.png'}")
    print(f"  - Calibration curve: {output_dir / 'calibration_curve.png'}")
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Dilemma Zone Prediction Model')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='Path to CSV file for evaluation (optional if using pre-built sequences)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint_path,
        csv_path=args.csv_path,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()


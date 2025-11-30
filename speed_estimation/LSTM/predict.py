"""
Prediction Script for Dilemma Zone Model

Use trained model to make predictions on new vehicle sequences.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import json

from .evaluate_model import load_model
from .sequence_builder import build_sequences_from_dataframe
from .utils import load_csv_data
from .config import SEQUENCE_LENGTH, FEATURE_COLUMNS


def predict_single_sequence(
    model,
    sequence: np.ndarray,
    device: torch.device,
    threshold: float = 0.5
) -> dict:
    """
    Predict STOP/GO for a single sequence.
    
    Args:
        model: Trained model
        sequence: Input sequence of shape (sequence_length, feature_dim)
        device: Device to run on
        threshold: Probability threshold for classification
        
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    
    # Convert to tensor
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob_stop = model(sequence_tensor).cpu().item()
    
    prediction = "STOP" if prob_stop >= threshold else "GO"
    confidence = prob_stop if prob_stop >= threshold else (1 - prob_stop)
    
    return {
        'prediction': prediction,
        'probability_stop': float(prob_stop),
        'probability_go': float(1 - prob_stop),
        'confidence': float(confidence),
        'threshold': threshold
    }


def predict_from_csv(
    checkpoint_path: str,
    csv_path: str,
    output_path: str = None,
    threshold: float = 0.5
):
    """
    Make predictions on sequences from CSV file.
    
    Args:
        checkpoint_path: Path to model checkpoint
        csv_path: Path to CSV file with vehicle data
        output_path: Path to save predictions (JSON)
        threshold: Probability threshold for classification
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model, normalizer_params, model_config = load_model(checkpoint_path, device)
    print(f"Model type: {model_config.get('model_type', 'unknown')}")
    
    # Load data and build sequences
    print(f"Loading data from: {csv_path}")
    df = load_csv_data(csv_path)
    
    print("Building sequences...")
    sequences, labels, _ = build_sequences_from_dataframe(
        df,
        sequence_length=SEQUENCE_LENGTH,
        normalize=True,
        fit_normalizer=False,
        normalizer_params=normalizer_params
    )
    
    print(f"Making predictions on {len(sequences)} sequences...")
    
    # Make predictions
    predictions = []
    for i, sequence in enumerate(sequences):
        result = predict_single_sequence(model, sequence, device, threshold)
        result['sequence_id'] = i
        result['true_label'] = 'STOP' if labels[i] == 1 else 'GO'
        result['correct'] = (result['prediction'] == result['true_label'])
        predictions.append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Prediction Summary")
    print("=" * 80)
    
    total = len(predictions)
    correct = sum(1 for p in predictions if p['correct'])
    stop_pred = sum(1 for p in predictions if p['prediction'] == 'STOP')
    go_pred = sum(1 for p in predictions if p['prediction'] == 'GO')
    
    print(f"Total sequences: {total}")
    print(f"Correct predictions: {correct} ({100*correct/total:.1f}%)")
    print(f"STOP predictions: {stop_pred}")
    print(f"GO predictions: {go_pred}")
    print("\nDetailed Predictions:")
    print("-" * 80)
    
    for pred in predictions:
        print(f"Sequence {pred['sequence_id']}: {pred['prediction']} "
              f"(P={pred['probability_stop']:.3f}, True: {pred['true_label']}, "
              f"{'✓' if pred['correct'] else '✗'})")
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'model_checkpoint': checkpoint_path,
            'csv_path': csv_path,
            'threshold': threshold,
            'total_sequences': total,
            'accuracy': correct / total,
            'predictions': predictions
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nPredictions saved to: {output_path}")
    
    return predictions


def predict_from_features(
    checkpoint_path: str,
    speed_ms: float,
    distance_to_stop_line: float,
    ttc: float = None,
    distance_to_front_vehicle: float = 10.0,
    traffic_density: float = 5.0,
    class_id: int = 2,
    threshold: float = 0.5
):
    """
    Make prediction from individual feature values.
    
    Args:
        checkpoint_path: Path to model checkpoint
        speed_ms: Speed in m/s
        distance_to_stop_line: Distance to stop line in meters
        ttc: Time to collision (optional, calculated if None)
        distance_to_front_vehicle: Distance to front vehicle
        traffic_density: Traffic density
        class_id: Vehicle class ID (2=car, 3=motorcycle, 5=bus, 7=truck)
        threshold: Probability threshold for classification
    """
    from .dilemma_zone_generator import create_synthetic_sequence
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, normalizer_params, model_config = load_model(checkpoint_path, device)
    
    # Create synthetic sequence
    sequence = create_synthetic_sequence(
        speed_ms=speed_ms,
        distance_to_stop_line=distance_to_stop_line,
        ttc=ttc,
        distance_to_front_vehicle=distance_to_front_vehicle,
        traffic_density=traffic_density,
        class_id=class_id,
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Normalize if needed
    if normalizer_params:
        from .utils import normalize_features
        if normalizer_params.get('method') == 'standard':
            sequence_flat = sequence.reshape(-1, FEATURE_DIM)
            normalized_flat, _ = normalize_features(
                sequence_flat,
                fit=False,
                mean=normalizer_params.get('mean'),
                std=normalizer_params.get('std')
            )
            sequence = normalized_flat.reshape(SEQUENCE_LENGTH, FEATURE_DIM)
        elif normalizer_params.get('method') == 'minmax':
            sequence_flat = sequence.reshape(-1, FEATURE_DIM)
            normalized_flat, _ = normalize_features(
                sequence_flat,
                fit=False,
                min_val=normalizer_params.get('min'),
                max_val=normalizer_params.get('max')
            )
            sequence = normalized_flat.reshape(SEQUENCE_LENGTH, FEATURE_DIM)
    
    # Make prediction
    result = predict_single_sequence(model, sequence, device, threshold)
    
    print("\n" + "=" * 80)
    print("Prediction Result")
    print("=" * 80)
    print(f"Input Features:")
    print(f"  Speed: {speed_ms} m/s")
    print(f"  Distance to Stop Line: {distance_to_stop_line} m")
    print(f"  TTC: {ttc if ttc else distance_to_stop_line/speed_ms if speed_ms > 0 else 0:.2f} s")
    print(f"  Distance to Front Vehicle: {distance_to_front_vehicle} m")
    print(f"  Traffic Density: {traffic_density}")
    print(f"  Vehicle Class: {class_id}")
    print(f"\nPrediction: {result['prediction']}")
    print(f"Probability (STOP): {result['probability_stop']:.3f}")
    print(f"Probability (GO): {result['probability_go']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("=" * 80)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Make Predictions with Trained Model')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['csv', 'features'],
        default='csv',
        help='Prediction mode: csv (from CSV file) or features (from individual values)'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='Path to CSV file (required for csv mode)'
    )
    parser.add_argument(
        '--speed_ms',
        type=float,
        default=None,
        help='Speed in m/s (required for features mode)'
    )
    parser.add_argument(
        '--distance_to_stop_line',
        type=float,
        default=None,
        help='Distance to stop line in meters (required for features mode)'
    )
    parser.add_argument(
        '--ttc',
        type=float,
        default=None,
        help='Time to collision in seconds (optional, calculated if not provided)'
    )
    parser.add_argument(
        '--distance_to_front_vehicle',
        type=float,
        default=10.0,
        help='Distance to front vehicle in meters'
    )
    parser.add_argument(
        '--traffic_density',
        type=float,
        default=5.0,
        help='Traffic density'
    )
    parser.add_argument(
        '--class_id',
        type=int,
        default=2,
        choices=[2, 3, 5, 7],
        help='Vehicle class ID (2=car, 3=motorcycle, 5=bus, 7=truck)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for classification'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save predictions (JSON file)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'csv':
        if not args.csv_path:
            parser.error("--csv_path is required for csv mode")
        predict_from_csv(
            args.checkpoint_path,
            args.csv_path,
            args.output_path,
            args.threshold
        )
    else:  # features mode
        if args.speed_ms is None or args.distance_to_stop_line is None:
            parser.error("--speed_ms and --distance_to_stop_line are required for features mode")
        predict_from_features(
            args.checkpoint_path,
            args.speed_ms,
            args.distance_to_stop_line,
            args.ttc,
            args.distance_to_front_vehicle,
            args.traffic_density,
            args.class_id,
            args.threshold
        )


if __name__ == '__main__':
    main()


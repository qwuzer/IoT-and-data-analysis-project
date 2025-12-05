"""
Dynamic Dilemma Zone Generator

Generates dilemma zone heatmaps and contour plots for different vehicle types.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from .config import (
    DZ_SPEED_RANGE,
    DZ_DISTANCE_RANGE,
    DZ_GRID_RESOLUTION,
    DZ_CONTOUR_LEVELS,
    DZ_BOUNDARY,
    DZ_DIR,
    VEHICLE_TYPES,
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


def create_synthetic_sequence(
    speed_ms: float,
    distance_to_stop_line: float,
    ttc: Optional[float] = None,
    distance_to_front_vehicle: float = 10.0,
    traffic_density: float = 5.0,
    class_id: int = 2,  # Default to car
    sequence_length: int = SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Create a synthetic input sequence for a given speed and distance.
    
    Args:
        speed_ms: Speed in m/s
        distance_to_stop_line: Distance to stop line in meters
        ttc: Time to collision (if None, calculated from speed and distance)
        distance_to_front_vehicle: Distance to front vehicle
        traffic_density: Traffic density
        class_id: Vehicle class ID
        sequence_length: Length of sequence
        
    Returns:
        Synthetic sequence of shape (sequence_length, feature_dim)
    """
    # Calculate TTC if not provided
    if ttc is None and speed_ms > 0:
        ttc = distance_to_stop_line / speed_ms if speed_ms > 0 else 0.0
    elif ttc is None:
        ttc = 0.0
    
    # Create feature vector
    features = np.array([
        speed_ms,
        distance_to_stop_line,
        ttc,
        distance_to_front_vehicle,
        traffic_density,
        class_id
    ], dtype=np.float32)
    
    # Create sequence by varying speed and distance slightly over time
    # (simulating approach to stop line)
    sequence = np.zeros((sequence_length, FEATURE_DIM), dtype=np.float32)
    
    for t in range(sequence_length):
        # Gradually decrease distance (approaching stop line)
        time_factor = t / sequence_length
        current_distance = distance_to_stop_line * (1 - time_factor * 0.1)  # Small decrease
        
        # Speed might decrease slightly (deceleration)
        current_speed = speed_ms * (1 - time_factor * 0.05)  # Small deceleration
        
        # Recalculate TTC
        current_ttc = current_distance / current_speed if current_speed > 0 else 0.0
        
        sequence[t] = np.array([
            current_speed,
            current_distance,
            current_ttc,
            distance_to_front_vehicle,
            traffic_density,
            class_id
        ], dtype=np.float32)
    
    return sequence


def generate_dilemma_zone_grid(
    model: DilemmaZoneModel,
    speed_range: Tuple[float, float] = DZ_SPEED_RANGE,
    distance_range: Tuple[float, float] = DZ_DISTANCE_RANGE,
    grid_resolution: int = DZ_GRID_RESOLUTION,
    vehicle_class_id: int = 2,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dilemma zone grid by predicting P(stop) for each grid point.
    
    Args:
        model: Trained model
        speed_range: (min, max) speed range in m/s
        distance_range: (min, max) distance range in meters
        grid_resolution: Number of grid points per dimension
        vehicle_class_id: Vehicle class ID for synthetic sequences
        device: Device to run on
        
    Returns:
        Tuple of (speed_grid, distance_grid, probability_grid)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create grid
    speed_values = np.linspace(speed_range[0], speed_range[1], grid_resolution)
    distance_values = np.linspace(distance_range[0], distance_range[1], grid_resolution)
    
    speed_grid, distance_grid = np.meshgrid(speed_values, distance_values)
    probability_grid = np.zeros_like(speed_grid)
    
    print(f"Generating dilemma zone grid ({grid_resolution}x{grid_resolution})...")
    
    model.eval()
    with torch.no_grad():
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                speed = speed_grid[i, j]
                distance = distance_grid[i, j]
                
                # Create synthetic sequence
                sequence = create_synthetic_sequence(
                    speed_ms=speed,
                    distance_to_stop_line=distance,
                    class_id=vehicle_class_id
                )
                
                # Predict
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                prob = model(sequence_tensor).cpu().item()
                probability_grid[i, j] = prob
    
    return speed_grid, distance_grid, probability_grid


def plot_dilemma_zone_heatmap(
    speed_grid: np.ndarray,
    distance_grid: np.ndarray,
    probability_grid: np.ndarray,
    contour_levels: List[float] = DZ_CONTOUR_LEVELS,
    dz_boundary: Tuple[float, float] = DZ_BOUNDARY,
    vehicle_type: str = "car",
    save_path: Path = None
):
    """
    Plot dilemma zone heatmap with speed (x-axis) and distance (y-axis).
    
    Args:
        speed_grid: Speed grid (x-axis)
        distance_grid: Distance grid (y-axis)
        probability_grid: Probability grid (P(stop))
        contour_levels: Probability levels for contour lines
        dz_boundary: Dilemma zone boundary (min, max) probability
        vehicle_type: Vehicle type name
        save_path: Path to save plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    # Extract unique values for proper heatmap
    speed_values = np.unique(speed_grid)
    distance_values = np.unique(distance_grid)
    
    # Create proper heatmap using pcolormesh (better for grid data)
    # Note: probability_grid is indexed as [distance_idx, speed_idx]
    # We need to transpose it so speed is x-axis and distance is y-axis
    im = plt.pcolormesh(
        speed_grid, distance_grid, probability_grid,
        cmap='RdYlGn_r', vmin=0, vmax=1, shading='auto'
    )
    cbar = plt.colorbar(im, label='P(STOP)', aspect=30)
    cbar.set_label('P(STOP)', fontsize=12, rotation=270, labelpad=20)
    
    # Add contour lines for probability levels
    contours = plt.contour(
        speed_grid, distance_grid, probability_grid,
        levels=contour_levels, colors='black', linewidths=1.5, linestyles='--', alpha=0.7
    )
    plt.clabel(contours, inline=True, fontsize=9, fmt='%.2f', colors='black')
    
    # Highlight dilemma zone boundary (P ∈ [0.45, 0.55])
    dz_contour = plt.contour(
        speed_grid, distance_grid, probability_grid,
        levels=[dz_boundary[0], dz_boundary[1]],
        colors='red', linewidths=2.5, linestyles='-'
    )
    # Note: fontweight not supported in all matplotlib versions, using fontsize for emphasis
    plt.clabel(dz_contour, inline=True, fontsize=12, fmt='DZ: %.2f', colors='red')
    
    plt.xlabel('Speed (m/s)', fontsize=12, fontweight='bold')
    plt.ylabel('Distance to Stop Line (m)', fontsize=12, fontweight='bold')
    plt.title(f'Dynamic Dilemma Zone Heatmap: {vehicle_type.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Dilemma zone heatmap saved to: {save_path}")
    else:
        plt.show()


def plot_training_history_per_vehicle(
    train_losses: List[float] = None,
    val_losses: List[float] = None,
    vehicle_type: str = "all",
    save_path: Path = None
):
    """
    Plot training history (loss curves) for visualization.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        vehicle_type: Vehicle type name (for title)
        save_path: Path to save plot
    """
    if train_losses is None or val_losses is None:
        print("Warning: Training history not available, skipping plot.")
        return
    
    plt.figure(figsize=FIGURE_SIZE)
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
    plt.title(f'Training History: {vehicle_type.upper()}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def generate_dilemma_zones_for_all_vehicle_types(
    model: DilemmaZoneModel,
    speed_range: Tuple[float, float] = DZ_SPEED_RANGE,
    distance_range: Tuple[float, float] = DZ_DISTANCE_RANGE,
    grid_resolution: int = DZ_GRID_RESOLUTION,
    output_dir: Path = None,
    device: torch.device = None,
    train_losses: List[float] = None,
    val_losses: List[float] = None
):
    """
    Generate dilemma zone maps for all vehicle types with optional training history.
    
    Args:
        model: Trained model
        speed_range: Speed range
        distance_range: Distance range
        grid_resolution: Grid resolution
        output_dir: Output directory
        device: Device to run on
        train_losses: Training losses for history plot (optional)
        val_losses: Validation losses for history plot (optional)
    """
    if output_dir is None:
        output_dir = DZ_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = next(model.parameters()).device
    
    print("Generating dilemma zones for all vehicle types...")
    
    # Plot training history if available
    if train_losses and val_losses:
        history_path = output_dir / 'training_history_all_vehicles.png'
        plot_training_history_per_vehicle(
            train_losses, val_losses, vehicle_type="all", save_path=history_path
        )
    
    for class_id, vehicle_type in VEHICLE_TYPES.items():
        print(f"\nProcessing {vehicle_type} (class_id={class_id})...")
        
        # Generate grid
        speed_grid, distance_grid, probability_grid = generate_dilemma_zone_grid(
            model, speed_range, distance_range, grid_resolution,
            vehicle_class_id=class_id, device=device
        )
        
        # Plot and save heatmap
        save_path = output_dir / f'dilemma_zone_{vehicle_type}.png'
        plot_dilemma_zone_heatmap(
            speed_grid, distance_grid, probability_grid,
            vehicle_type=vehicle_type,
            save_path=save_path
        )
        
        # Plot training history per vehicle if available
        if train_losses and val_losses:
            history_path = output_dir / f'training_history_{vehicle_type}.png'
            plot_training_history_per_vehicle(
                train_losses, val_losses, vehicle_type=vehicle_type, save_path=history_path
            )
    
    print(f"\nAll dilemma zone maps saved to: {output_dir}")


def main():
    """
    Command-line interface for dilemma zone generation.
    """
    import argparse
    from .evaluate_model import load_model
    
    parser = argparse.ArgumentParser(description='Generate Dynamic Dilemma Zone Maps')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for dilemma zone maps'
    )
    parser.add_argument(
        '--vehicle_type',
        type=str,
        default=None,
        choices=['car', 'truck', 'bus', 'motorcycle', 'all'],
        help='Vehicle type to generate DZ for (default: all)'
    )
    parser.add_argument(
        '--grid_resolution',
        type=int,
        default=DZ_GRID_RESOLUTION,
        help='Grid resolution (default: 50)'
    )
    parser.add_argument(
        '--speed_range',
        type=float,
        nargs=2,
        default=None,
        metavar=('MIN', 'MAX'),
        help='Speed range in m/s (default: 0 25). Example: --speed_range 0 30'
    )
    parser.add_argument(
        '--distance_range',
        type=float,
        nargs=2,
        default=None,
        metavar=('MIN', 'MAX'),
        help='Distance range in meters (default: 0 60). Example: --distance_range 0 80'
    )
    
    args = parser.parse_args()
    
    # Set speed and distance ranges
    speed_range = tuple(args.speed_range) if args.speed_range else DZ_SPEED_RANGE
    distance_range = tuple(args.distance_range) if args.distance_range else DZ_DISTANCE_RANGE
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, _, _ = load_model(args.checkpoint_path, device)
    
    # Try to load training history if available
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
            if train_losses and val_losses:
                print(f"Loaded training history: {len(train_losses)} epochs")
    
    # Generate dilemma zones
    if args.vehicle_type is None or args.vehicle_type == 'all':
        generate_dilemma_zones_for_all_vehicle_types(
            model, 
            speed_range=speed_range,
            distance_range=distance_range,
            output_dir=args.output_dir, 
            device=device,
            train_losses=train_losses, 
            val_losses=val_losses
        )
    else:
        # Generate for specific vehicle type
        class_id = None
        for cid, vtype in VEHICLE_TYPES.items():
            if vtype == args.vehicle_type:
                class_id = cid
                break
        
        if class_id is None:
            raise ValueError(f"Unknown vehicle type: {args.vehicle_type}")
        
        speed_grid, distance_grid, probability_grid = generate_dilemma_zone_grid(
            model, 
            speed_range=speed_range,
            distance_range=distance_range,
            vehicle_class_id=class_id, 
            device=device,
            grid_resolution=args.grid_resolution
        )
        
        output_dir = Path(args.output_dir) if args.output_dir else DZ_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / f'dilemma_zone_{args.vehicle_type}.png'
        plot_dilemma_zone_heatmap(
            speed_grid, distance_grid, probability_grid,
            vehicle_type=args.vehicle_type,
            save_path=save_path
        )


if __name__ == '__main__':
    main()


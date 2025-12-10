# Dilemma Zone Generation Guide

This guide explains how to generate dynamic dilemma zone heatmaps using the `dilemma_zone_generator.py` script. These visualizations show the probability of a vehicle stopping (P(STOP)) across different combinations of speed and distance to the stop line.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Required Files](#required-files)
4. [Understanding the Metadata](#understanding-the-metadata)
5. [Basic Usage](#basic-usage)
6. [Command-Line Options](#command-line-options)
7. [Examples](#examples)
8. [Output Files](#output-files)
9. [Understanding the Visualizations](#understanding-the-visualizations)
10. [Troubleshooting](#troubleshooting)

## Overview

The dilemma zone generator creates 2D heatmaps that visualize the model's predictions across a grid of speed and distance values. Each point on the grid represents a synthetic scenario where:

- **X-axis**: Speed (m/s)
- **Y-axis**: Distance to stop line (meters)
- **Color intensity**: Probability of stopping (P(STOP))

The script generates separate heatmaps for different vehicle types (car, truck, bus, motorcycle) since each vehicle type has different stopping behaviors.

## Prerequisites

### Software Requirements

- Python 3.7+
- PyTorch (CPU or GPU)
- Required Python packages (see `requirements.txt`):
  - `torch`
  - `numpy`
  - `matplotlib`
  - `seaborn`

### Required Files

1. **Model Checkpoint** (`.pt` file): A trained model checkpoint containing:

   - Model architecture and weights
   - Normalization parameters
   - Model configuration

2. **Training Metadata** (optional but recommended): `training_metadata.json` file in the same directory as the checkpoint, containing:
   - Training and validation loss history
   - Training statistics
   - Model configuration

## Required Files

### Model Checkpoint Structure

The checkpoint file should be a PyTorch `.pt` file containing:

```python
{
    'model_state_dict': {...},      # Model weights
    'normalizer_params': {...},     # Feature normalization parameters
    'model_config': {...}           # Model architecture configuration
}
```

### Training Metadata Structure

The `training_metadata.json` file (located in the same directory as the checkpoint) should contain:

```json
{
    "train_losses": [...],          # List of training losses per epoch
    "val_losses": [...],            # List of validation losses per epoch
    "best_val_loss": 0.0067,        # Best validation loss achieved
    "num_epochs_trained": 38,       # Number of epochs trained
    "train_stats": {...},            # Training dataset statistics
    "val_stats": {...},             # Validation dataset statistics
    "normalizer_params": {...},      # Normalization parameters
    "model_config": {...}            # Model configuration
}
```

### Example Metadata File

See `speed_estimation/LSTM/models/checkpoints/training_metadata.json` for a complete example. Key fields include:

- **train_losses**: Array of training losses (one per epoch)
- **val_losses**: Array of validation losses (one per epoch)
- **train_stats**: Dataset statistics including:
  - `total_sequences`: Number of training sequences
  - `stop_count` / `go_count`: Class distribution
  - `stop_ratio` / `go_ratio`: Class ratios
- **normalizer_params**: Mean and standard deviation for feature normalization
- **model_config**: Model architecture parameters

## Understanding the Metadata

The training metadata provides important context for the generated visualizations:

### Loss History

The training and validation loss arrays are used to generate training history plots:

- **Training Loss**: Shows how well the model fits the training data
- **Validation Loss**: Shows generalization performance
- **Best Validation Loss**: The lowest validation loss achieved (indicates best model performance)

### Dataset Statistics

Understanding the training data helps interpret the dilemma zones:

- **Class Distribution**: If the dataset is imbalanced (e.g., 87% STOP, 13% GO), the model may be biased
- **Sequence Count**: Small datasets may lead to less reliable predictions

### Normalization Parameters

These are critical for generating correct synthetic sequences:

- **Mean**: Average values for each feature
- **Std**: Standard deviation for each feature
- The script uses these to create properly normalized synthetic sequences

## Basic Usage

### Generate Dilemma Zones for All Vehicle Types

```bash
python -m speed_estimation.LSTM.dilemma_zone_generator \
    --checkpoint_path speed_estimation/LSTM/models/checkpoints/best_model.pt
```

This will:

1. Load the model from the checkpoint
2. Automatically detect and load `training_metadata.json` if available
3. Generate dilemma zone heatmaps for all vehicle types (car, truck, bus, motorcycle)
4. Save outputs to `speed_estimation/LSTM/outputs/dilemma_zones/`

### Generate for a Specific Vehicle Type

```bash
python -m speed_estimation.LSTM.dilemma_zone_generator \
    --checkpoint_path speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --vehicle_type car
```

Available vehicle types: `car`, `truck`, `bus`, `motorcycle`, `all` (default)

## Command-Line Options

### Required Arguments

- `--checkpoint_path`: Path to the model checkpoint file (`.pt`)

### Optional Arguments

- `--output_dir`: Output directory for dilemma zone images (default: `LSTM/outputs/dilemma_zones/`)
- `--vehicle_type`: Vehicle type to generate DZ for (`car`, `truck`, `bus`, `motorcycle`, `all`)
- `--grid_resolution`: Number of grid points per dimension (default: 50)
  - Higher values = more detailed but slower generation
  - Recommended: 50-100
- `--speed_range MIN MAX`: Speed range in m/s (default: 0 25)
  - Example: `--speed_range 0 30`
- `--distance_range MIN MAX`: Distance range in meters (default: 0 60)
  - Example: `--distance_range 0 80`

### Full Command Example

```bash
python -m speed_estimation.LSTM.dilemma_zone_generator \
    --checkpoint_path speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --output_dir custom_output/dilemma_zones \
    --vehicle_type all \
    --grid_resolution 75 \
    --speed_range 0 30 \
    --distance_range 0 80
```

## Examples

### Example 1: Quick Generation (Default Settings)

Generate dilemma zones for all vehicle types with default settings:

```bash
python -m speed_estimation.LSTM.dilemma_zone_generator \
    --checkpoint_path speed_estimation/LSTM/models/checkpoints/best_model.pt
```

**Output**: 4 heatmap images (one per vehicle type) + training history plots

### Example 2: High-Resolution for Cars Only

Generate a high-resolution dilemma zone specifically for cars:

```bash
python -m speed_estimation.LSTM.dilemma_zone_generator \
    --checkpoint_path speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --vehicle_type car \
    --grid_resolution 100 \
    --speed_range 0 30 \
    --distance_range 0 100
```

**Output**: Single high-resolution heatmap for cars

### Example 3: Custom Output Directory

Save outputs to a custom location:

```bash
python -m speed_estimation.LSTM.dilemma_zone_generator \
    --checkpoint_path speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --output_dir results/dilemma_analysis
```

**Output**: All files saved to `results/dilemma_analysis/`

### Example 4: Fast Preview (Low Resolution)

Quick preview with lower resolution for faster generation:

```bash
python -m speed_estimation.LSTM.dilemma_zone_generator \
    --checkpoint_path speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --grid_resolution 25
```

**Note**: Lower resolution = faster but less detailed visualizations

## Output Files

### Dilemma Zone Heatmaps

For each vehicle type, a heatmap image is generated:

- `dilemma_zone_car.png`
- `dilemma_zone_truck.png`
- `dilemma_zone_bus.png`
- `dilemma_zone_motorcycle.png`

### Training History Plots (if metadata available)

- `training_history_all_vehicles.png`: Combined training history
- `training_history_car.png`: Training history for car-specific model (if applicable)
- `training_history_truck.png`: Training history for truck-specific model
- `training_history_bus.png`: Training history for bus-specific model
- `training_history_motorcycle.png`: Training history for motorcycle-specific model

### Default Output Location

All files are saved to: `speed_estimation/LSTM/outputs/dilemma_zones/`

## Understanding the Visualizations

### Heatmap Interpretation

1. **Color Scale**:

   - **Red/Orange**: High probability of stopping (P(STOP) > 0.7)
   - **Yellow**: Medium probability (P(STOP) ≈ 0.5) - **Dilemma Zone**
   - **Green**: Low probability of stopping (P(STOP) < 0.3) - likely to go

2. **Contour Lines**:

   - **Black dashed lines**: Probability levels (0.3, 0.5, 0.7)
   - **Red solid lines**: Dilemma zone boundary (P ∈ [0.45, 0.55])

3. **Dilemma Zone**:
   - Defined as the region where P(STOP) ∈ [0.45, 0.55]
   - Vehicles in this zone have high uncertainty about stopping vs. going
   - Critical for traffic safety analysis

### Reading the Heatmap

- **Top-left (high speed, far distance)**: Usually low P(STOP) - vehicle likely to go
- **Bottom-right (low speed, close distance)**: Usually high P(STOP) - vehicle likely to stop
- **Middle region**: Dilemma zone - uncertain behavior

### Training History Plots

If `training_metadata.json` is available, training history plots show:

- **Training Loss (blue)**: How well the model fits training data
- **Validation Loss (red)**: Generalization performance
- **Convergence**: Both losses should decrease and stabilize
- **Overfitting**: If validation loss increases while training loss decreases

## Troubleshooting

### Issue: "FileNotFoundError: checkpoint file not found"

**Solution**: Ensure the checkpoint path is correct:

```bash
# Use absolute path or relative path from project root
--checkpoint_path speed_estimation/LSTM/models/checkpoints/best_model.pt
```

### Issue: "CUDA out of memory"

**Solution**:

- Use CPU instead: The script automatically uses CPU if CUDA is unavailable
- Reduce grid resolution: `--grid_resolution 25` instead of 50
- Process one vehicle type at a time: `--vehicle_type car`

### Issue: "Training history not available"

**Solution**: This is a warning, not an error. The script will still generate dilemma zones. To include training history:

1. Ensure `training_metadata.json` exists in the same directory as the checkpoint
2. The metadata file should contain `train_losses` and `val_losses` arrays

### Issue: Generated images look incorrect

**Possible causes**:

1. **Wrong normalization**: Ensure the checkpoint contains correct `normalizer_params`
2. **Model mismatch**: Ensure the checkpoint matches the expected model architecture
3. **Feature order**: The model expects features in this order:
   - `speed_ms`
   - `distance_to_stop_line`
   - `ttc` (time to collision)
   - `distance_to_front_vehicle`
   - `traffic_density`
   - `class_id`

### Issue: Slow generation

**Solutions**:

- Reduce grid resolution: `--grid_resolution 30`
- Use GPU if available (automatically detected)
- Generate one vehicle type at a time
- The generation time is O(grid_resolution²), so halving resolution = 4x faster

### Issue: "Unknown vehicle type"

**Solution**: Use one of the supported vehicle types:

- `car` (class_id: 2)
- `motorcycle` (class_id: 3)
- `bus` (class_id: 5)
- `truck` (class_id: 7)
- `all` (generates for all types)

## Advanced Usage

### Programmatic Usage

You can also use the generator programmatically:

```python
from pathlib import Path
import torch
from speed_estimation.LSTM.dilemma_zone_generator import (
    generate_dilemma_zones_for_all_vehicle_types
)
from speed_estimation.LSTM.evaluate_model import load_model

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _, _ = load_model('path/to/checkpoint.pt', device)

# Load training metadata
import json
with open('path/to/training_metadata.json', 'r') as f:
    metadata = json.load(f)

# Generate dilemma zones
generate_dilemma_zones_for_all_vehicle_types(
    model=model,
    output_dir=Path('custom/output'),
    device=device,
    train_losses=metadata['train_losses'],
    val_losses=metadata['val_losses'],
    grid_resolution=75,
    speed_range=(0, 30),
    distance_range=(0, 80)
)
```

### Custom Synthetic Sequence Generation

The script uses `create_synthetic_sequence()` to generate input sequences. You can customize this function to:

- Add noise to simulate real-world variability
- Include temporal dynamics (acceleration/deceleration patterns)
- Incorporate additional features

## Configuration

Default configuration values are defined in `config.py`:

- `DZ_SPEED_RANGE = (0, 25)` m/s
- `DZ_DISTANCE_RANGE = (0, 60)` meters
- `DZ_GRID_RESOLUTION = 50` points per dimension
- `DZ_CONTOUR_LEVELS = [0.3, 0.5, 0.7]` probability levels
- `DZ_BOUNDARY = (0.45, 0.55)` dilemma zone definition
- `VEHICLE_TYPES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}`

To modify defaults, edit `speed_estimation/LSTM/config.py` or override via command-line arguments.

## Summary

The dilemma zone generator is a powerful tool for visualizing model behavior across different scenarios. Key points:

1. **Required**: Model checkpoint (`.pt` file)
2. **Recommended**: Training metadata (`training_metadata.json`) for complete visualizations
3. **Output**: Heatmaps showing P(STOP) across speed/distance combinations
4. **Vehicle-specific**: Separate visualizations for different vehicle types
5. **Customizable**: Grid resolution, ranges, and output location

For questions or issues, refer to the main project documentation or check the code comments in `dilemma_zone_generator.py`.

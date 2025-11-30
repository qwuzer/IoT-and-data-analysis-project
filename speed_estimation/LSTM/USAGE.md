# Dilemma Zone Model Usage Guide

## Quick Start

### 1. Make Predictions

#### Predict from individual feature values:
```bash
python -m LSTM.predict \
  --checkpoint_path LSTM/models/checkpoints/best_model.pt \
  --mode features \
  --speed_ms 10.0 \
  --distance_to_stop_line 30.0 \
  --class_id 2
```

#### Predict from CSV file:
```bash
python -m LSTM.predict \
  --checkpoint_path LSTM/models/checkpoints/best_model.pt \
  --mode csv \
  --csv_path logs/vehicles_speed_log.csv \
  --output_path LSTM/outputs/predictions.json
```

### 2. Generate SHAP Explainability Analysis

```bash
python -m LSTM.explainability \
  --checkpoint_path LSTM/models/checkpoints/best_model.pt \
  --csv_path logs/vehicles_speed_log.csv \
  --output_dir LSTM/outputs/shap_analysis
```

**Output location:** `LSTM/outputs/shap_analysis/`
- `feature_importance.json` - Feature importance scores
- `shap_summary.png` - SHAP summary plot
- `temporal_attribution.png` - Temporal feature attribution heatmap
- `shap_force_plot_sample_*.png` - Individual prediction explanations
- `partial_dependence_*.png` - Partial dependence plots for key features

### 3. Evaluate Model

```bash
python -m LSTM.evaluate_model \
  --checkpoint_path LSTM/models/checkpoints/best_model.pt \
  --csv_path logs/vehicles_speed_log.csv
```

**Output location:** `LSTM/outputs/evaluation/`
- `evaluation_results.json` - Metrics (AUC, F1, Accuracy, etc.)
- `roc_curve.png` - ROC curve
- `precision_recall_curve.png` - Precision-Recall curve
- `calibration_curve.png` - Calibration curve

### 4. Generate Visualizations

```bash
python -m LSTM.visualization \
  --checkpoint_path LSTM/models/checkpoints/best_model.pt \
  --csv_path logs/vehicles_speed_log.csv
```

**Output location:** `LSTM/outputs/visualizations/`
- `training_history.png` - Training loss curves
- `prediction_distribution.png` - Prediction distributions by class
- `feature_distributions.png` - Feature value distributions
- `correlation_matrix.png` - Feature correlation matrix
- `speed_vs_distance.png` - Speed vs distance scatter plot
- `temporal_evolution_*.png` - Temporal feature evolution plots

### 5. Generate Dilemma Zone Maps

```bash
python -m LSTM.dilemma_zone_generator \
  --checkpoint_path LSTM/models/checkpoints/best_model.pt \
  --grid_resolution 50
```

**Output location:** `LSTM/outputs/dilemma_zones/`
- `dilemma_zone_car.png` - Dilemma zone for cars
- `dilemma_zone_motorcycle.png` - Dilemma zone for motorcycles
- `dilemma_zone_bus.png` - Dilemma zone for buses
- `dilemma_zone_truck.png` - Dilemma zone for trucks

## Prediction Parameters

### Feature Mode Parameters:
- `--speed_ms`: Speed in m/s (required)
- `--distance_to_stop_line`: Distance to stop line in meters (required)
- `--ttc`: Time to collision in seconds (optional, auto-calculated)
- `--distance_to_front_vehicle`: Distance to front vehicle (default: 10.0)
- `--traffic_density`: Traffic density (default: 5.0)
- `--class_id`: Vehicle class (2=car, 3=motorcycle, 5=bus, 7=truck, default: 2)
- `--threshold`: Probability threshold for classification (default: 0.5)

## Example Predictions

### Example 1: Fast car approaching stop line
```bash
python -m LSTM.predict \
  --checkpoint_path LSTM/models/checkpoints/best_model.pt \
  --mode features \
  --speed_ms 15.0 \
  --distance_to_stop_line 20.0 \
  --class_id 2
```

### Example 2: Slow motorcycle with more distance
```bash
python -m LSTM.predict \
  --checkpoint_path LSTM/models/checkpoints/best_model.pt \
  --mode features \
  --speed_ms 8.0 \
  --distance_to_stop_line 40.0 \
  --class_id 3
```

## Output Locations Summary

All outputs are saved in `LSTM/outputs/`:
- `evaluation/` - Model evaluation metrics and curves
- `visualizations/` - Training and data analysis plots
- `dilemma_zones/` - Dynamic dilemma zone heatmaps
- `shap_analysis/` - SHAP explainability reports
- `predictions.json` - Prediction results (when using CSV mode)

## Notes

- The model requires normalized features. The prediction script handles normalization automatically.
- For SHAP analysis, ensure you have enough data (at least 10-20 sequences recommended).
- Dilemma zone generation can take a few minutes depending on grid resolution.
- All scripts support both CPU and GPU (automatically detected).


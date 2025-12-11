# CSV Animation Tool

## Overview

The `animate_csv.py` script creates an animated visualization that scrolls through vehicle speed log CSV files, showing:
- Vehicle positions (x, y coordinates) as colored circles
- Vehicle speeds in real-time
- Traffic light status
- STOP/GO decisions
- Vehicle movement trajectories (optional)

## Features

- **Color-coded vehicles**: Each vehicle type has a distinct color
  - Red: Car
  - Green: Truck
  - Blue: Bus
  - Yellow: Motorcycle

- **Trajectory trails**: Shows the path each vehicle has taken (last N frames)

- **Real-time information**:
  - Frame number
  - Number of vehicles
  - Traffic density
  - Traffic light status (RED/YELLOW/GREEN)
  - Yellow light warning indicator

- **Vehicle labels**: Shows vehicle ID, type, speed, and decision (STOP/GO)

## Usage

### Basic Usage (Display Only)

```bash
python speed_estimation/animate_csv.py \
    --csv_path speed_estimation/logs/vehicles_2_speed_log_2.csv
```

### Save as Video (MP4)

```bash
python speed_estimation/animate_csv.py \
    --csv_path speed_estimation/logs/vehicles_2_speed_log_2.csv \
    --output_path animation.mp4 \
    --fps 10
```

### Save as GIF

```bash
python speed_estimation/animate_csv.py \
    --csv_path speed_estimation/logs/vehicles_2_speed_log_2.csv \
    --output_path animation.gif \
    --fps 10
```

### Custom Options

```bash
python speed_estimation/animate_csv.py \
    --csv_path speed_estimation/logs/vehicles_2_speed_log_2.csv \
    --output_path animation.mp4 \
    --fps 15 \
    --trajectory_length 30 \
    --figsize 20 12
```

## Command-Line Arguments

- `--csv_path` (required): Path to the CSV file to animate
- `--output_path` (optional): Output path for animation file (MP4 or GIF). If not provided, displays interactively
- `--fps` (default: 10): Frames per second for animation
- `--no_trajectories`: Disable trajectory trails
- `--trajectory_length` (default: 20): Number of points to show in trajectory trail
- `--figsize` (default: 16 10): Figure size in inches (width height)

## Requirements

- `matplotlib` (with animation support)
- `pandas`
- `numpy`
- `pillow` (for GIF export)
- `ffmpeg` (optional, for MP4 export - install separately)

## Examples

### Example 1: Quick Preview

```bash
python speed_estimation/animate_csv.py \
    --csv_path speed_estimation/logs/vehicles_2_speed_log_2.csv \
    --fps 5
```

### Example 2: High-Quality Video

```bash
python speed_estimation/animate_csv.py \
    --csv_path speed_estimation/logs/vehicles_2_speed_log_2.csv \
    --output_path high_quality_animation.mp4 \
    --fps 30 \
    --figsize 20 12 \
    --trajectory_length 50
```

### Example 3: Simple GIF (No Trajectories)

```bash
python speed_estimation/animate_csv.py \
    --csv_path speed_estimation/logs/vehicles_2_speed_log_2.csv \
    --output_path simple_animation.gif \
    --fps 10 \
    --no_trajectories
```

## Output Format

The animation shows:

1. **Main Plot**: X-Y coordinate space with vehicles as colored circles
2. **Vehicle Labels**: Each vehicle shows:
   - Vehicle ID (#1, #2, etc.)
   - Vehicle type (car, truck, bus, motorcycle)
   - Speed in km/h (if available)
   - Decision symbol (🛑 for STOP, 🚗 for GO)

3. **Info Panel** (top-left):
   - Current frame number
   - Number of vehicles
   - Traffic density

4. **Traffic Light Indicator** (top-right):
   - Current traffic light status
   - Color-coded background

5. **Yellow Light Warning** (bottom-center):
   - Appears when yellow light is active

6. **Trajectory Trails** (optional):
   - Dashed lines showing vehicle paths
   - Color-matched to vehicle type

## Troubleshooting

### Issue: "ffmpeg not found" when saving MP4

**Solution**: 
- Install ffmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)
- Or use GIF format instead: `--output_path animation.gif`

### Issue: Animation is too slow/fast

**Solution**: Adjust FPS:
- Faster: `--fps 20` or higher
- Slower: `--fps 5` or lower

### Issue: Too many vehicles overlap

**Solution**: 
- Increase figure size: `--figsize 24 16`
- Disable trajectories: `--no_trajectories`
- Reduce trajectory length: `--trajectory_length 10`

### Issue: Labels overlap

The script automatically positions labels to avoid overlap, but with many vehicles, some overlap may occur. This is normal.

## Notes

- The animation loops continuously when displayed interactively
- For large CSV files, saving to file may take some time
- GIF files are larger than MP4 but don't require ffmpeg
- The coordinate system matches the original video frame coordinates


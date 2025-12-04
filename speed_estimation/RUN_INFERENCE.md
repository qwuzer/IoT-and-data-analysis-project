# Running inference_refactored.py

## Basic Usage

The script processes video files to detect vehicles, track them, estimate speeds, and detect traffic lights.

### Minimum Required Command

```bash
python inference_refactored.py --source_video_path <path_to_video>
```

### Example with Default Settings

```bash
python inference_refactored.py \
  --source_video_path logs/vehicles_2.mp4
```

This will:
- Use default YOLO model (`yolo11x.pt`)
- Auto-generate output video: `logs/vehicles_2_result.mp4`
- Auto-generate CSV log: `logs/vehicles_2_speed_log.csv`
- Use default traffic light ROI coordinates

## Full Example with All Options

```bash
python inference_refactored.py \
  --source_video_path logs/vehicles_2.mp4 \
  --model_path yolo11x.pt \
  --imgsz 1280 \
  --target_video_path logs/vehicles_2_annotated.mp4 \
  --csv_output_path logs/vehicles_2_speed_log.csv \
  --confidence_threshold 0.4 \
  --iou_threshold 0.7 \
  --visualize_first_frame
```

## Common Options

### Model Configuration
- `--model_path`: Path to YOLO model (default: `yolo11x.pt`)
- `--imgsz`: Input image size (default: 1280). Higher = more accurate but slower. Options: 640, 1280, 1920

### Output Paths
- `--target_video_path`: Output video path (auto-generated if not provided)
- `--csv_output_path`: CSV log path (auto-generated if not provided)

### Detection Thresholds
- `--confidence_threshold`: Minimum confidence for detections (default: 0.4)
- `--iou_threshold`: IOU threshold for NMS (default: 0.7)

### Traffic Light Detection
- `--traffic_light_roi`: ROI coordinates as `'x1,y1,x2,y2'` (rectangle) or `'x1,y1,x2,y2,x3,y3,x4,y4'` (polygon)
- `--traffic_light_segments`: Override segments as `'x1,y1,x2,y2;x1,y1,x2,y2;x1,y1,x2,y2'` (red;yellow;green)
- `--segment_change_threshold`: Minimum intensity change to detect light on (default: 12.0)
- `--segment_release_threshold`: Intensity change to detect light off (default: 6.0)
- `--segment_min_intensity`: Minimum intensity to consider light illuminated (default: 60.0)

### Visualization
- `--visualize_first_frame`: Show first frame with ROI polygons to verify coordinates

## Examples

### 1. Basic Run (Auto-generated outputs)
```bash
python inference_refactored.py \
  --source_video_path logs/vehicles_2.mp4
```

### 2. With Custom Traffic Light ROI
```bash
python inference_refactored.py \
  --source_video_path logs/vehicles_2.mp4 \
  --traffic_light_roi "500,50,518,55"
```

### 3. Verify ROI Coordinates First
```bash
python inference_refactored.py \
  --source_video_path logs/vehicles_2.mp4 \
  --visualize_first_frame
```

### 4. High Accuracy (Slower)
```bash
python inference_refactored.py \
  --source_video_path logs/vehicles_2.mp4 \
  --imgsz 1920 \
  --confidence_threshold 0.3
```

### 5. Fast Processing (Lower Accuracy)
```bash
python inference_refactored.py \
  --source_video_path logs/vehicles_2.mp4 \
  --imgsz 640 \
  --confidence_threshold 0.5
```

## Output Files

1. **Annotated Video**: Shows detected vehicles with bounding boxes, speeds, and traffic light status
2. **CSV Log**: Contains per-frame data with:
   - Frame index
   - Tracker ID
   - Vehicle type and class
   - Traffic light status
   - Speed (m/s and km/h)
   - Distance to stop line
   - Distance to front vehicle
   - Traffic density
   - TTC (Time to Collision)
   - Yellow light decision (STOP/GO)

## Notes

- The script uses default ROI coordinates defined in the code
- Output paths are auto-generated if not specified (prevents overwriting)
- Make sure you have the YOLO model file (`yolo11x.pt`) in the working directory or specify path
- Processing time depends on video length, resolution, and `--imgsz` setting


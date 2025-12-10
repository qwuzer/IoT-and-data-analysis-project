# Real-Time Pipeline Documentation

## Overview

The `realtime_pipeline.py` script provides a unified pipeline that combines:

1. **Live Stream Reading**: Reads video from m3u8 stream URLs (or extracts from webpages)
2. **Real-Time Vehicle Detection**: Uses YOLO for vehicle detection and ByteTrack for tracking
3. **Real-Time LSTM Predictions**: Makes STOP/GO predictions using the trained LSTM model
4. **Live Visualization**: Displays predictions overlaid on the video stream

This eliminates the need for separate recording and processing steps - everything happens in real-time.

## Features

- **Real-time processing**: No need to record videos first
- **Live predictions**: STOP/GO predictions displayed on each vehicle
- **Stream URL extraction**: Automatically extracts m3u8 URLs from webpages
- **Traffic light detection**: Detects red/yellow/green states
- **Vehicle tracking**: Maintains sequences for each tracked vehicle
- **Visual feedback**: Color-coded vehicles with speed and prediction labels

## Prerequisites

### Software Requirements

- Python 3.7+
- PyTorch (CPU or GPU)
- OpenCV (`cv2`)
- Ultralytics YOLO
- Supervision library
- All dependencies from `requirements.txt`

### Model Files Required

1. **YOLO Model**: Vehicle detection model (e.g., `yolo11x.pt`)
2. **LSTM Checkpoint**: Trained dilemma zone model (`.pt` file from training)

## Usage

### Basic Usage

```bash
python speed_estimation/realtime_pipeline.py \
    --lstm_checkpoint speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --webpage_url https://tw.live/cam/?id=BOT243
```

### With Direct Stream URL

```bash
python speed_estimation/realtime_pipeline.py \
    --lstm_checkpoint speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --stream_url https://example.com/stream.m3u8
```

### With Custom YOLO Model

```bash
python speed_estimation/realtime_pipeline.py \
    --lstm_checkpoint speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --yolo_model path/to/custom_yolo.pt \
    --webpage_url https://tw.live/cam/?id=BOT243
```

### Save Output Video

```bash
python speed_estimation/realtime_pipeline.py \
    --lstm_checkpoint speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --webpage_url https://tw.live/cam/?id=BOT243 \
    --save_output output_video.mp4
```

### Headless Mode (No Display)

```bash
python speed_estimation/realtime_pipeline.py \
    --lstm_checkpoint speed_estimation/LSTM/models/checkpoints/best_model.pt \
    --webpage_url https://tw.live/cam/?id=BOT243 \
    --no_display \
    --save_output output_video.mp4
```

## Command-Line Arguments

### Required Arguments

- `--lstm_checkpoint`: Path to LSTM model checkpoint file (`.pt`)

### Optional Arguments

- `--stream_url`: Direct m3u8 stream URL (alternative to `--webpage_url`)
- `--webpage_url`: Webpage URL to extract stream from (e.g., `https://tw.live/cam/?id=BOT243`)
- `--yolo_model`: Path to YOLO model file (default: `yolo11x.pt`)
- `--confidence_threshold`: YOLO confidence threshold (default: 0.4)
- `--iou_threshold`: YOLO IOU threshold (default: 0.7)
- `--imgsz`: YOLO input image size (default: 1280)
- `--no_display`: Disable video display window
- `--save_output`: Path to save output video

## How It Works

### 1. Stream Initialization

The pipeline opens the video stream (either directly from m3u8 URL or extracted from webpage) using OpenCV's `VideoCapture`.

### 2. Model Loading

- **YOLO Model**: Loaded for vehicle detection
- **LSTM Model**: Loaded for STOP/GO prediction
- **Normalization Parameters**: Loaded from checkpoint for feature normalization

### 3. Frame Processing Loop

For each frame:

1. **Traffic Light Detection**: Detects current traffic light state (RED/YELLOW/GREEN)
2. **Vehicle Detection**: YOLO detects vehicles in the frame
3. **Tracking**: ByteTrack maintains vehicle IDs across frames
4. **Feature Extraction**: Calculates:
   - Speed (m/s and km/h)
   - Distance to stop line
   - Time to collision (TTC)
   - Distance to front vehicle
   - Traffic density
   - Vehicle class ID
5. **Sequence Building**: Maintains a rolling window of the last N frames (default: 12) for each vehicle
6. **LSTM Prediction**: When a sequence is complete, makes STOP/GO prediction
7. **Visualization**: Draws bounding boxes, labels, and predictions on frame

### 4. Sequence Tracking

The `VehicleSequenceTracker` class maintains sequences for each tracked vehicle:

- Stores the last N feature vectors for each vehicle
- Returns a complete sequence when enough frames are collected
- Automatically handles vehicle entry/exit

### 5. Prediction Display

Each vehicle label shows:
- Vehicle ID: `#123`
- Vehicle type: `car`, `truck`, `bus`, `motorcycle`
- Speed: `45 km/h`
- Prediction: `STOP (0.75)` or `GO (0.23)`
  - Number in parentheses is P(STOP) probability

## Configuration

The pipeline uses the same configuration constants as `inference_refactored.py`:

- `SOURCE`: Polygon defining detection zone
- `TARGET_WIDTH`, `TARGET_HEIGHT`: Top-down view dimensions
- `PIXELS_TO_METERS`: Calibration factor
- `TRAFFIC_LIGHT_ROI`: Traffic light detection region
- `STOP_LINE`: Stop line coordinates

To modify these, edit the constants at the top of `realtime_pipeline.py`.

## Performance Considerations

### Real-Time Processing

- **Frame Rate**: Processing speed depends on:
  - YOLO inference time
  - LSTM inference time
  - Frame resolution
  - Number of vehicles tracked

### Optimization Tips

1. **Reduce YOLO Image Size**: Use `--imgsz 640` instead of 1280 for faster processing
2. **GPU Acceleration**: Ensure PyTorch and YOLO use GPU if available
3. **Lower Confidence Threshold**: Use `--confidence_threshold 0.3` to reduce false positives
4. **Limit Display**: Use `--no_display` when saving to file for better performance

### Expected Performance

- **CPU Only**: ~5-10 FPS (depending on hardware)
- **GPU (CUDA)**: ~15-30 FPS (depending on GPU)
- **Stream Latency**: 1-3 seconds (network + processing)

## Troubleshooting

### Issue: "Failed to open stream"

**Solutions**:
- Check if the stream URL is accessible in a browser
- Try using `--webpage_url` instead of `--stream_url` for automatic extraction
- Verify network connectivity
- Some streams may require authentication or specific headers

### Issue: Low FPS / Laggy Processing

**Solutions**:
- Reduce `--imgsz` to 640
- Use GPU if available
- Reduce number of tracked vehicles (adjust confidence threshold)
- Disable display with `--no_display`

### Issue: No Predictions Showing

**Possible Causes**:
1. **Insufficient Sequence Length**: Vehicle needs to be tracked for at least 12 frames
2. **Missing Features**: Speed or distance calculations may be failing
3. **Model Not Loaded**: Check that LSTM checkpoint path is correct

**Solutions**:
- Wait for vehicles to be tracked longer
- Check console output for errors
- Verify LSTM checkpoint file exists and is valid

### Issue: Stream URL Extraction Fails

**Solutions**:
- Manually find the m3u8 URL using browser developer tools
- Use `--stream_url` with the direct URL
- Install `yt-dlp` for better extraction: `pip install yt-dlp`

### Issue: CUDA Out of Memory

**Solutions**:
- Reduce `--imgsz` to 640
- Process fewer vehicles (increase `--confidence_threshold`)
- Use CPU mode (if CUDA is causing issues)

## Integration with Existing Workflow

### Comparison to Previous Workflow

**Previous Workflow**:
1. `grep.py` → Record videos from stream
2. `inference_refactored.py` → Process videos → Generate CSV
3. LSTM model → Predict from CSV

**New Real-Time Workflow**:
1. `realtime_pipeline.py` → Stream + Process + Predict in real-time

### When to Use Each

**Use Real-Time Pipeline When**:
- You need immediate feedback
- You want to monitor live traffic
- You don't need to save all data
- You want to see predictions as they happen

**Use Batch Processing When**:
- You need to process historical videos
- You need detailed CSV logs
- You want to analyze data offline
- You need to train models on collected data

## Example Output

The pipeline displays a video window with:

- **Red polygon**: Vehicle detection zone
- **Colored boxes**: Detected vehicles (red=car, green=truck, blue=bus, yellow=motorcycle)
- **Traces**: Vehicle movement trails
- **Labels**: Vehicle ID, type, speed, and prediction
- **Traffic light indicator**: Current light state (RED/YELLOW/GREEN)

Example label: `#123 car 45 km/h STOP (0.75)`

This means:
- Vehicle ID: 123
- Type: car
- Speed: 45 km/h
- Prediction: STOP with 75% probability

## Future Enhancements

Potential improvements:

1. **Multi-stream Support**: Process multiple streams simultaneously
2. **Database Logging**: Save predictions to database in real-time
3. **Alert System**: Trigger alerts for high-risk situations
4. **Web Dashboard**: Real-time web interface for monitoring
5. **Model Ensembling**: Combine multiple models for better accuracy

## Notes

- The pipeline requires vehicles to be tracked for at least 12 frames before making predictions
- Predictions are cached per vehicle until new sequence is available
- Traffic light detection uses intensity-based change detection (not color-based)
- All coordinate transformations assume a calibrated camera setup

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Verify all model files and dependencies are correctly installed
4. Test with a known-good stream URL first


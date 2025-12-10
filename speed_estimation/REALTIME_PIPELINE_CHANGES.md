# Real-Time Pipeline - Changes Summary

## Overview

This document summarizes the changes made to create a unified real-time pipeline that processes video streams and makes LSTM predictions in real-time.

## Files Created/Modified

### New Files

1. **`speed_estimation/realtime_pipeline.py`** (977 lines)
   - Main real-time processing pipeline
   - Combines stream reading, vehicle detection, tracking, and LSTM predictions
   - Supports both m3u8 URLs and webpage URL extraction

2. **`speed_estimation/REALTIME_PIPELINE.md`**
   - Comprehensive documentation for the real-time pipeline
   - Usage examples, troubleshooting, and configuration

3. **`speed_estimation/QUICKSTART_REALTIME.md`**
   - Quick start guide for the real-time pipeline

4. **`speed_estimation/LSTM/DILEMMA_ZONE_GENERATION.md`**
   - Documentation for generating dilemma zone visualizations

### Modified Files

None (all changes are in new files)

## Key Features Implemented

### 1. Real-Time Stream Processing
- Reads from m3u8 stream URLs directly
- Extracts stream URLs from webpages (like `grep.py`)
- Handles stream interruptions gracefully

### 2. Vehicle Detection & Tracking
- Uses YOLO for vehicle detection
- ByteTrack for multi-object tracking
- Maintains vehicle IDs across frames

### 3. Feature Extraction
- Calculates speed (m/s and km/h)
- Distance to stop line
- Time to collision (TTC)
- Distance to front vehicle
- Traffic density
- Vehicle class ID

### 4. Sequence Building
- `VehicleSequenceTracker` class maintains rolling sequences
- Stores last N frames (default: 12) for each vehicle
- Automatically handles vehicle entry/exit

### 5. LSTM Predictions
- Makes STOP/GO predictions when sequences are complete
- Caches predictions until new sequence is available
- Displays predictions with confidence scores

### 6. Visualization
- Real-time video display with annotations
- Color-coded vehicles by type
- Labels show: ID, type, speed, and prediction
- Traffic light status indicator
- Vehicle movement trails

## Code Improvements Made

### 1. Normalization Fix
**Issue**: Normalization was manually parsing strings, which could fail.

**Fix**: Now uses the `normalize_features` function from `LSTM.utils`, which:
- Handles both string and numpy array formats
- Supports standard and min-max normalization
- Properly handles edge cases

**Location**: `predict_stop_go()` method in `RealTimePipeline` class

### 2. Import Error Handling
**Issue**: Import errors weren't clearly explained.

**Fix**: Added try-except blocks with helpful error messages:
- Clear instructions on how to fix missing dependencies
- Suggests running from correct directory

**Location**: Module imports at top of file

### 3. Stream URL Extraction
**Issue**: No error handling for missing `requests` package.

**Fix**: Added try-except for imports and better error messages.

**Location**: `extract_stream_url()` function

## Architecture

### Class Structure

1. **`ViewTransformer`**: Transforms camera coordinates to top-down view
2. **`TrafficLightChangeDetector`**: Detects traffic light states
3. **`VehicleSequenceTracker`**: Maintains sequences for each vehicle
4. **`RealTimePipeline`**: Main pipeline orchestrating all components

### Data Flow

```
Stream URL → VideoCapture → Frame
    ↓
Traffic Light Detection
    ↓
YOLO Detection → ByteTrack Tracking
    ↓
Feature Extraction (speed, distance, TTC, etc.)
    ↓
Sequence Building (rolling window)
    ↓
LSTM Prediction (when sequence complete)
    ↓
Visualization (annotated frame)
    ↓
Display / Save
```

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

## Testing Checklist

- [ ] Test with a known-good m3u8 stream URL
- [ ] Test webpage URL extraction
- [ ] Verify LSTM predictions appear after ~12 frames
- [ ] Test with different vehicle types
- [ ] Verify normalization works correctly
- [ ] Test error handling (missing dependencies, invalid URLs)
- [ ] Test headless mode (--no_display)
- [ ] Test video saving (--save_output)

## Known Limitations

1. **Sequence Length**: Vehicles need to be tracked for at least 12 frames before predictions
2. **Performance**: Real-time processing depends on hardware (GPU recommended)
3. **Stream Latency**: Network latency affects real-time feedback
4. **Calibration**: Requires calibrated camera setup for accurate distance/speed

## Future Enhancements

1. Multi-stream support
2. Database logging of predictions
3. Alert system for high-risk situations
4. Web dashboard for monitoring
5. Model ensembling for better accuracy

## Dependencies

Required packages:
- `torch` (PyTorch)
- `opencv-python` (cv2)
- `ultralytics` (YOLO)
- `supervision`
- `numpy`
- `requests` (for stream URL extraction)

All should be in `requirements.txt`.

## Notes

- The pipeline uses the same configuration constants as `inference_refactored.py`
- Normalization parameters are loaded from the LSTM checkpoint
- Predictions are cached per vehicle to avoid redundant computation
- The pipeline automatically handles GPU/CPU selection


#!/usr/bin/env python3
"""
Unified Real-Time Pipeline for Vehicle Speed Estimation and Dilemma Zone Prediction

This script combines:
1. Live stream reading from m3u8 URLs
2. Real-time vehicle detection and tracking
3. Real-time LSTM model predictions for STOP/GO decisions
4. Live visualization with predictions overlaid on video

Usage:
    python speed_estimation/realtime_pipeline.py \
        --stream_url <m3u8_url> \
        --model_checkpoint <path_to_lstm_model.pt> \
        --yolo_model <path_to_yolo_model.pt>
"""

import argparse
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Dict, Deque
import time
import sys

try:
    from ultralytics import YOLO
    import supervision as sv
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip install ultralytics supervision")
    sys.exit(1)

# Import LSTM model components
import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import LSTM modules
try:
    from LSTM.evaluate_model import load_model
    from LSTM.dilemma_zone_generator import create_synthetic_sequence
    from LSTM.utils import normalize_features
    from LSTM.config import SEQUENCE_LENGTH, FEATURE_DIM, FEATURE_COLUMNS, NORMALIZATION_METHOD
except ImportError as e:
    print(f"Error importing LSTM modules: {e}")
    print("Make sure you're running from the correct directory")
    print("Try: cd speed_estimation && python realtime_pipeline.py ...")
    sys.exit(1)

# Vehicle type mapping (COCO classes)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Color mapping for different vehicle types
VEHICLE_COLORS_HEX = {
    "car": "#FF0000",          # Red
    "truck": "#00FF00",        # Green
    "bus": "#0000FF",          # Blue
    "motorcycle": "#FFFF00",   # Yellow
}

VEHICLE_COLOR_PALETTE = sv.ColorPalette.from_hex(
    list(VEHICLE_COLORS_HEX.values())
)

VEHICLE_COLOR_INDICES = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
}

MIN_CAR_WIDTH_PX = 20

# Default configuration (should match inference_refactored.py)
SOURCE = np.array([[420, 101], [536, 101], [800, 240], [435, 250]])
TARGET_WIDTH = 10
TARGET_HEIGHT = 60
PIXELS_TO_METERS = 1.0
TRAFFIC_LIGHT_ROI = np.array([[500, 50], [520, 50], [520, 55], [500, 55]])
STOP_LINE = np.array([[420, 101], [536, 101]])
TRAFFIC_LIGHT_SEGMENT_ORDER = ("red", "yellow", "green", "unused")


class ViewTransformer:
    """Transform points from camera view to top-down view."""
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class TrafficLightChangeDetector:
    """Detects traffic light states by monitoring intensity changes."""
    
    def __init__(
        self,
        segment_boxes,
        on_change_threshold=12.0,
        off_change_threshold=6.0,
        min_intensity=60.0,
        initial_on_threshold=80.0,
        initialization_frames=10,
    ):
        self.segment_boxes = segment_boxes or {}
        self.on_change_threshold = on_change_threshold
        self.off_change_threshold = off_change_threshold
        self.min_intensity = min_intensity
        self.initial_on_threshold = initial_on_threshold
        self.initialization_frames = initialization_frames
        self.frame_count = 0
        self.initialization_complete = False
        self.initial_intensities = []
        self.previous_intensity = {name: None for name in self.segment_boxes}
        self.off_reference = {name: None for name in self.segment_boxes}
        self.states = {name: False for name in self.segment_boxes}

    @staticmethod
    def _extract_region(frame, box):
        if box is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def detect(self, frame):
        """Detect traffic light state from frame."""
        if not self.segment_boxes:
            return False, False, False, "N/A", {}
        
        statuses = {}
        intensities = {}
        current_frame_intensities = {}
        intensity_deltas = {}
        
        for name, box in self.segment_boxes.items():
            if name == "unused":
                continue
            region = self._extract_region(frame, box)
            if region is None or region.size == 0:
                statuses[name] = False
                intensities[name] = 0.0
                current_frame_intensities[name] = 0.0
                intensity_deltas[name] = 0.0
                continue

            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            intensity = float(np.mean(gray_region))
            intensities[name] = intensity
            current_frame_intensities[name] = intensity
            
            prev_intensity = self.previous_intensity.get(name)
            off_reference = self.off_reference.get(name)
            
            if not self.initialization_complete:
                if off_reference is None:
                    off_reference = intensity
                self.off_reference[name] = off_reference
                intensity_deltas[name] = 0.0
                continue
            
            if off_reference is None:
                off_reference = intensity

            delta_prev = 0.0 if prev_intensity is None else intensity - prev_intensity
            delta_reference = intensity - off_reference
            max_positive_delta = max(delta_prev, delta_reference, 0.0)
            intensity_deltas[name] = max_positive_delta
            state = self.states.get(name, False)

            if not state:
                if (
                    (delta_prev >= self.on_change_threshold or delta_reference >= self.on_change_threshold)
                    and intensity >= self.min_intensity
                ):
                    state = True
            else:
                if (
                    (-delta_prev) >= self.off_change_threshold
                    or intensity < off_reference + self.on_change_threshold
                ):
                    state = False

            if not state:
                off_reference = (off_reference * 0.9) + (intensity * 0.1)

            self.previous_intensity[name] = intensity
            self.off_reference[name] = off_reference
            self.states[name] = state
            statuses[name] = state
        
        if not self.initialization_complete:
            self.initial_intensities.append(current_frame_intensities.copy())
            self.frame_count += 1
            
            if self.frame_count >= self.initialization_frames:
                avg_intensities = {}
                for name in current_frame_intensities.keys():
                    if name == "unused":
                        continue
                    samples = [frame_data.get(name, 0.0) for frame_data in self.initial_intensities if name in frame_data]
                    if samples:
                        avg_intensities[name] = sum(samples) / len(samples)
                
                for name, avg_intensity in avg_intensities.items():
                    if avg_intensity >= self.initial_on_threshold:
                        self.states[name] = True
                        statuses[name] = True
                        estimated_off_intensity = max(avg_intensity - self.on_change_threshold * 2, self.min_intensity - 10)
                        self.off_reference[name] = estimated_off_intensity
                        self.previous_intensity[name] = avg_intensity
                    else:
                        self.states[name] = False
                        statuses[name] = False
                        self.off_reference[name] = avg_intensity
                        self.previous_intensity[name] = avg_intensity
                
                self.initialization_complete = True
                red_on = statuses.get("red", False)
                yellow_on = statuses.get("yellow", False)
                green_on = statuses.get("green", False)
                
                if red_on and yellow_on:
                    red_intensity = avg_intensities.get("red", 0.0)
                    yellow_intensity = avg_intensities.get("yellow", 0.0)
                    if yellow_intensity > red_intensity:
                        red_on = False
                    else:
                        yellow_on = False
                
                if red_on:
                    status_text = "RED"
                elif yellow_on:
                    status_text = "YELLOW"
                elif green_on:
                    status_text = "GREEN"
                else:
                    status_text = "OFF"
                
                return red_on, yellow_on, green_on, status_text, statuses
            else:
                return False, False, False, "OFF", {}

        red_on = statuses.get("red", False)
        yellow_on = statuses.get("yellow", False)
        green_on = statuses.get("green", False)
        
        if red_on and yellow_on:
            red_delta = intensity_deltas.get("red", 0.0)
            yellow_delta = intensity_deltas.get("yellow", 0.0)
            red_intensity = intensities.get("red", 0.0)
            yellow_intensity = intensities.get("yellow", 0.0)
            
            red_score = red_delta * 0.6 + (red_intensity / 255.0) * 100.0 * 0.4
            yellow_score = yellow_delta * 0.6 + (yellow_intensity / 255.0) * 100.0 * 0.4
            
            if yellow_score > red_score:
                red_on = False
            else:
                yellow_on = False

        if red_on:
            status_text = "RED"
        elif yellow_on:
            status_text = "YELLOW"
        elif green_on:
            status_text = "GREEN"
        else:
            status_text = "OFF"

        return red_on, yellow_on, green_on, status_text, statuses


def derive_segment_boxes(roi_coords):
    """Split ROI into segments for red, yellow, green lights."""
    if roi_coords is None:
        return {}
    if isinstance(roi_coords, np.ndarray):
        polygon = roi_coords.astype(np.int32)
        if polygon.size == 0:
            return {}
        x, y, w, h = cv2.boundingRect(polygon)
        x2, y2 = x + w, y + h
    elif isinstance(roi_coords, tuple) and len(roi_coords) == 4:
        x, y, x2, y2 = [int(v) for v in roi_coords]
        w = x2 - x
        h = y2 - y
    else:
        return {}
    if w <= 0 or h <= 0:
        return {}
    horizontal = w >= h
    segment_count = len(TRAFFIC_LIGHT_SEGMENT_ORDER)
    boxes = {}
    for idx, name in enumerate(TRAFFIC_LIGHT_SEGMENT_ORDER):
        if horizontal:
            start = int(round(x + idx * (w / segment_count)))
            end = int(round(x + (idx + 1) * (w / segment_count)))
            boxes[name] = (start, y, end, y2)
        else:
            start = int(round(y + idx * (h / segment_count)))
            end = int(round(y + (idx + 1) * (h / segment_count)))
            boxes[name] = (x, start, x2, end)
    return boxes


class VehicleSequenceTracker:
    """Tracks vehicle sequences for LSTM prediction."""
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        # Store sequences for each vehicle: tracker_id -> deque of feature vectors
        self.sequences: Dict[int, Deque] = defaultdict(lambda: deque(maxlen=sequence_length))
        # Store last known features for each vehicle
        self.last_features: Dict[int, Dict] = {}
    
    def update(self, tracker_id: int, features: Dict) -> Optional[np.ndarray]:
        """
        Update sequence for a vehicle and return sequence if ready.
        
        Args:
            tracker_id: Vehicle tracker ID
            features: Dictionary with feature values:
                - speed_ms
                - distance_to_stop_line
                - ttc
                - distance_to_front_vehicle
                - traffic_density
                - class_id
        
        Returns:
            Sequence array of shape (sequence_length, feature_dim) if ready, None otherwise
        """
        # Create feature vector
        feature_vector = np.array([
            features.get('speed_ms', 0.0),
            features.get('distance_to_stop_line', 0.0),
            features.get('ttc', 0.0),
            features.get('distance_to_front_vehicle', 10.0),
            features.get('traffic_density', 0.0),
            features.get('class_id', 2.0)
        ], dtype=np.float32)
        
        # Add to sequence
        self.sequences[tracker_id].append(feature_vector)
        self.last_features[tracker_id] = features
        
        # Return sequence if we have enough frames
        if len(self.sequences[tracker_id]) >= self.sequence_length:
            sequence = np.array(list(self.sequences[tracker_id]), dtype=np.float32)
            return sequence
        return None
    
    def get_last_sequence(self, tracker_id: int) -> Optional[np.ndarray]:
        """Get the last complete sequence for a vehicle."""
        if tracker_id in self.sequences and len(self.sequences[tracker_id]) >= self.sequence_length:
            return np.array(list(self.sequences[tracker_id]), dtype=np.float32)
        return None
    
    def remove(self, tracker_id: int):
        """Remove a vehicle from tracking."""
        if tracker_id in self.sequences:
            del self.sequences[tracker_id]
        if tracker_id in self.last_features:
            del self.last_features[tracker_id]


class RealTimePipeline:
    """Main pipeline for real-time processing."""
    
    def __init__(
        self,
        stream_url: str,
        yolo_model_path: str,
        lstm_checkpoint_path: str,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.7,
        imgsz: int = 1280,
        display: bool = True,
        save_output: Optional[str] = None,
    ):
        self.stream_url = stream_url
        self.yolo_model_path = yolo_model_path
        self.lstm_checkpoint_path = lstm_checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.display = display
        self.save_output = save_output
        
        # Initialize models
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        print("Loading LSTM model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model, self.normalizer_params, self.model_config = load_model(
            lstm_checkpoint_path, self.device
        )
        self.lstm_model.eval()
        print(f"LSTM model loaded on {self.device}")
        
        # Initialize components
        self.byte_track = None  # Will be initialized when we know FPS
        self.view_transformer = ViewTransformer(source=SOURCE, target=np.array([
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1],
        ]))
        
        stop_line_points_topview = self.view_transformer.transform_points(STOP_LINE.astype(np.float32))
        self.stop_line_y_topview = float(np.mean(stop_line_points_topview[:, 1])) if len(stop_line_points_topview) > 0 else None
        
        traffic_light_segments = derive_segment_boxes(TRAFFIC_LIGHT_ROI)
        self.traffic_light_detector = TrafficLightChangeDetector(segment_boxes=traffic_light_segments)
        
        self.sequence_tracker = VehicleSequenceTracker(sequence_length=SEQUENCE_LENGTH)
        
        # Tracking data
        self.coordinates = defaultdict(lambda: deque(maxlen=30))  # Store last 30 frames
        self.vehicle_crossing_state = {}
        self.predictions_cache = {}  # Cache predictions for each vehicle
        
        # Annotators
        self.box_annotator = None
        self.trace_annotator = None
        self.polygon_zone = sv.PolygonZone(polygon=SOURCE)
        
        # Video writer
        self.video_writer = None
        self.video_info = None
    
    def predict_stop_go(self, sequence: np.ndarray) -> Dict:
        """
        Make STOP/GO prediction from a sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, feature_dim)
        
        Returns:
            Dictionary with prediction results
        """
        # Normalize sequence using the normalize_features function
        if self.normalizer_params:
            # Handle both string format (from JSON) and numpy array format
            norm_params = {}
            
            if NORMALIZATION_METHOD == "standard":
                mean = self.normalizer_params.get('mean')
                std = self.normalizer_params.get('std')
                
                # Parse from string format if needed
                if isinstance(mean, str):
                    mean = np.fromstring(mean.strip('[]'), sep=' ', dtype=np.float32)
                elif mean is not None:
                    mean = np.array(mean, dtype=np.float32)
                
                if isinstance(std, str):
                    std = np.fromstring(std.strip('[]'), sep=' ', dtype=np.float32)
                elif std is not None:
                    std = np.array(std, dtype=np.float32)
                
                if mean is not None and std is not None:
                    norm_params = {'mean': mean, 'std': std}
            elif NORMALIZATION_METHOD == "minmax":
                min_val = self.normalizer_params.get('min')
                max_val = self.normalizer_params.get('max')
                
                if isinstance(min_val, str):
                    min_val = np.fromstring(min_val.strip('[]'), sep=' ', dtype=np.float32)
                elif min_val is not None:
                    min_val = np.array(min_val, dtype=np.float32)
                
                if isinstance(max_val, str):
                    max_val = np.fromstring(max_val.strip('[]'), sep=' ', dtype=np.float32)
                elif max_val is not None:
                    max_val = np.array(max_val, dtype=np.float32)
                
                if min_val is not None and max_val is not None:
                    norm_params = {'min': min_val, 'max': max_val}
            
            if norm_params:
                # Flatten sequence for normalization
                sequence_flat = sequence.reshape(-1, FEATURE_DIM)
                # Normalize using the utility function
                sequence_normalized, _ = normalize_features(
                    sequence_flat,
                    fit=False,
                    **norm_params
                )
                # Reshape back to sequence
                sequence = sequence_normalized.reshape(SEQUENCE_LENGTH, FEATURE_DIM)
        
        # Convert to tensor and predict
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob_stop = self.lstm_model(sequence_tensor).cpu().item()
        
        prediction = "STOP" if prob_stop >= 0.5 else "GO"
        confidence = prob_stop if prob_stop >= 0.5 else (1 - prob_stop)
        
        return {
            'prediction': prediction,
            'probability_stop': float(prob_stop),
            'probability_go': float(1 - prob_stop),
            'confidence': float(confidence),
        }
    
    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Process a single frame and return annotated frame."""
        # Detect traffic light
        red_on, yellow_on, green_on, traffic_light_status, _ = self.traffic_light_detector.detect(frame)
        if red_on:
            traffic_light_status = "RED"
        elif yellow_on:
            traffic_light_status = "YELLOW"
        elif green_on:
            traffic_light_status = "GREEN"
        
        yellow_active = traffic_light_status == "YELLOW"
        
        # Run YOLO detection
        results = self.yolo_model(frame, imgsz=self.imgsz)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections[self.polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=self.iou_threshold)
        
        # Filter vehicles
        if len(detections) > 0 and hasattr(detections, "class_id") and detections.class_id is not None:
            keep_mask = np.ones(len(detections), dtype=bool)
            for idx in range(len(detections)):
                cls_id = int(detections.class_id[idx])
                if cls_id == 0:  # Person
                    keep_mask[idx] = False
                    continue
                if cls_id == 2:  # Car
                    x1, y1, x2, y2 = detections.xyxy[idx]
                    if (x2 - x1) < MIN_CAR_WIDTH_PX:
                        keep_mask[idx] = False
                        continue
                if cls_id not in VEHICLE_CLASSES:
                    keep_mask[idx] = False
            detections = detections[keep_mask]
        
        # Update tracking
        detections = self.byte_track.update_with_detections(detections=detections)
        
        # Transform to top-down view
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = self.view_transformer.transform_points(points=points).astype(int)
        
        # Update coordinates for speed calculation
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            self.coordinates[tracker_id].append(y)
        
        traffic_density = len(detections)
        vehicle_positions = {
            tracker_id: points[idx] for idx, tracker_id in enumerate(detections.tracker_id)
        }
        
        # Process each vehicle
        labels = []
        color_lookup_indices = []
        
        for det_idx, tracker_id in enumerate(detections.tracker_id):
            x_curr, y_curr = points[det_idx]
            
            # Get vehicle type
            class_id = detections.class_id[det_idx] if hasattr(detections, 'class_id') and detections.class_id is not None else None
            vehicle_type = VEHICLE_CLASSES.get(int(class_id) if class_id is not None else 2, "car")
            color_idx = VEHICLE_COLOR_INDICES.get(vehicle_type, 0)
            color_lookup_indices.append(color_idx)
            
            # Calculate distance to stop line
            distance_to_stop_line_m = None
            if self.stop_line_y_topview is not None:
                distance_to_stop_line_px = float(y_curr - self.stop_line_y_topview)
                distance_to_stop_line_m = distance_to_stop_line_px * PIXELS_TO_METERS
            
            # Calculate speed
            history = self.coordinates[tracker_id]
            speed_ms = None
            speed_kmh = None
            ttc = None
            
            if len(history) >= 2:
                coordinate_start = history[-1]
                coordinate_end = history[0]
                distance_px = abs(coordinate_start - coordinate_end)
                time_s = len(history) / (self.video_info.fps if self.video_info else 30.0)
                if time_s > 0:
                    speed_pixels_per_sec = distance_px / time_s
                    speed_ms = speed_pixels_per_sec * PIXELS_TO_METERS
                    speed_kmh = speed_ms * 3.6
            
            # Calculate TTC
            if speed_ms is not None and distance_to_stop_line_m is not None and distance_to_stop_line_m > 0 and speed_ms > 0:
                ttc = distance_to_stop_line_m / speed_ms
            
            # Calculate distance to front vehicle
            distance_to_front_vehicle_m = None
            current_y = y_curr
            front_vehicles = []
            for other_tracker_id, (other_x, other_y) in vehicle_positions.items():
                if other_tracker_id == tracker_id:
                    continue
                y_diff = current_y - other_y
                x_diff = abs(x_curr - other_x)
                if y_diff > 0 and x_diff < TARGET_WIDTH * 2:
                    front_vehicles.append(y_diff)
            
            if front_vehicles:
                distance_to_front_vehicle_m = float(min(front_vehicles)) * PIXELS_TO_METERS
            
            # Update sequence tracker and get prediction
            if speed_ms is not None and distance_to_stop_line_m is not None:
                features = {
                    'speed_ms': speed_ms,
                    'distance_to_stop_line': distance_to_stop_line_m,
                    'ttc': ttc if ttc is not None else 0.0,
                    'distance_to_front_vehicle': distance_to_front_vehicle_m if distance_to_front_vehicle_m is not None else 10.0,
                    'traffic_density': traffic_density,
                    'class_id': float(class_id) if class_id is not None else 2.0
                }
                
                sequence = self.sequence_tracker.update(tracker_id, features)
                
                if sequence is not None:
                    # Make prediction
                    prediction_result = self.predict_stop_go(sequence)
                    self.predictions_cache[tracker_id] = prediction_result
                elif tracker_id in self.predictions_cache:
                    # Use cached prediction
                    prediction_result = self.predictions_cache[tracker_id]
                else:
                    prediction_result = None
            else:
                prediction_result = None
            
            # Build label
            label_parts = [f"#{tracker_id}", vehicle_type]
            if speed_kmh is not None:
                label_parts.append(f"{int(speed_kmh)} km/h")
            if prediction_result:
                pred_text = prediction_result['prediction']
                conf = prediction_result['confidence']
                prob_stop = prediction_result['probability_stop']
                label_parts.append(f"{pred_text} ({prob_stop:.2f})")
            
            labels.append(" ".join(label_parts))
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Draw polygons
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame,
            polygon=SOURCE,
            color=sv.Color(255, 0, 0),
            thickness=1
        )
        
        if traffic_light_status in ["RED", "YELLOW", "GREEN"]:
            roi_color_map = {
                "RED": sv.Color(255, 0, 0),
                "YELLOW": sv.Color(255, 255, 0),
                "GREEN": sv.Color(0, 255, 0),
            }
            roi_color = roi_color_map.get(traffic_light_status, sv.Color(128, 128, 128))
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=TRAFFIC_LIGHT_ROI,
                color=roi_color,
                thickness=2
            )
        
        # Draw traffic light status
        status_text = f"Traffic Light: {traffic_light_status}"
        cv2.putText(
            annotated_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if traffic_light_status == "GREEN" else (0, 0, 255) if traffic_light_status == "RED" else (0, 255, 255),
            2
        )
        
        # Draw traces
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        # Draw boxes
        custom_color_lookup = np.array(color_lookup_indices) if color_lookup_indices else None
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            custom_color_lookup=custom_color_lookup
        )
        
        # Draw labels
        for idx, (label, bbox) in enumerate(zip(labels, detections.xyxy)):
            if idx < len(detections):
                x1, y1, x2, y2 = bbox.astype(int)
                label_y = y1 - 10 if y1 > 20 else y2 + 20
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        return annotated_frame
    
    def run(self):
        """Run the real-time pipeline."""
        print(f"Opening stream: {self.stream_url}")
        cap = cv2.VideoCapture(self.stream_url)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open stream: {self.stream_url}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream properties: {width}x{height} @ {fps} fps")
        
        # Initialize ByteTrack with known FPS
        self.byte_track = sv.ByteTrack(
            frame_rate=fps,
            track_activation_threshold=self.confidence_threshold
        )
        
        # Initialize annotators
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=(width, height))
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(width, height))
        
        self.box_annotator = sv.BoxAnnotator(
            color=VEHICLE_COLOR_PALETTE,
            thickness=thickness,
            color_lookup=sv.ColorLookup.INDEX
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=int(fps * 2),
            position=sv.Position.BOTTOM_CENTER,
        )
        
        # Create video info object
        self.video_info = type('VideoInfo', (), {
            'fps': fps,
            'width': width,
            'height': height
        })()
        
        # Initialize video writer if needed
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.save_output,
                fourcc,
                fps,
                (width, height)
            )
            print(f"Saving output to: {self.save_output}")
        
        frame_index = 0
        print("\nStarting real-time processing...")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame or stream ended")
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame, frame_index)
                
                # Save frame if needed
                if self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                # Display frame
                if self.display:
                    cv2.imshow('Real-Time Pipeline', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quitting...")
                        break
                
                frame_index += 1
                
                # Print status every 30 frames
                if frame_index % 30 == 0:
                    print(f"Processed {frame_index} frames, Tracking {len(self.sequence_tracker.sequences)} vehicles")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            if self.display:
                cv2.destroyAllWindows()
            print(f"\nProcessed {frame_index} frames total")


def extract_stream_url(webpage_url: str) -> Optional[str]:
    """Extract m3u8 stream URL from webpage (from grep.py)."""
    try:
        import requests
        import re
    except ImportError:
        print("Error: 'requests' package not found. Install with: pip install requests")
        return None
    
    print(f"Fetching webpage: {webpage_url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(webpage_url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
        
        m3u8_patterns = [
            r'https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*',
            r'["\']([^"\']*\.m3u8[^"\']*)["\']',
        ]
        
        for pattern in m3u8_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                for match in matches:
                    url = match if match.startswith('http') else match
                    if '.m3u8' in url:
                        print(f"Found stream URL: {url}")
                        return url
        
        print("Warning: Could not find m3u8 URL in webpage. Try using --stream_url with direct URL.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching webpage: {e}")
        return None
    except Exception as e:
        print(f"Error extracting stream URL: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Unified Real-Time Pipeline for Vehicle Speed Estimation and Dilemma Zone Prediction"
    )
    parser.add_argument(
        '--stream_url',
        type=str,
        default=None,
        help='Direct m3u8 stream URL (or use --webpage_url to extract automatically)'
    )
    parser.add_argument(
        '--webpage_url',
        type=str,
        default=None,
        help='Webpage URL to extract stream from (e.g., https://tw.live/cam/?id=BOT243)'
    )
    parser.add_argument(
        '--yolo_model',
        type=str,
        default='yolo11x.pt',
        help='Path to YOLO model file (default: yolo11x.pt)'
    )
    parser.add_argument(
        '--lstm_checkpoint',
        type=str,
        required=True,
        help='Path to LSTM model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.4,
        help='YOLO confidence threshold (default: 0.4)'
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.7,
        help='YOLO IOU threshold (default: 0.7)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=1280,
        help='YOLO input image size (default: 1280)'
    )
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='Disable video display window'
    )
    parser.add_argument(
        '--save_output',
        type=str,
        default=None,
        help='Path to save output video (optional)'
    )
    
    args = parser.parse_args()
    
    # Determine stream URL
    stream_url = args.stream_url
    if not stream_url and args.webpage_url:
        stream_url = extract_stream_url(args.webpage_url)
        if not stream_url:
            print("Failed to extract stream URL from webpage")
            return
    
    if not stream_url:
        print("Error: Must provide either --stream_url or --webpage_url")
        return
    
    # Create and run pipeline
    pipeline = RealTimePipeline(
        stream_url=stream_url,
        yolo_model_path=args.yolo_model,
        lstm_checkpoint_path=args.lstm_checkpoint,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        imgsz=args.imgsz,
        display=not args.no_display,
        save_output=args.save_output,
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()


import argparse
import os
from collections import defaultdict, deque
from pathlib import Path

import cv2
import csv   # NEW
import numpy as np
from ultralytics import YOLO

import supervision as sv

# Vehicle type mapping (COCO classes)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Color mapping for different vehicle types (hex format for ColorPalette)
VEHICLE_COLORS_HEX = {
    "car": "#FF0000",          # Red
    "truck": "#00FF00",        # Green
    "bus": "#0000FF",          # Blue
    "motorcycle": "#FFFF00",   # Yellow
    # "unknown": "#808080",      # Gray (default)
}

# Create color palette from vehicle colors
VEHICLE_COLOR_PALETTE = sv.ColorPalette.from_hex(
    list(VEHICLE_COLORS_HEX.values())
)

# Map vehicle types to color indices in the palette
VEHICLE_COLOR_INDICES = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    # "unknown": 4,
}

# Minimum pixel width for car detections to be considered valid.
# Cars narrower than this are likely noise or riders and will be ignored.
MIN_CAR_WIDTH_PX = 20

# SOURCE = np.array([[25, 210],
#     [270, 220],
#     [859, 520],
#     [35, 520]])

SOURCE = np.array([[420, 101], [536, 101], [800, 240], [435, 250]])

TARGET_WIDTH = 10
TARGET_HEIGHT = 60

# Calibration factor: pixels to meters conversion
# This should be calibrated based on your specific setup
# Example: If TARGET_HEIGHT (60 pixels) represents 60 meters, then PIXELS_TO_METERS = 1.0
# Adjust this value based on your actual setup
PIXELS_TO_METERS = 1.0  # Default: 1 pixel = 1 meter (user should calibrate)

# Traffic light ROI coordinates (x1, y1, x2, y2)
# Default values - user should provide their coordinates
# TRAFFIC_LIGHT_ROI = np.array([[183, 100], [224, 100], [223, 120], [182, 120]]) #old vid
TRAFFIC_LIGHT_ROI = np.array([[500, 50], [520, 50], [520, 55], [500, 55]]) 

# Stop line coordinates (horizontal line: [x1, y1], [x2, y2])
STOP_LINE = np.array(
    [[420, 101], [536, 101]]
)
TRAFFIC_LIGHT_SEGMENT_ORDER = ("red", "yellow", "green", "unused")


def derive_segment_boxes(roi_coords):
    """
    Split a polygon/rectangle ROI into four segments.
    Leftmost segment is red, second from left is yellow.
    Returns a dict mapping segment names to rectangular boxes (x1, y1, x2, y2).
    """
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


def parse_segment_overrides(segment_string):
    """
    Parse a user supplied string with four boxes:
    "x1,y1,x2,y2;x1,y1,x2,y2;x1,y1,x2,y2;x1,y1,x2,y2" in red-yellow-green-unused order.
    """
    entries = [seg.strip() for seg in segment_string.split(";") if seg.strip()]
    if len(entries) != len(TRAFFIC_LIGHT_SEGMENT_ORDER):
        raise ValueError(
            f"Expected {len(TRAFFIC_LIGHT_SEGMENT_ORDER)} segments, got {len(entries)}"
        )

    boxes = {}
    for idx, entry in enumerate(entries):
        coords = [int(value.strip()) for value in entry.split(",")]
        if len(coords) != 4:
            raise ValueError(
                f"Segment {idx+1} must have 4 integers (x1,y1,x2,y2), got {entry}"
            )
        boxes[TRAFFIC_LIGHT_SEGMENT_ORDER[idx]] = tuple(coords)

    return boxes

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
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
    """
    Detects traffic light states by monitoring intensity changes (not color-based).
    Uses grayscale intensity change detection: a light is considered "on"
    when its ROI becomes significantly brighter than its recent baseline.
    
    Spatial logic: If both red and yellow are detected, yellow prevails if it's
    positioned more to the right (horizontal) or lower (vertical) than red.
    """

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
        self.initial_on_threshold = initial_on_threshold  # Threshold for detecting lights already on at start
        self.initialization_frames = initialization_frames
        self.frame_count = 0
        self.initialization_complete = False
        self.initial_intensities = []  # Store intensities during initialization
        self.previous_intensity = {name: None for name in self.segment_boxes}
        self.off_reference = {name: None for name in self.segment_boxes}
        self.states = {name: False for name in self.segment_boxes}

    def update_segments(self, segment_boxes):
        self.segment_boxes = segment_boxes or {}
        for name in TRAFFIC_LIGHT_SEGMENT_ORDER:
            if name not in self.segment_boxes:
                continue
            self.previous_intensity.setdefault(name, None)
            self.off_reference.setdefault(name, None)
            self.states.setdefault(name, False)

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
        if not self.segment_boxes:
            return False, False, False, "N/A", {}

        statuses = {}
        intensities = {}
        current_frame_intensities = {}
        intensity_deltas = {}  # Track intensity changes for red/yellow comparison
        
        # First pass: detect intensity changes for all segments (skip "unused")
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

            # Convert to grayscale for intensity-based change detection
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            intensity = float(np.mean(gray_region))
            intensities[name] = intensity
            current_frame_intensities[name] = intensity
            
            prev_intensity = self.previous_intensity.get(name)
            off_reference = self.off_reference.get(name)
            
            # During initialization phase: collect intensity samples
            if not self.initialization_complete:
                if off_reference is None:
                    off_reference = intensity
                self.off_reference[name] = off_reference
                intensity_deltas[name] = 0.0
                continue  # Skip state detection during initialization
            
            if off_reference is None:
                off_reference = intensity

            # Calculate intensity change from previous frame and from reference
            delta_prev = 0.0 if prev_intensity is None else intensity - prev_intensity
            delta_reference = intensity - off_reference
            # Store the maximum positive change for comparison (positive = light turning on)
            # We care about increases, not decreases, when comparing which light is more active
            max_positive_delta = max(delta_prev, delta_reference, 0.0)
            intensity_deltas[name] = max_positive_delta
            state = self.states.get(name, False)

            # Turn ON: if intensity increased significantly (change detection)
            if not state:
                if (
                    (delta_prev >= self.on_change_threshold or delta_reference >= self.on_change_threshold)
                    and intensity >= self.min_intensity
                ):
                    state = True
            else:
                # Turn OFF: if intensity decreased significantly
                if (
                    (-delta_prev) >= self.off_change_threshold
                    or intensity < off_reference + self.off_change_threshold
                ):
                    state = False

            # Update reference when off (adaptive baseline)
            if not state:
                off_reference = (off_reference * 0.9) + (intensity * 0.1)

            self.previous_intensity[name] = intensity
            self.off_reference[name] = off_reference
            self.states[name] = state
            statuses[name] = state
        
        # Handle initialization phase: detect lights that are already on
        if not self.initialization_complete:
            self.initial_intensities.append(current_frame_intensities.copy())
            self.frame_count += 1
            
            if self.frame_count >= self.initialization_frames:
                # Calculate average intensities during initialization
                avg_intensities = {}
                for name in current_frame_intensities.keys():
                    if name == "unused":
                        continue
                    samples = [frame_data.get(name, 0.0) for frame_data in self.initial_intensities if name in frame_data]
                    if samples:
                        avg_intensities[name] = sum(samples) / len(samples)
                
                # Detect lights that are already on based on absolute intensity
                for name, avg_intensity in avg_intensities.items():
                    if avg_intensity >= self.initial_on_threshold:
                        self.states[name] = True
                        statuses[name] = True
                        # Set off_reference to a lower value (estimated "off" state) 
                        # to enable change detection when it turns off later
                        estimated_off_intensity = max(avg_intensity - self.on_change_threshold * 2, self.min_intensity - 10)
                        self.off_reference[name] = estimated_off_intensity
                        self.previous_intensity[name] = avg_intensity
                    else:
                        self.states[name] = False
                        statuses[name] = False
                        # For lights that are off, use the average as the off_reference
                        self.off_reference[name] = avg_intensity
                        self.previous_intensity[name] = avg_intensity
                
                self.initialization_complete = True
                # Return the detected states after initialization
                red_on = statuses.get("red", False)
                yellow_on = statuses.get("yellow", False)
                green_on = statuses.get("green", False)
                
                # Apply change-based precedence logic (use intensity deltas from initialization)
                if red_on and yellow_on:
                    # During initialization, compare absolute intensities
                    red_intensity = avg_intensities.get("red", 0.0)
                    yellow_intensity = avg_intensities.get("yellow", 0.0)
                    
                    # Use the segment with higher intensity (more active)
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
                # Still initializing, return all False
                return False, False, False, "OFF", {}

        # Second pass: Apply change-based logic for red/yellow precedence
        red_on = statuses.get("red", False)
        yellow_on = statuses.get("yellow", False)
        green_on = statuses.get("green", False)
        
        # If both red and yellow are detected, use the one with more significant intensity changes
        if red_on and yellow_on:
            red_delta = intensity_deltas.get("red", 0.0)
            yellow_delta = intensity_deltas.get("yellow", 0.0)
            red_intensity = intensities.get("red", 0.0)
            yellow_intensity = intensities.get("yellow", 0.0)
            
            # Combine change magnitude and current intensity for a more robust comparison
            # Weight: 60% change delta, 40% current intensity (normalized)
            red_score = red_delta * 0.6 + (red_intensity / 255.0) * 100.0 * 0.4
            yellow_score = yellow_delta * 0.6 + (yellow_intensity / 255.0) * 100.0 * 0.4
            
            # Use the segment with higher score (more activity)
            if yellow_score > red_score:
                # Yellow has more significant changes/activity, use yellow
                red_on = False
            else:
                # Red has more significant changes/activity (or equal), use red
                yellow_on = False

        # Determine final status text
        if red_on:
            status_text = "RED"
        elif yellow_on:
            status_text = "YELLOW"
        elif green_on:
            status_text = "GREEN"
        else:
            status_text = "OFF"

        return red_on, yellow_on, green_on, status_text, statuses


def draw_labels_with_overlap_prevention(frame, detections, labels, color_lookup_indices, text_scale=0.5, text_thickness=1):
    """
    Draw labels with transparent background and overlap prevention.
    Uses outline text for visibility without blocking the view.
    """
    if len(detections) == 0 or len(labels) == 0 or not hasattr(detections, "xyxy"):
        return frame
    
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Get bounding boxes and calculate label positions
    label_positions = []  # List of (x, y, label, color) tuples
    used_rectangles = []  # List of (x1, y1, x2, y2) for occupied areas
    
    sample_count = min(len(detections), len(labels))
    
    for idx in range(sample_count):
        label = labels[idx]
        # Get bounding box
        bbox = detections.xyxy[idx]
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Get color
        if color_lookup_indices and idx < len(color_lookup_indices):
            color_idx = color_lookup_indices[idx]
            color = VEHICLE_COLOR_PALETTE.by_idx(color_idx)
        else:
            color = sv.Color(255, 255, 255)  # Default white
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = text_scale
        thickness = text_thickness
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label at top center of bounding box
        label_x = int((x1 + x2) / 2 - text_width / 2)
        label_y = y1 - 5  # 5 pixels above the box
        
        # If label would go off screen, move it below the box
        if label_y < text_height + 5:
            label_y = y2 + text_height + 5
        
        # Check for overlap with existing labels
        label_rect = (label_x - 2, label_y - text_height - 2, 
                     label_x + text_width + 2, label_y + baseline + 2)
        
        # Try to find a non-overlapping position
        max_attempts = 5
        for attempt in range(max_attempts):
            overlap = False
            for used_rect in used_rectangles:
                # Check if rectangles overlap
                if not (label_rect[2] < used_rect[0] or label_rect[0] > used_rect[2] or
                       label_rect[3] < used_rect[1] or label_rect[1] > used_rect[3]):
                    overlap = True
                    break
            
            if not overlap:
                break
            
            # Try alternative positions
            if attempt == 0:
                # Try right side
                label_x = x2 + 5
                label_y = int((y1 + y2) / 2)
            elif attempt == 1:
                # Try left side
                label_x = x1 - text_width - 5
                label_y = int((y1 + y2) / 2)
            elif attempt == 2:
                # Try bottom
                label_x = int((x1 + x2) / 2 - text_width / 2)
                label_y = y2 + text_height + 10
            else:
                # Skip this label if we can't find a good position
                break
            
            label_rect = (label_x - 2, label_y - text_height - 2, 
                         label_x + text_width + 2, label_y + baseline + 2)
        
        if not overlap:
            label_positions.append((label_x, label_y, label, color))
            used_rectangles.append(label_rect)
    
    # Draw labels with outline for visibility (no solid background)
    for label_x, label_y, label, color in label_positions:
        # Draw text outline (black) for better visibility
        outline_color = (0, 0, 0)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cv2.putText(annotated_frame, label, 
                              (label_x + dx, label_y + dy), 
                              font, font_scale, outline_color, thickness + 1, cv2.LINE_AA)
        
        # Draw main text in color
        cv2.putText(annotated_frame, label, 
                   (label_x, label_y), 
                   font, font_scale, color.as_bgr(), thickness, cv2.LINE_AA)
    
    return annotated_frame


def visualize_rois(frame, source_polygon, traffic_light_roi, segment_boxes=None):
    """
    Visualize the ROI polygons on a frame for verification.
    Useful for checking if coordinates are correct.
    
    Args:
        frame: Input frame
        source_polygon: SOURCE polygon for vehicle detection zone
        traffic_light_roi: Traffic light ROI (polygon or rectangle)
        segment_boxes: Optional dict of per-light rectangles
    
    Returns:
        Annotated frame with polygons drawn
    """
    annotated_frame = frame.copy()
    
    # Draw SOURCE polygon in red
    annotated_frame = sv.draw_polygon(
        scene=annotated_frame, 
        polygon=source_polygon, 
        color=sv.Color(255, 0, 0),  # Red
        thickness=1
    )
    
    # Draw traffic light ROI in green/yellow
    if traffic_light_roi is not None:
        if isinstance(traffic_light_roi, np.ndarray) and len(traffic_light_roi.shape) == 2:
            # Polygon ROI
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=traffic_light_roi,
                color=sv.Color(0, 255, 0),  # Green
                thickness=1
            )
        elif isinstance(traffic_light_roi, tuple) and len(traffic_light_roi) == 4:
            # Rectangle ROI
            x1, y1, x2, y2 = traffic_light_roi
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    
    if segment_boxes:
        color_lookup = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
        }
        for name, (x1, y1, x2, y2) in segment_boxes.items():
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color_lookup.get(name, (255, 255, 255)), 1)
            cv2.putText(
                annotated_frame,
                name.upper(),
                (int(x1), int(max(y1 - 4, 0))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_lookup.get(name, (255, 255, 255)),
                1,
                cv2.LINE_AA,
            )

    return annotated_frame


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using YOLO 11 and Supervision"
    )
    parser.add_argument(
        "--model_path",
        default="yolo11x.pt",
        help="Path to YOLO model file (default: yolo11x.pt)",
        type=str,
    )
    parser.add_argument(
        "--imgsz",
        default=1280,
        type=int,
        help="Input image size for YOLO model (default: 1280). Higher values improve accuracy but slow inference. Common values: 640, 1280",
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output). If not provided, will auto-generate from source video name.",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.4,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--csv_output_path",                     # NEW
        default=None,               # NEW
        help="Path to the CSV file with speed data. If not provided, will auto-generate from source video name.",  # NEW
        type=str,                                # NEW
    )
    parser.add_argument(
        "--traffic_light_roi",
        default=None,
        help="Traffic light ROI coordinates as 'x1,y1,x2,y2' (rectangle) or 'x1,y1,x2,y2,x3,y3,x4,y4' (polygon)",
        type=str,
    )
    parser.add_argument(
        "--traffic_light_segments",
        default=None,
        help="Override per-light rectangles as 'x1,y1,x2,y2;x1,y1,x2,y2;x1,y1,x2,y2' for red, yellow, green.",
        type=str,
    )
    parser.add_argument(
        "--visualize_first_frame",
        action="store_true",
        help="Visualize the first frame with ROI polygons drawn to verify coordinates",
    )
    parser.add_argument(
        "--segment_change_threshold",
        default=12.0,
        type=float,
        help="Minimum HSV-V delta to mark a segment as changed/on.",
    )
    parser.add_argument(
        "--segment_release_threshold",
        default=6.0,
        type=float,
        help="Delta required to declare a segment off after it was on.",
    )
    parser.add_argument(
        "--segment_min_intensity",
        default=60.0,
        type=float,
        help="Minimum HSV-V intensity to consider a traffic light illuminated.",
    )
    parser.add_argument(
        "--initial_on_threshold",
        default=80.0,
        type=float,
        help="Absolute intensity threshold for detecting lights already on at video start (default: 80.0).",
    )
    parser.add_argument(
        "--initialization_frames",
        default=10,
        type=int,
        help="Number of frames to use for initial state detection (default: 10).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Auto-generate output paths if not provided
    source_path = Path(args.source_video_path)
    source_stem = source_path.stem  # filename without extension
    source_dir = source_path.parent
    
    # Generate target video path if not provided
    if args.target_video_path is None:
        # Try to find a unique filename
        counter = 0
        while True:
            if counter == 0:
                target_filename = f"{source_stem}_result.mp4"
            else:
                target_filename = f"{source_stem}_result_{counter}.mp4"
            
            target_path = source_dir / target_filename
            if not target_path.exists():
                args.target_video_path = str(target_path)
                break
            counter += 1
        print(f"Auto-generated target video path: {args.target_video_path}")
    
    # Generate CSV output path if not provided
    if args.csv_output_path is None:
        # Try to find a unique filename
        counter = 0
        while True:
            if counter == 0:
                csv_filename = f"{source_stem}_speed_log.csv"
            else:
                csv_filename = f"{source_stem}_speed_log_{counter}.csv"
            
            csv_path = source_dir / csv_filename
            if not csv_path.exists():
                args.csv_output_path = str(csv_path)
                break
            counter += 1
        print(f"Auto-generated CSV output path: {args.csv_output_path}")

    # Parse traffic light ROI coordinates if provided, otherwise use default
    traffic_light_roi = TRAFFIC_LIGHT_ROI  # Use default polygon
    if args.traffic_light_roi:
        try:
            coords = [int(x.strip()) for x in args.traffic_light_roi.split(',')]
            if len(coords) == 4:
                # Rectangle format (x1, y1, x2, y2)
                traffic_light_roi = tuple(coords)
                print(f"Traffic light ROI set to rectangle: {traffic_light_roi}")
            elif len(coords) == 8:
                # Polygon format (x1, y1, x2, y2, x3, y3, x4, y4)
                traffic_light_roi = np.array([[coords[0], coords[1]], 
                                             [coords[2], coords[3]], 
                                             [coords[4], coords[5]], 
                                             [coords[6], coords[7]]])
                print(f"Traffic light ROI set to polygon: {traffic_light_roi}")
            else:
                print(f"Warning: Invalid traffic light ROI format. Expected 'x1,y1,x2,y2' or 'x1,y1,x2,y2,x3,y3,x4,y4', got: {args.traffic_light_roi}")
                print(f"Using default ROI: {TRAFFIC_LIGHT_ROI}")
        except ValueError as e:
            print(f"Warning: Could not parse traffic light ROI: {e}")
            print(f"Using default ROI: {TRAFFIC_LIGHT_ROI}")
    else:
        print(f"Using default traffic light ROI (polygon): {TRAFFIC_LIGHT_ROI}")

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO(args.model_path)

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    text_scale = max(0.4, text_scale * 0.7)
    
    box_annotator = sv.BoxAnnotator(
        color=VEHICLE_COLOR_PALETTE,
        thickness=thickness,
        color_lookup=sv.ColorLookup.INDEX
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    stop_line_points_topview = view_transformer.transform_points(STOP_LINE.astype(np.float32))
    stop_line_y_topview = float(np.mean(stop_line_points_topview[:, 1])) if len(stop_line_points_topview) > 0 else None

    traffic_light_segments = derive_segment_boxes(traffic_light_roi)
    if args.traffic_light_segments:
        try:
            traffic_light_segments = parse_segment_overrides(args.traffic_light_segments)
            print(f"Traffic light segments overridden: {traffic_light_segments}")
        except ValueError as exc:
            print(f"Warning: {exc}. Falling back to ROI-derived segments.")

    traffic_light_detector = TrafficLightChangeDetector(
        segment_boxes=traffic_light_segments,
        on_change_threshold=args.segment_change_threshold,
        off_change_threshold=args.segment_release_threshold,
        min_intensity=args.segment_min_intensity,
        initial_on_threshold=args.initial_on_threshold,
        initialization_frames=args.initialization_frames,
    )

    csv_rows = []
    frame_index = 0
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    vehicle_crossing_state = {}
    previous_yellow_active = False

    # Optional: Visualize first frame to verify coordinates
    if args.visualize_first_frame:
        try:
            first_frame = next(frame_generator)
            first_frame_viz = visualize_rois(first_frame, SOURCE, traffic_light_roi, traffic_light_segments)
            sv.plot_image(first_frame_viz)
            print("First frame visualization displayed. Close the window to continue processing.")
            # Reset generator
            frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)
        except StopIteration:
            print("Warning: Could not read first frame for visualization")

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            red_on, yellow_on, green_on, detected_status, segment_states = traffic_light_detector.detect(frame)
            if red_on:
                traffic_light_status = "RED"
            elif yellow_on:
                traffic_light_status = "YELLOW"
            elif green_on:
                traffic_light_status = "GREEN"
            else:
                traffic_light_status = detected_status
            
            yellow_active = traffic_light_status == "YELLOW"
            yellow_ended = previous_yellow_active and not yellow_active

            if yellow_ended:
                for state in vehicle_crossing_state.values():
                    if state.get("pending") and not state.get("crossed"):
                        state["decision"] = state.get("decision") or "stop"
                        state["pending"] = False
                        state["pending_since"] = None
            
            results = model(frame, imgsz=args.imgsz)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            
            # Filter out non-vehicle detections and small cars
            if (
                len(detections) > 0
                and hasattr(detections, "xyxy")
                and hasattr(detections, "class_id")
                and detections.class_id is not None
            ):
                keep_mask = np.ones(len(detections), dtype=bool)
                for idx in range(len(detections)):
                    cls_id = int(detections.class_id[idx])
                    
                    # Filter out persons (class_id 0) - we only track vehicles
                    if cls_id == 0:  # COCO person
                        keep_mask[idx] = False
                        continue
                    
                    # Filter out car detections with very small width; motorcycles are kept
                    if cls_id == 2:  # COCO car
                        x1, y1, x2, y2 = detections.xyxy[idx]
                        bbox_width = x2 - x1
                        if bbox_width < MIN_CAR_WIDTH_PX:
                            keep_mask[idx] = False
                            continue
                    
                    # Only keep vehicle classes we care about (car, motorcycle, bus, truck)
                    if cls_id not in VEHICLE_CLASSES:
                        keep_mask[idx] = False
                
                detections = detections[keep_mask]

            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            traffic_density = len(detections)
            vehicle_positions = {
                tracker_id: points[idx] for idx, tracker_id in enumerate(detections.tracker_id)
            }

            if traffic_light_status == "GREEN":
                phase_state = "pre_yellow"
            elif traffic_light_status == "YELLOW":
                phase_state = "yellow"
            elif traffic_light_status == "RED":
                phase_state = "red"
            else:
                phase_state = "unknown"

            labels = []
            color_lookup_indices = []
            for det_idx, tracker_id in enumerate(detections.tracker_id):
                x_curr, y_curr = points[det_idx]

                # Get vehicle type from class_id
                class_id = detections.class_id[det_idx] if hasattr(detections, 'class_id') and detections.class_id is not None else None
                # Map to vehicle type, default to "car" if class_id is not recognized
                if class_id is not None:
                    vehicle_type = VEHICLE_CLASSES.get(int(class_id), "car")  # Default to "car" if unknown
                else:
                    vehicle_type = "car"  # Default to "car" if no class_id
                color_idx = VEHICLE_COLOR_INDICES.get(vehicle_type, 0)  # Default to first color (car) if not found
                color_lookup_indices.append(color_idx)

                distance_to_stop_line_px = None
                distance_to_stop_line_m = None
                if stop_line_y_topview is not None:
                    distance_to_stop_line_px = float(y_curr - stop_line_y_topview)
                    distance_to_stop_line_m = distance_to_stop_line_px * PIXELS_TO_METERS

                state = vehicle_crossing_state.setdefault(
                    tracker_id,
                    {
                        'crossed': False,
                        'pending': False,
                        'pending_since': None,
                        'decision': '',
                    },
                )

                before_stop_line = distance_to_stop_line_px is not None and distance_to_stop_line_px > 0
                crossed_now = distance_to_stop_line_px is not None and distance_to_stop_line_px <= 0

                if yellow_active and before_stop_line and not state['crossed']:
                    if not state['pending']:
                        state['pending'] = True
                        state['pending_since'] = frame_index

                if crossed_now and not state['crossed']:
                    state['crossed'] = True
                    if state['pending']:
                        state['decision'] = 'go'
                    state['pending'] = False
                    state['pending_since'] = None

                if traffic_light_status == "RED" and state['pending'] and not state['crossed']:
                    state['decision'] = state['decision'] or 'stop'
                    state['pending'] = False
                    state['pending_since'] = None

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

                history = coordinates[tracker_id]
                speed_ms = None
                speed_kmh = None
                ttc = None
                if len(history) >= max(2, int(video_info.fps / 2)):
                    coordinate_start = history[-1]
                    coordinate_end = history[0]
                    distance_px = abs(coordinate_start - coordinate_end)
                    time_s = len(history) / video_info.fps
                    if time_s > 0:
                        speed_pixels_per_sec = distance_px / time_s
                        speed_ms = speed_pixels_per_sec * PIXELS_TO_METERS
                        speed_kmh = speed_ms * 3.6

                if (
                    speed_ms is not None
                    and distance_to_stop_line_m is not None
                    and distance_to_stop_line_m > 0
                    and speed_ms > 0
                ):
                    ttc = distance_to_stop_line_m / speed_ms

                if speed_kmh is not None:
                    labels.append(f"#{tracker_id} {vehicle_type} {int(speed_kmh)} km/h")
                else:
                    labels.append(f"#{tracker_id} {vehicle_type}")

                # log one row per detection on this frame
                csv_rows.append(
                    {
                        "frame_index": frame_index,
                        "tracker_id": int(tracker_id),
                        "vehicle_type": vehicle_type,
                        "class_id": int(class_id) if class_id is not None else "",
                        "traffic_light_status": traffic_light_status,
                        "phase_state": phase_state,
                        "speed_ms": round(speed_ms, 3) if speed_ms is not None else "",
                        "speed_kmh": round(speed_kmh, 3) if speed_kmh is not None else "",
                        "distance_to_stop_line_m": round(distance_to_stop_line_m, 3) if distance_to_stop_line_m is not None else "",
                        "distance_to_front_vehicle_m": round(distance_to_front_vehicle_m, 3) if distance_to_front_vehicle_m is not None else "",
                        "traffic_density": traffic_density,
                        "ttc_s": round(ttc, 3) if ttc is not None else "",
                        "pending_yellow_decision": int(state.get('pending', False)),
                        "yellow_light_decision": state.get('decision', '') if phase_state != "pre_yellow" else "",
                    }
                )

            annotated_frame = frame.copy()
            
            # Draw SOURCE polygon for verification
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, 
                polygon=SOURCE, 
                color=sv.Color(255, 0, 0),  # Red
                thickness=1
            )
            
            if traffic_light_roi is not None or traffic_light_segments:
                if red_on:
                    roi_color = sv.Color(255, 0, 0)
                elif yellow_on:
                    roi_color = sv.Color(255, 255, 0)
                elif green_on:
                    roi_color = sv.Color(0, 255, 0)
                else:
                    roi_color = sv.Color(128, 128, 128)

                if traffic_light_roi is not None:
                    if isinstance(traffic_light_roi, np.ndarray) and len(traffic_light_roi.shape) == 2:
                        annotated_frame = sv.draw_polygon(
                            scene=annotated_frame,
                            polygon=traffic_light_roi,
                            color=roi_color,
                            thickness=1
                        )
                    elif isinstance(traffic_light_roi, tuple) and len(traffic_light_roi) == 4:
                        x1, y1, x2, y2 = traffic_light_roi
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), roi_color.as_bgr(), 2)

                if traffic_light_segments:
                    color_lookup = {
                        "red": (0, 0, 255),
                        "yellow": (0, 255, 255),
                        "green": (0, 255, 0),
                    }
                    for name, (x1, y1, x2, y2) in traffic_light_segments.items():
                        active = segment_states.get(name, False)
                        color = color_lookup.get(name, (200, 200, 200)) if active else (120, 120, 120)
                        thickness_value = 1
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            thickness_value,
                        )
                        if active:
                            cv2.putText(
                                annotated_frame,
                                name.upper(),
                                (int(x1), int(max(y1 - 4, 0))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color,
                                1,
                                cv2.LINE_AA,
                            )

                status_text_display = f"Traffic Light: {traffic_light_status}"
                if phase_state != "unknown":
                    status_text_display += f" ({phase_state})"

                cv2.putText(
                    annotated_frame,
                    status_text_display,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    roi_color.as_bgr(),
                    2
                )
            
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            # Use custom colors for boxes based on vehicle type
            custom_color_lookup = np.array(color_lookup_indices) if color_lookup_indices else None
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections, custom_color_lookup=custom_color_lookup
            )
            # Use custom label drawing with overlap prevention and transparent background
            annotated_frame = draw_labels_with_overlap_prevention(
                annotated_frame, detections, labels, color_lookup_indices, 
                text_scale=text_scale, text_thickness=max(1, int(thickness * 0.8))
            )

            sink.write_frame(annotated_frame)
            # cv2.imshow("frame", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
            frame_index += 1
            previous_yellow_active = yellow_active
        # cv2.destroyAllWindows()
    
    fieldnames = [
        "frame_index",
        "tracker_id",
        "vehicle_type",
        "class_id",
        "traffic_light_status",
        "phase_state",
        "speed_ms",
        "speed_kmh",
        "distance_to_stop_line_m",
        "distance_to_front_vehicle_m",
        "traffic_density",
        "ttc_s",
        "pending_yellow_decision",
        "yellow_light_decision",
    ]
    os.makedirs(os.path.dirname(args.csv_output_path), exist_ok=True) if os.path.dirname(args.csv_output_path) else None

    with open(args.csv_output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
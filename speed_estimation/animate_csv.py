#!/usr/bin/env python3
"""
Animated Visualization of Vehicle Speed Log CSV

Creates an animation that scrolls through a CSV file showing:
- Vehicle positions (x, y coordinates)
- Vehicle speeds
- Traffic light status
- STOP/GO decisions
- Vehicle trajectories

Usage:
    python speed_estimation/animate_csv.py \
        --csv_path speed_estimation/logs/vehicles_2_speed_log_2.csv \
        --output_path animation.mp4
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Vehicle type colors (matching inference_refactored.py)
VEHICLE_COLORS = {
    "car": "#FF0000",          # Red
    "truck": "#00FF00",        # Green
    "bus": "#0000FF",          # Blue
    "motorcycle": "#FFFF00",   # Yellow
}

# Traffic light colors
TRAFFIC_LIGHT_COLORS = {
    "RED": "#FF0000",
    "YELLOW": "#FFFF00",
    "GREEN": "#00FF00",
    "OFF": "#808080",
}


class CSVAnimator:
    """Creates animated visualization from vehicle speed log CSV."""
    
    def __init__(
        self,
        csv_path: str,
        output_path: Optional[str] = None,
        fps: int = 10,
        show_trajectories: bool = True,
        trajectory_length: int = 20,
        figsize: Tuple[int, int] = (16, 10)
    ):
        self.csv_path = csv_path
        self.output_path = output_path
        self.fps = fps
        self.show_trajectories = show_trajectories
        self.trajectory_length = trajectory_length
        self.figsize = figsize
        
        # Load data
        print(f"Loading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Clean data
        self.df = self.df.fillna(0)
        
        # Get frame range
        self.frames = sorted(self.df['frame_index'].unique())
        self.min_frame = min(self.frames)
        self.max_frame = max(self.frames)
        
        print(f"Loaded {len(self.df)} rows, {len(self.frames)} frames")
        print(f"Frame range: {self.min_frame} to {self.max_frame}")
        
        # Get coordinate ranges for setting plot limits
        self.x_min = self.df['x'].min() - 5
        self.x_max = self.df['x'].max() + 5
        self.y_min = self.df['y'].min() - 5
        self.y_max = self.df['y'].max() + 5
        
        # Store trajectories for each vehicle
        self.trajectories: Dict[int, List[Tuple[float, float]]] = {}
        
        # Initialize figure
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='bold')
        self.ax.set_title('Vehicle Speed Log Animation', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Text elements for frame info
        self.frame_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                       fontsize=12, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Traffic light indicator
        self.traffic_light_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes,
                                                fontsize=14, fontweight='bold',
                                                verticalalignment='top', horizontalalignment='right',
                                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Vehicle annotations (will be updated each frame)
        self.vehicle_annotations = []
        self.vehicle_circles = []
        self.trajectory_lines = []
    
    def get_vehicle_color(self, vehicle_type: str) -> str:
        """Get color for vehicle type."""
        return VEHICLE_COLORS.get(vehicle_type.lower(), "#808080")
    
    def update_trajectories(self, frame_data: pd.DataFrame):
        """Update trajectories for each vehicle."""
        for _, row in frame_data.iterrows():
            tracker_id = int(row['tracker_id'])
            x, y = float(row['x']), float(row['y'])
            
            if tracker_id not in self.trajectories:
                self.trajectories[tracker_id] = []
            
            self.trajectories[tracker_id].append((x, y))
            
            # Keep only last N points
            if len(self.trajectories[tracker_id]) > self.trajectory_length:
                self.trajectories[tracker_id].pop(0)
    
    def draw_trajectories(self):
        """Draw vehicle trajectories."""
        # Clear previous trajectories
        for line in self.trajectory_lines:
            line.remove()
        self.trajectory_lines.clear()
        
        if not self.show_trajectories:
            return
        
        for tracker_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            # Get vehicle type for color (use last known type)
            vehicle_rows = self.df[self.df['tracker_id'] == tracker_id]
            if len(vehicle_rows) > 0:
                vehicle_type = vehicle_rows.iloc[-1]['vehicle_type']
                color = self.get_vehicle_color(vehicle_type)
            else:
                color = "#808080"
            
            # Draw trajectory line
            trajectory_array = np.array(trajectory)
            line = self.ax.plot(trajectory_array[:, 0], trajectory_array[:, 1],
                               color=color, alpha=0.3, linewidth=1, linestyle='--')[0]
            self.trajectory_lines.append(line)
    
    def animate_frame(self, frame_idx: int):
        """Animate a single frame."""
        frame_num = self.frames[frame_idx]
        frame_data = self.df[self.df['frame_index'] == frame_num].copy()
        
        # Clear previous annotations and circles
        for ann in self.vehicle_annotations:
            ann.remove()
        self.vehicle_annotations.clear()
        
        for circle in self.vehicle_circles:
            circle.remove()
        self.vehicle_circles.clear()
        
        # Update trajectories
        self.update_trajectories(frame_data)
        
        # Draw trajectories
        self.draw_trajectories()
        
        # Draw vehicles
        for _, row in frame_data.iterrows():
            x, y = float(row['x']), float(row['y'])
            tracker_id = int(row['tracker_id'])
            vehicle_type = str(row['vehicle_type'])
            speed_kmh = row.get('speed_kmh', 0)
            speed_ms = row.get('speed_ms', 0)
            decision = row.get('yellow_light_decision', '')
            
            # Get color
            color = self.get_vehicle_color(vehicle_type)
            
            # Draw vehicle circle
            circle = Circle((x, y), radius=3, color=color, alpha=0.7, zorder=5)
            self.ax.add_patch(circle)
            self.vehicle_circles.append(circle)
            
            # Create label
            label_parts = [f"#{tracker_id}", vehicle_type]
            if not pd.isna(speed_kmh) and speed_kmh > 0:
                label_parts.append(f"{int(speed_kmh)}km/h")
            if decision and decision != '':
                decision_symbol = "🛑" if decision.lower() == 'stop' else "🚗"
                label_parts.append(decision_symbol)
            
            label = " ".join(label_parts)
            
            # Add annotation
            ann = self.ax.annotate(
                label,
                xy=(x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                color='black'
            )
            self.vehicle_annotations.append(ann)
        
        # Update frame info
        traffic_light_status = frame_data['traffic_light_status'].iloc[0] if len(frame_data) > 0 else 'UNKNOWN'
        yellow_active = frame_data['yellow_light'].iloc[0] if len(frame_data) > 0 else False
        
        frame_info = f"Frame: {frame_num}/{self.max_frame}\n"
        frame_info += f"Vehicles: {len(frame_data)}\n"
        frame_info += f"Traffic Density: {int(frame_data['traffic_density'].iloc[0]) if len(frame_data) > 0 else 0}"
        
        self.frame_text.set_text(frame_info)
        
        # Update traffic light
        if traffic_light_status in TRAFFIC_LIGHT_COLORS:
            light_color = TRAFFIC_LIGHT_COLORS[traffic_light_status]
            self.traffic_light_text.set_text(f"🚦 {traffic_light_status}")
            self.traffic_light_text.set_bbox(dict(boxstyle='round', facecolor=light_color, alpha=0.9))
        else:
            self.traffic_light_text.set_text(f"🚦 {traffic_light_status}")
            self.traffic_light_text.set_bbox(dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add yellow light indicator
        if yellow_active:
            yellow_indicator = self.ax.text(0.5, 0.02, '⚠️ YELLOW LIGHT ACTIVE', 
                                           transform=self.ax.transAxes,
                                           fontsize=16, fontweight='bold',
                                           horizontalalignment='center',
                                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            # Remove previous yellow indicator if exists
            if hasattr(self, 'yellow_indicator'):
                self.yellow_indicator.remove()
            self.yellow_indicator = yellow_indicator
        
        return self.vehicle_circles + self.vehicle_annotations + [self.frame_text, self.traffic_light_text]
    
    def create_animation(self):
        """Create the animation."""
        print(f"Creating animation with {len(self.frames)} frames at {self.fps} fps...")
        
        anim = animation.FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=len(self.frames),
            interval=1000 / self.fps,  # milliseconds
            repeat=True,
            blit=False
        )
        
        if self.output_path:
            print(f"Saving animation to: {self.output_path}")
            # Determine format from extension
            if self.output_path.endswith('.gif'):
                anim.save(self.output_path, writer='pillow', fps=self.fps)
            else:
                # Use ffmpeg for MP4
                try:
                    Writer = animation.writers['ffmpeg']
                    writer = Writer(fps=self.fps, metadata=dict(artist='CSV Animator'), bitrate=1800)
                    anim.save(self.output_path, writer=writer)
                except Exception as e:
                    print(f"Error saving with ffmpeg: {e}")
                    print("Trying with pillow (GIF format)...")
                    gif_path = self.output_path.replace('.mp4', '.gif')
                    anim.save(gif_path, writer='pillow', fps=self.fps)
                    print(f"Saved as GIF instead: {gif_path}")
            
            print("Animation saved!")
        else:
            print("Displaying animation (close window to stop)...")
            plt.show()
        
        return anim


def main():
    parser = argparse.ArgumentParser(
        description="Animate vehicle speed log CSV file"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for animation (e.g., animation.mp4 or animation.gif). If not provided, displays interactively.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for animation (default: 10)'
    )
    parser.add_argument(
        '--no_trajectories',
        action='store_true',
        help='Disable trajectory trails'
    )
    parser.add_argument(
        '--trajectory_length',
        type=int,
        default=20,
        help='Number of points to show in trajectory trail (default: 20)'
    )
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[16, 10],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 16 10)'
    )
    
    args = parser.parse_args()
    
    # Create animator
    animator = CSVAnimator(
        csv_path=args.csv_path,
        output_path=args.output_path,
        fps=args.fps,
        show_trajectories=not args.no_trajectories,
        trajectory_length=args.trajectory_length,
        figsize=tuple(args.figsize)
    )
    
    # Create animation
    animator.create_animation()


if __name__ == '__main__':
    main()


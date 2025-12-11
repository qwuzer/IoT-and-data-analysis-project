#!/usr/bin/env python3
"""
Animated Dilemma Zone Visualization

Creates an animated visualization showing vehicle positions and decisions
as they approach the stop line, highlighting the dilemma zone.

Usage:
    python speed_estimation/animate_dilemma_zone.py \
        --csv_dir speed_estimation/logs \
        --output_path dilemma_zone_animation.mp4
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

# Vehicle type colors
VEHICLE_COLORS = {
    "car": "#FF0000",          # Red
    "truck": "#00FF00",        # Green
    "bus": "#0000FF",          # Blue
    "motorcycle": "#FFFF00",   # Yellow
}


class DilemmaZoneAnimator:
    """Creates animated visualization of dilemma zone from CSV data."""
    
    def __init__(
        self,
        csv_files: List[str],
        output_path: Optional[str] = None,
        fps: int = 10,
        figsize: Tuple[int, int] = (16, 10),
        show_trajectories: bool = True,
        trajectory_length: int = 20
    ):
        self.csv_files = csv_files
        self.output_path = output_path
        self.fps = fps
        self.figsize = figsize
        self.show_trajectories = show_trajectories
        self.trajectory_length = trajectory_length
        
        # Load data
        print(f"Loading {len(csv_files)} CSV files...")
        self.df = self.load_all_csvs(csv_files)
        
        # Filter for yellow light events (where decisions are made)
        self.yellow_data = self.df[
            (self.df['yellow_light'] == True) |
            (self.df['yellow_light_decision'].notna()) |
            (self.df['yellow_light_decision'] != '')
        ].copy()
        
        print(f"Loaded {len(self.df)} total rows")
        print(f"Yellow light events: {len(self.yellow_data)}")
        
        # Get frame range
        self.frames = sorted(self.df['frame_index'].unique())
        self.min_frame = min(self.frames)
        self.max_frame = max(self.frames)
        
        # Get coordinate ranges
        self.x_min = self.df['x'].min() - 5
        self.x_max = self.df['x'].max() + 5
        self.y_min = self.df['y'].min() - 5
        self.y_max = self.df['y'].max() + 5
        
        # Find stop line position
        stop_line_data = self.df[self.df['distance_to_stop_line'] <= 0]
        if len(stop_line_data) > 0:
            self.stop_line_y = stop_line_data['y'].median()
        else:
            self.stop_line_y = self.df['y'].min()
        
        # Store trajectories
        self.trajectories: dict = {}
        self.decisions: dict = {}  # tracker_id -> decision
        
        # Initialize figure
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Position (pixels) - Distance from Stop Line', fontsize=12, fontweight='bold')
        self.ax.set_title('Dilemma Zone Animation', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Draw stop line
        self.ax.axhline(y=self.stop_line_y, color='red', linestyle='--', 
                       linewidth=3, label='Stop Line', zorder=1, alpha=0.8)
        self.ax.text(self.x_max * 0.95, self.stop_line_y + 1, 'STOP LINE', 
                    fontsize=12, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    ha='right', va='bottom', zorder=2)
        
        # Add distance zones
        zone1_y = self.stop_line_y + 10
        zone2_y = self.stop_line_y + 30
        self.ax.axhspan(self.stop_line_y, zone1_y, alpha=0.1, color='red', zorder=0)
        self.ax.axhspan(zone1_y, zone2_y, alpha=0.1, color='yellow', zorder=0)
        self.ax.axhspan(zone2_y, self.y_max, alpha=0.1, color='green', zorder=0)
        
        # Add zone labels
        self.ax.text(self.x_min + 1, self.stop_line_y + 5, 'High Risk\n(0-10m)', 
                    fontsize=9, ha='left', va='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        self.ax.text(self.x_min + 1, self.stop_line_y + 20, 'Dilemma Zone\n(10-30m)', 
                    fontsize=9, ha='left', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        self.ax.text(self.x_min + 1, self.stop_line_y + 50, 'Safe Zone\n(30m+)', 
                    fontsize=9, ha='left', va='center',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        # Text elements
        self.frame_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                       fontsize=11, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                                       zorder=100)
        
        self.stats_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                                       zorder=100)
        
        # Vehicle elements (will be updated each frame)
        self.vehicle_circles = []
        self.vehicle_annotations = []
        self.trajectory_lines = []
    
    def load_all_csvs(self, csv_files: List[str]) -> pd.DataFrame:
        """Load and combine multiple CSV files."""
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = Path(csv_file).stem
                dataframes.append(df)
                print(f"  Loaded {len(df)} rows from {Path(csv_file).name}")
            except Exception as e:
                print(f"  Warning: Could not load {csv_file}: {e}")
                continue
        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")
        return pd.concat(dataframes, ignore_index=True)
    
    def get_vehicle_color(self, vehicle_type: str, decision: str = '') -> str:
        """Get color for vehicle type, with decision overlay."""
        base_color = VEHICLE_COLORS.get(vehicle_type.lower(), "#808080")
        if decision and decision.lower() == 'stop':
            return '#FF0000'  # Red for stop
        elif decision and decision.lower() == 'go':
            return '#00FF00'  # Green for go
        return base_color
    
    def update_trajectories(self, frame_data: pd.DataFrame):
        """Update trajectories and decisions for each vehicle."""
        for _, row in frame_data.iterrows():
            tracker_id = int(row['tracker_id'])
            x, y = float(row['x']), float(row['y'])
            decision = str(row.get('yellow_light_decision', ''))
            
            if tracker_id not in self.trajectories:
                self.trajectories[tracker_id] = []
            
            self.trajectories[tracker_id].append((x, y))
            
            if len(self.trajectories[tracker_id]) > self.trajectory_length:
                self.trajectories[tracker_id].pop(0)
            
            # Update decision if available
            if decision and decision != '' and decision != 'nan':
                self.decisions[tracker_id] = decision
    
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
            
            # Get vehicle type and decision
            vehicle_rows = self.df[self.df['tracker_id'] == tracker_id]
            if len(vehicle_rows) > 0:
                vehicle_type = vehicle_rows.iloc[-1]['vehicle_type']
                decision = self.decisions.get(tracker_id, '')
                color = self.get_vehicle_color(vehicle_type, decision)
            else:
                color = "#808080"
            
            # Draw trajectory line
            trajectory_array = np.array(trajectory)
            line = self.ax.plot(trajectory_array[:, 0], trajectory_array[:, 1],
                               color=color, alpha=0.3, linewidth=1.5, linestyle='--', zorder=3)[0]
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
        
        # Get traffic light status
        traffic_light_status = frame_data['traffic_light_status'].iloc[0] if len(frame_data) > 0 else 'UNKNOWN'
        yellow_active = frame_data['yellow_light'].iloc[0] if len(frame_data) > 0 else False
        
        # Count decisions in this frame
        stop_count = len(frame_data[frame_data['yellow_light_decision'].isin(['stop', 'STOP'])])
        go_count = len(frame_data[frame_data['yellow_light_decision'].isin(['go', 'GO'])])
        
        # Draw vehicles
        for _, row in frame_data.iterrows():
            x, y = float(row['x']), float(row['y'])
            tracker_id = int(row['tracker_id'])
            vehicle_type = str(row['vehicle_type'])
            speed_kmh = row.get('speed_kmh', 0)
            distance = row.get('distance_to_stop_line', 0)
            decision = str(row.get('yellow_light_decision', ''))
            
            # Get color based on decision
            color = self.get_vehicle_color(vehicle_type, decision)
            
            # Draw vehicle circle (larger if has decision)
            size = 40 if (decision and decision != '' and decision != 'nan') else 25
            circle = Circle((x, y), radius=size/10, color=color, alpha=0.7, zorder=5, 
                          edgecolor='black', linewidth=1.5 if decision else 1)
            self.ax.add_patch(circle)
            self.vehicle_circles.append(circle)
            
            # Create label
            label_parts = [f"#{tracker_id}", vehicle_type]
            if not pd.isna(speed_kmh) and speed_kmh > 0:
                label_parts.append(f"{int(speed_kmh)}km/h")
            if decision and decision != '' and decision != 'nan':
                decision_symbol = "🛑" if decision.lower() == 'stop' else "🚗"
                label_parts.append(decision_symbol)
            
            label = " ".join(label_parts)
            
            # Add annotation
            ann = self.ax.annotate(
                label,
                xy=(x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                         edgecolor=color, linewidth=1.5 if decision else 1),
                color='black',
                zorder=6
            )
            self.vehicle_annotations.append(ann)
        
        # Update frame info
        frame_info = f"Frame: {frame_num}/{self.max_frame}\n"
        frame_info += f"Vehicles: {len(frame_data)}\n"
        frame_info += f"Traffic Light: {traffic_light_status}"
        if yellow_active:
            frame_info += " ⚠️ YELLOW"
        self.frame_text.set_text(frame_info)
        
        # Update statistics
        stats_info = f"Stop Decisions: {stop_count}\n"
        stats_info += f"Go Decisions: {go_count}\n"
        if len(frame_data) > 0:
            avg_distance = frame_data['distance_to_stop_line'].mean()
            stats_info += f"Avg Distance: {avg_distance:.1f}m"
        self.stats_text.set_text(stats_info)
        
        return self.vehicle_circles + self.vehicle_annotations + [self.frame_text, self.stats_text]
    
    def create_animation(self):
        """Create the animation."""
        print(f"Creating animation with {len(self.frames)} frames at {self.fps} fps...")
        
        anim = animation.FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=len(self.frames),
            interval=1000 / self.fps,
            repeat=True,
            blit=False
        )
        
        if self.output_path:
            print(f"Saving animation to: {self.output_path}")
            if self.output_path.endswith('.gif'):
                anim.save(self.output_path, writer='pillow', fps=self.fps)
            else:
                try:
                    Writer = animation.writers['ffmpeg']
                    writer = Writer(fps=self.fps, metadata=dict(artist='Dilemma Zone Animator'), bitrate=1800)
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


def find_csv_files(directory: str, pattern: str = "*_speed_log*.csv") -> List[str]:
    """Find all CSV files matching pattern in directory."""
    dir_path = Path(directory)
    csv_files = sorted(dir_path.glob(pattern))
    return [str(f) for f in csv_files]


def main():
    parser = argparse.ArgumentParser(
        description="Animate dilemma zone from vehicle speed log CSV files"
    )
    parser.add_argument(
        '--csv_dir',
        type=str,
        default='speed_estimation/logs',
        help='Directory containing CSV files'
    )
    parser.add_argument(
        '--csv_files',
        type=str,
        nargs='+',
        default=None,
        help='Specific CSV files to use (overrides --csv_dir)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_speed_log*.csv',
        help='Glob pattern for CSV files'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for animation (MP4 or GIF). If not provided, displays interactively.'
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
    
    # Find CSV files
    if args.csv_files:
        csv_files = args.csv_files
    else:
        csv_files = find_csv_files(args.csv_dir, args.pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found in {args.csv_dir} matching pattern {args.pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {Path(f).name}")
    
    # Create animator
    animator = DilemmaZoneAnimator(
        csv_files=csv_files,
        output_path=args.output_path,
        fps=args.fps,
        figsize=tuple(args.figsize),
        show_trajectories=not args.no_trajectories,
        trajectory_length=args.trajectory_length
    )
    
    # Create animation
    animator.create_animation()


if __name__ == '__main__':
    main()


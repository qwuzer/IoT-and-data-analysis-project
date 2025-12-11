#!/usr/bin/env python3
"""
Dilemma Zone Heatmap Generator

Creates a heatmap visualization showing where vehicles stop relative to the stop line.
This helps identify the empirical dilemma zone from collected data.

Usage:
    python speed_estimation/dilemma_zone_heatmap.py \
        --csv_dir speed_estimation/logs \
        --output_path dilemma_zone_heatmap.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
from scipy import stats
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

# Vehicle type colors
VEHICLE_COLORS = {
    "car": "#FF0000",          # Red
    "truck": "#00FF00",        # Green
    "bus": "#0000FF",          # Blue
    "motorcycle": "#FFFF00",   # Yellow
}


class DilemmaZoneHeatmap:
    """Creates heatmap visualization of vehicle stop positions."""
    
    def __init__(
        self,
        csv_files: List[str],
        output_path: Optional[str] = None,
        grid_resolution: int = 50,
        smoothing: float = 1.0,
        figsize: Tuple[int, int] = (14, 10)
    ):
        self.csv_files = csv_files
        self.output_path = output_path
        self.grid_resolution = grid_resolution
        self.smoothing = smoothing
        self.figsize = figsize
        
        # Load and combine all CSV files
        print(f"Loading {len(csv_files)} CSV files...")
        self.df = self.load_all_csvs(csv_files)
        
        print(f"Loaded {len(self.df)} total rows")
        print(f"Unique vehicles: {self.df['tracker_id'].nunique()}")
        
        # Filter for stop decisions
        self.stop_data = self.df[
            (self.df['yellow_light_decision'] == 'stop') |
            (self.df['yellow_light_decision'] == 'STOP')
        ].copy()
        
        print(f"Stop decisions: {len(self.stop_data)}")
        print(f"Go decisions: {len(self.df) - len(self.stop_data)}")
        
        # Filter out invalid data
        self.stop_data = self.stop_data[
            (self.stop_data['distance_to_stop_line'].notna()) &
            (self.stop_data['distance_to_stop_line'] >= 0) &
            (self.stop_data['speed_ms'].notna()) &
            (self.stop_data['speed_ms'] > 0)
        ].copy()
        
        print(f"Valid stop data points: {len(self.stop_data)}")
    
    def load_all_csvs(self, csv_files: List[str]) -> pd.DataFrame:
        """Load and combine multiple CSV files."""
        dataframes = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add source file identifier
                df['source_file'] = Path(csv_file).stem
                dataframes.append(df)
                print(f"  Loaded {len(df)} rows from {Path(csv_file).name}")
            except Exception as e:
                print(f"  Warning: Could not load {csv_file}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    
    def create_position_heatmap(self) -> plt.Figure:
        """Create heatmap of stop positions in x-y space with stop line."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get stop positions
        x_positions = self.stop_data['x'].values
        y_positions = self.stop_data['y'].values
        distances = self.stop_data['distance_to_stop_line'].values
        
        # Create 2D histogram
        x_range = (self.df['x'].min() - 5, self.df['x'].max() + 5)
        y_range = (self.df['y'].min() - 5, self.df['y'].max() + 5)
        
        H, xedges, yedges = np.histogram2d(
            x_positions, y_positions,
            bins=self.grid_resolution,
            range=[x_range, y_range]
        )
        
        # Apply smoothing
        if self.smoothing > 0:
            H = gaussian_filter(H, sigma=self.smoothing)
        
        # Create heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            H.T,
            origin='lower',
            extent=extent,
            cmap='YlOrRd',
            aspect='auto',
            interpolation='bilinear'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Stop Frequency')
        cbar.set_label('Number of Stop Events', fontsize=12, rotation=270, labelpad=20)
        
        # Draw stop line (where distance_to_stop_line = 0)
        # Find approximate stop line position from data
        # Stop line is where vehicles cross (distance becomes 0 or negative)
        stop_line_y = None
        if len(self.df[self.df['distance_to_stop_line'] <= 0]) > 0:
            # Use the y position where distance becomes 0
            crossing_data = self.df[self.df['distance_to_stop_line'] <= 0]
            if len(crossing_data) > 0:
                stop_line_y = crossing_data['y'].median()
        
        # Alternative: use the minimum y where we have stop data
        if stop_line_y is None:
            stop_line_y = self.df['y'].min()
        
        # Draw stop line
        ax.axhline(y=stop_line_y, color='red', linestyle='--', linewidth=3, 
                  label='Stop Line', zorder=10, alpha=0.8)
        
        # Add text label for stop line
        ax.text(x_range[1] * 0.95, stop_line_y + 1, 'STOP LINE', 
               fontsize=12, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               ha='right', va='bottom', zorder=11)
        
        # Overlay individual stop points
        scatter = ax.scatter(
            x_positions, y_positions,
            c=distances,
            cmap='viridis',
            s=30,
            alpha=0.5,
            edgecolors='black',
            linewidths=0.5,
            label='Stop positions',
            zorder=5
        )
        
        # Add distance colorbar
        cbar2 = plt.colorbar(scatter, ax=ax, label='Distance to Stop Line (m)', pad=0.1)
        cbar2.set_label('Distance to Stop Line (m)', fontsize=12, rotation=270, labelpad=20)
        
        # Add distance zones
        if distances.max() > 0:
            # Zone 1: Very close (0-10m) - High risk
            zone1_y = stop_line_y + 10
            ax.axhspan(stop_line_y, zone1_y, alpha=0.1, color='red', label='High Risk Zone (0-10m)')
            
            # Zone 2: Medium distance (10-30m) - Dilemma zone
            zone2_y = stop_line_y + 30
            ax.axhspan(zone1_y, zone2_y, alpha=0.1, color='yellow', label='Dilemma Zone (10-30m)')
            
            # Zone 3: Far (30m+) - Low risk
            ax.axhspan(zone2_y, y_range[1], alpha=0.1, color='green', label='Safe Zone (30m+)')
        
        ax.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (pixels) - Distance from Stop Line', fontsize=12, fontweight='bold')
        ax.set_title('Dilemma Zone Heatmap: Vehicle Stop Positions', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_stop_vs_go_comparison(self) -> plt.Figure:
        """Create side-by-side comparison of STOP vs GO positions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        # Get GO data
        go_data = self.df[
            ((self.df['yellow_light_decision'] == 'go') | 
             (self.df['yellow_light_decision'] == 'GO')) &
            (self.df['distance_to_stop_line'].notna()) &
            (self.df['distance_to_stop_line'] >= 0)
        ].copy()
        
        # Get ranges
        x_range = (self.df['x'].min() - 5, self.df['x'].max() + 5)
        y_range = (self.df['y'].min() - 5, self.df['y'].max() + 5)
        
        # Find stop line position
        stop_line_y = None
        if len(self.df[self.df['distance_to_stop_line'] <= 0]) > 0:
            crossing_data = self.df[self.df['distance_to_stop_line'] <= 0]
            if len(crossing_data) > 0:
                stop_line_y = crossing_data['y'].median()
        if stop_line_y is None:
            stop_line_y = self.df['y'].min()
        
        # LEFT PLOT: STOP positions
        x_stop = self.stop_data['x'].values
        y_stop = self.stop_data['y'].values
        dist_stop = self.stop_data['distance_to_stop_line'].values
        
        H_stop, xedges, yedges = np.histogram2d(
            x_stop, y_stop,
            bins=self.grid_resolution,
            range=[x_range, y_range]
        )
        if self.smoothing > 0:
            H_stop = gaussian_filter(H_stop, sigma=self.smoothing)
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im1 = ax1.imshow(
            H_stop.T,
            origin='lower',
            extent=extent,
            cmap='Reds',
            aspect='auto',
            interpolation='bilinear'
        )
        
        # Draw stop line
        ax1.axhline(y=stop_line_y, color='red', linestyle='--', linewidth=3, zorder=10, alpha=0.8)
        ax1.text(x_range[1] * 0.95, stop_line_y + 1, 'STOP LINE', 
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                ha='right', va='bottom', zorder=11)
        
        ax1.scatter(x_stop, y_stop, c=dist_stop, cmap='viridis', s=20, 
                   alpha=0.6, edgecolors='black', linewidths=0.3, zorder=5)
        plt.colorbar(im1, ax=ax1, label='Stop Frequency')
        ax1.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='bold')
        ax1.set_title('🛑 STOP Decisions', fontsize=14, fontweight='bold', color='red')
        ax1.grid(True, alpha=0.3)
        
        # RIGHT PLOT: GO positions
        if len(go_data) > 0:
            x_go = go_data['x'].values
            y_go = go_data['y'].values
            dist_go = go_data['distance_to_stop_line'].values
            
            H_go, _, _ = np.histogram2d(
                x_go, y_go,
                bins=self.grid_resolution,
                range=[x_range, y_range]
            )
            if self.smoothing > 0:
                H_go = gaussian_filter(H_go, sigma=self.smoothing)
            
            im2 = ax2.imshow(
                H_go.T,
                origin='lower',
                extent=extent,
                cmap='Greens',
                aspect='auto',
                interpolation='bilinear'
            )
            
            # Draw stop line
            ax2.axhline(y=stop_line_y, color='red', linestyle='--', linewidth=3, zorder=10, alpha=0.8)
            ax2.text(x_range[1] * 0.95, stop_line_y + 1, 'STOP LINE', 
                    fontsize=12, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    ha='right', va='bottom', zorder=11)
            
            ax2.scatter(x_go, y_go, c=dist_go, cmap='viridis', s=20, 
                       alpha=0.6, edgecolors='black', linewidths=0.3, zorder=5)
            plt.colorbar(im2, ax=ax2, label='Go Frequency')
            ax2.set_title('🚗 GO Decisions', fontsize=14, fontweight='bold', color='green')
        else:
            ax2.text(0.5, 0.5, 'No GO decisions in data', 
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.set_title('🚗 GO Decisions', fontsize=14, fontweight='bold', color='green')
        
        ax2.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('STOP vs GO Decision Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def create_speed_distance_heatmap(self) -> plt.Figure:
        """Create heatmap of speed vs distance to stop line when vehicles stop."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get data
        speeds = self.stop_data['speed_ms'].values
        distances = self.stop_data['distance_to_stop_line'].values
        
        # Create 2D histogram
        speed_range = (0, self.stop_data['speed_ms'].max() * 1.1)
        distance_range = (0, self.stop_data['distance_to_stop_line'].max() * 1.1)
        
        H, speed_edges, dist_edges = np.histogram2d(
            speeds, distances,
            bins=self.grid_resolution,
            range=[speed_range, distance_range]
        )
        
        # Apply smoothing
        if self.smoothing > 0:
            H = gaussian_filter(H, sigma=self.smoothing)
        
        # Create heatmap
        extent = [speed_edges[0], speed_edges[-1], dist_edges[0], dist_edges[-1]]
        im = ax.imshow(
            H.T,
            origin='lower',
            extent=extent,
            cmap='RdYlGn_r',
            aspect='auto',
            interpolation='bilinear'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Stop Frequency')
        cbar.set_label('Number of Stop Events', fontsize=12, rotation=270, labelpad=20)
        
        # Overlay scatter points
        ax.scatter(
            speeds, distances,
            c='black',
            s=20,
            alpha=0.3,
            edgecolors='white',
            linewidths=0.5
        )
        
        # Add contour lines for density
        try:
            X, Y = np.meshgrid(
                (speed_edges[:-1] + speed_edges[1:]) / 2,
                (dist_edges[:-1] + dist_edges[1:]) / 2
            )
            contours = ax.contour(X, Y, H.T, levels=5, colors='black', alpha=0.5, linewidths=1)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%d')
        except:
            pass
        
        # Highlight dilemma zone (typical range: 15-45m, 5-15 m/s)
        dilemma_zone_rect = plt.Rectangle(
            (5, 15), 10, 30,
            fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Typical Dilemma Zone'
        )
        ax.add_patch(dilemma_zone_rect)
        
        ax.set_xlabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance to Stop Line (m)', fontsize=12, fontweight='bold')
        ax.set_title('Dilemma Zone: Speed vs Distance to Stop Line (Stop Events)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_distance_distribution(self) -> plt.Figure:
        """Create histogram of stop distances, separated by STOP and GO decisions."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Get GO data
        go_data = self.df[
            ((self.df['yellow_light_decision'] == 'go') | 
             (self.df['yellow_light_decision'] == 'GO')) &
            (self.df['distance_to_stop_line'].notna()) &
            (self.df['distance_to_stop_line'] >= 0)
        ].copy()
        
        stop_distances = self.stop_data['distance_to_stop_line'].values
        go_distances = go_data['distance_to_stop_line'].values if len(go_data) > 0 else np.array([])
        
        # Determine common x-axis range
        all_distances = np.concatenate([stop_distances, go_distances]) if len(go_distances) > 0 else stop_distances
        x_min = 0
        x_max = max(all_distances.max() if len(all_distances) > 0 else 0, stop_distances.max()) * 1.1
        bins = 30
        
        # TOP LEFT: STOP Histogram
        ax1 = axes[0, 0]
        if len(stop_distances) > 0:
            ax1.hist(stop_distances, bins=bins, edgecolor='black', alpha=0.7, 
                    color='red', range=(x_min, x_max))
            mean_stop = stop_distances.mean()
            median_stop = np.median(stop_distances)
            ax1.axvline(mean_stop, color='darkred', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_stop:.1f}m')
            ax1.axvline(median_stop, color='darkgreen', linestyle='--', linewidth=2, 
                       label=f'Median: {median_stop:.1f}m')
        ax1.set_xlabel('Distance to Stop Line (m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('🛑 STOP Decisions: Distance Distribution', fontsize=13, fontweight='bold', color='red')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(x_min, x_max)
        
        # TOP RIGHT: GO Histogram
        ax2 = axes[0, 1]
        if len(go_distances) > 0:
            ax2.hist(go_distances, bins=bins, edgecolor='black', alpha=0.7, 
                    color='green', range=(x_min, x_max))
            mean_go = go_distances.mean()
            median_go = np.median(go_distances)
            ax2.axvline(mean_go, color='darkred', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_go:.1f}m')
            ax2.axvline(median_go, color='darkgreen', linestyle='--', linewidth=2, 
                       label=f'Median: {median_go:.1f}m')
        else:
            ax2.text(0.5, 0.5, 'No GO decisions in data', 
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_xlabel('Distance to Stop Line (m)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('🚗 GO Decisions: Distance Distribution', fontsize=13, fontweight='bold', color='green')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(x_min, x_max)
        
        # BOTTOM LEFT: STOP KDE
        ax3 = axes[1, 0]
        if len(stop_distances) > 0:
            ax3.hist(stop_distances, bins=bins, density=True, edgecolor='black', alpha=0.5, 
                    color='red', label='Histogram', range=(x_min, x_max))
            
            # KDE curve
            try:
                kde = stats.gaussian_kde(stop_distances)
                x_range = np.linspace(x_min, x_max, 200)
                kde_values = kde(x_range)
                ax3.plot(x_range, kde_values, 'darkred', linewidth=2, label='KDE')
            except:
                pass
            
            mean_stop = stop_distances.mean()
            median_stop = np.median(stop_distances)
            ax3.axvline(mean_stop, color='darkred', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_stop:.1f}m')
            ax3.axvline(median_stop, color='darkgreen', linestyle='--', linewidth=2, 
                       label=f'Median: {median_stop:.1f}m')
        ax3.set_xlabel('Distance to Stop Line (m)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax3.set_title('🛑 STOP Decisions: Density Distribution', fontsize=13, fontweight='bold', color='red')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(x_min, x_max)
        
        # BOTTOM RIGHT: GO KDE
        ax4 = axes[1, 1]
        if len(go_distances) > 0:
            ax4.hist(go_distances, bins=bins, density=True, edgecolor='black', alpha=0.5, 
                    color='green', label='Histogram', range=(x_min, x_max))
            
            # KDE curve
            try:
                kde = stats.gaussian_kde(go_distances)
                x_range = np.linspace(x_min, x_max, 200)
                kde_values = kde(x_range)
                ax4.plot(x_range, kde_values, 'darkgreen', linewidth=2, label='KDE')
            except:
                pass
            
            mean_go = go_distances.mean()
            median_go = np.median(go_distances)
            ax4.axvline(mean_go, color='darkred', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_go:.1f}m')
            ax4.axvline(median_go, color='darkgreen', linestyle='--', linewidth=2, 
                       label=f'Median: {median_go:.1f}m')
        else:
            ax4.text(0.5, 0.5, 'No GO decisions in data', 
                    ha='center', va='center', fontsize=14, transform=ax4.transAxes)
        ax4.set_xlabel('Distance to Stop Line (m)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax4.set_title('🚗 GO Decisions: Density Distribution', fontsize=13, fontweight='bold', color='green')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(x_min, x_max)
        
        plt.suptitle('Distance Distribution: STOP vs GO Decisions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    def create_vehicle_type_analysis(self) -> plt.Figure:
        """Create analysis by vehicle type."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Stop distance by vehicle type
        ax = axes[0, 0]
        vehicle_types = self.stop_data['vehicle_type'].unique()
        data_by_type = [self.stop_data[self.stop_data['vehicle_type'] == vt]['distance_to_stop_line'].values 
                       for vt in vehicle_types]
        
        bp = ax.boxplot(data_by_type, labels=vehicle_types, patch_artist=True)
        for patch, vt in zip(bp['boxes'], vehicle_types):
            color = VEHICLE_COLORS.get(vt.lower(), '#808080')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Distance to Stop Line (m)', fontsize=12, fontweight='bold')
        ax.set_title('Stop Distance by Vehicle Type', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Stop speed by vehicle type
        ax = axes[0, 1]
        data_by_type = [self.stop_data[self.stop_data['vehicle_type'] == vt]['speed_ms'].values 
                       for vt in vehicle_types]
        
        bp = ax.boxplot(data_by_type, labels=vehicle_types, patch_artist=True)
        for patch, vt in zip(bp['boxes'], vehicle_types):
            color = VEHICLE_COLORS.get(vt.lower(), '#808080')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax.set_title('Stop Speed by Vehicle Type', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Scatter: Speed vs Distance by vehicle type
        ax = axes[1, 0]
        for vt in vehicle_types:
            vt_data = self.stop_data[self.stop_data['vehicle_type'] == vt]
            color = VEHICLE_COLORS.get(vt.lower(), '#808080')
            ax.scatter(
                vt_data['speed_ms'], vt_data['distance_to_stop_line'],
                label=vt, color=color, alpha=0.6, s=50
            )
        
        ax.set_xlabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance to Stop Line (m)', fontsize=12, fontweight='bold')
        ax.set_title('Speed vs Distance by Vehicle Type', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Stop count by vehicle type
        ax = axes[1, 1]
        stop_counts = self.stop_data['vehicle_type'].value_counts()
        colors = [VEHICLE_COLORS.get(vt.lower(), '#808080') for vt in stop_counts.index]
        
        bars = ax.bar(stop_counts.index, stop_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Stop Events', fontsize=12, fontweight='bold')
        ax.set_title('Stop Events by Vehicle Type', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_summary_statistics(self) -> str:
        """Create summary statistics text."""
        stats_text = []
        stats_text.append("=" * 80)
        stats_text.append("Dilemma Zone Analysis - Summary Statistics")
        stats_text.append("=" * 80)
        stats_text.append(f"\nTotal Stop Events: {len(self.stop_data)}")
        stats_text.append(f"Total Go Events: {len(self.df) - len(self.stop_data)}")
        
        distances = self.stop_data['distance_to_stop_line'].values
        speeds = self.stop_data['speed_ms'].values
        
        stats_text.append(f"\nStop Distance Statistics:")
        stats_text.append(f"  Mean: {distances.mean():.2f} m")
        stats_text.append(f"  Median: {np.median(distances):.2f} m")
        stats_text.append(f"  Std: {distances.std():.2f} m")
        stats_text.append(f"  Min: {distances.min():.2f} m")
        stats_text.append(f"  Max: {distances.max():.2f} m")
        stats_text.append(f"  25th percentile: {np.percentile(distances, 25):.2f} m")
        stats_text.append(f"  75th percentile: {np.percentile(distances, 75):.2f} m")
        
        stats_text.append(f"\nStop Speed Statistics:")
        stats_text.append(f"  Mean: {speeds.mean():.2f} m/s ({speeds.mean() * 3.6:.2f} km/h)")
        stats_text.append(f"  Median: {np.median(speeds):.2f} m/s ({np.median(speeds) * 3.6:.2f} km/h)")
        stats_text.append(f"  Std: {speeds.std():.2f} m/s")
        stats_text.append(f"  Min: {speeds.min():.2f} m/s ({speeds.min() * 3.6:.2f} km/h)")
        stats_text.append(f"  Max: {speeds.max():.2f} m/s ({speeds.max() * 3.6:.2f} km/h)")
        
        stats_text.append(f"\nVehicle Type Breakdown:")
        for vt in self.stop_data['vehicle_type'].value_counts().index:
            count = len(self.stop_data[self.stop_data['vehicle_type'] == vt])
            pct = (count / len(self.stop_data)) * 100
            stats_text.append(f"  {vt}: {count} ({pct:.1f}%)")
        
        stats_text.append("\n" + "=" * 80)
        
        return "\n".join(stats_text)
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\nGenerating visualizations...")
        
        # Create output directory if needed
        if self.output_path:
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            base_path = Path(self.output_path).stem
            output_dir_path = output_dir
        else:
            output_dir_path = Path(".")
            base_path = "dilemma_zone"
        
        # 1. Position heatmap
        print("  1. Creating position heatmap...")
        fig1 = self.create_position_heatmap()
        if self.output_path:
            path1 = output_dir_path / f"{base_path}_positions.png"
            fig1.savefig(path1, dpi=300, bbox_inches='tight')
            print(f"     Saved: {path1}")
        else:
            plt.show()
        plt.close(fig1)
        
        # 1b. STOP vs GO comparison
        print("  1b. Creating STOP vs GO comparison...")
        fig1b = self.create_stop_vs_go_comparison()
        if self.output_path:
            path1b = output_dir_path / f"{base_path}_stop_vs_go.png"
            fig1b.savefig(path1b, dpi=300, bbox_inches='tight')
            print(f"     Saved: {path1b}")
        else:
            plt.show()
        plt.close(fig1b)
        
        # 2. Speed-distance heatmap
        print("  2. Creating speed-distance heatmap...")
        fig2 = self.create_speed_distance_heatmap()
        if self.output_path:
            path2 = output_dir_path / f"{base_path}_speed_distance.png"
            fig2.savefig(path2, dpi=300, bbox_inches='tight')
            print(f"     Saved: {path2}")
        else:
            plt.show()
        plt.close(fig2)
        
        # 3. Distance distribution
        print("  3. Creating distance distribution...")
        fig3 = self.create_distance_distribution()
        if self.output_path:
            path3 = output_dir_path / f"{base_path}_distance_distribution.png"
            fig3.savefig(path3, dpi=300, bbox_inches='tight')
            print(f"     Saved: {path3}")
        else:
            plt.show()
        plt.close(fig3)
        
        # 4. Vehicle type analysis
        print("  4. Creating vehicle type analysis...")
        fig4 = self.create_vehicle_type_analysis()
        if self.output_path:
            path4 = output_dir_path / f"{base_path}_vehicle_types.png"
            fig4.savefig(path4, dpi=300, bbox_inches='tight')
            print(f"     Saved: {path4}")
        else:
            plt.show()
        plt.close(fig4)
        
        # 5. Summary statistics
        print("  5. Generating summary statistics...")
        stats = self.create_summary_statistics()
        print("\n" + stats)
        
        if self.output_path:
            stats_path = output_dir_path / f"{base_path}_statistics.txt"
            with open(stats_path, 'w') as f:
                f.write(stats)
            print(f"     Saved: {stats_path}")
        
        print("\nAll visualizations generated!")


def find_csv_files(directory: str, pattern: str = "*_speed_log*.csv") -> List[str]:
    """Find all CSV files matching pattern in directory."""
    dir_path = Path(directory)
    csv_files = sorted(dir_path.glob(pattern))
    return [str(f) for f in csv_files]


def main():
    parser = argparse.ArgumentParser(
        description="Generate dilemma zone heatmaps from vehicle speed log CSV files"
    )
    parser.add_argument(
        '--csv_dir',
        type=str,
        default='speed_estimation/logs',
        help='Directory containing CSV files (default: speed_estimation/logs)'
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
        help='Glob pattern for CSV files (default: *_speed_log*.csv)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='speed_estimation/outputs/dilemma_zone_heatmap.png',
        help='Output path for visualizations (default: speed_estimation/outputs/dilemma_zone_heatmap.png)'
    )
    parser.add_argument(
        '--grid_resolution',
        type=int,
        default=50,
        help='Grid resolution for heatmap (default: 50)'
    )
    parser.add_argument(
        '--smoothing',
        type=float,
        default=1.0,
        help='Gaussian smoothing sigma (default: 1.0, set to 0 to disable)'
    )
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[14, 10],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 14 10)'
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
    
    # Create heatmap generator
    generator = DilemmaZoneHeatmap(
        csv_files=csv_files,
        output_path=args.output_path,
        grid_resolution=args.grid_resolution,
        smoothing=args.smoothing,
        figsize=tuple(args.figsize)
    )
    
    # Generate all visualizations
    generator.generate_all_visualizations()


if __name__ == '__main__':
    main()


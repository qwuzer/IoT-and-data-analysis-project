"""
Training script for Dynamic Dilemma Zone Prediction Model

Main training script that loads data, builds sequences, trains model, and saves artifacts.
"""

import argparse
import os
import json
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm
import warnings

from .config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    OPTIMIZER,
    WEIGHT_DECAY,
    USE_SCHEDULER,
    SCHEDULER_TYPE,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
    TRAIN_VAL_SPLIT,
    SPLIT_BY_DATE,
    SPLIT_BY_SOURCE_FILE,
    MODEL_DIR,
    MODEL_CHECKPOINT_DIR,
    RANDOM_SEED,
    MODEL_TYPE,
    SEQUENCE_LENGTH,
    FEATURE_DIM
)
from .sequence_builder import build_sequences_from_csv, build_sequences_from_dataframe, get_sequence_statistics
from .utils import load_csv_data, load_multiple_csv_files, load_csv_files_from_directory, filter_valid_labels, split_by_date
from .model_architecture import create_model, count_parameters, print_model_summary

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for vehicle sequences.
    """
    
    def __init__(self, sequences: list, labels: list):
        """
        Args:
            sequences: List of numpy arrays, each of shape (sequence_length, feature_dim)
            labels: List of labels (0=GO, 1=STOP)
        """
        self.sequences = [torch.FloatTensor(seq) for seq in sequences]
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_data_loaders(
    train_sequences: list,
    train_labels: list,
    val_sequences: list,
    val_labels: list,
    batch_size: int = BATCH_SIZE
) -> tuple:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_sequences: Training sequences
        train_labels: Training labels
        val_sequences: Validation sequences
        val_labels: Validation labels
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SequenceDataset(train_sequences, train_labels)
    val_dataset = SequenceDataset(val_sequences, val_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # Shape: (batch_size, 1)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    normalizer_params,
    filepath,
    is_best=False
):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'normalizer_params': normalizer_params,
        'model_config': {
            'model_type': model.model_type,
            'input_dim': model.input_dim,
            'sequence_length': model.sequence_length
        }
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = Path(filepath).parent / 'best_model.pt'
        torch.save(checkpoint, best_path)


def train(
    csv_path: str = None,
    csv_paths: List[str] = None,
    csv_directory: str = None,
    csv_pattern: str = "*_speed_log*.csv",
    model_type: str = MODEL_TYPE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    output_dir: str = None,
    resume_from: str = None
):
    """
    Main training function.
    
    Args:
        csv_path: Path to single CSV file with vehicle data (mutually exclusive with csv_paths/csv_directory)
        csv_paths: List of paths to multiple CSV files (mutually exclusive with csv_path/csv_directory)
        csv_directory: Directory containing CSV files to load (mutually exclusive with csv_path/csv_paths)
        csv_pattern: Glob pattern for CSV files when using csv_directory (default: "*_speed_log*.csv")
        model_type: Type of model ('lstm' or 'cnn')
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        output_dir: Directory to save model and artifacts
        resume_from: Path to checkpoint to resume from
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = MODEL_CHECKPOINT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    
    # Determine which input method to use
    input_count = sum([csv_path is not None, csv_paths is not None, csv_directory is not None])
    if input_count == 0:
        raise ValueError("Must provide one of: csv_path, csv_paths, or csv_directory")
    if input_count > 1:
        raise ValueError("Can only provide one of: csv_path, csv_paths, or csv_directory")
    
    if csv_directory:
        df = load_csv_files_from_directory(csv_directory, pattern=csv_pattern, make_tracker_ids_unique=True)
    elif csv_paths:
        df = load_multiple_csv_files(csv_paths, make_tracker_ids_unique=True)
    else:
        df = load_csv_data(csv_path)
    
    # First, identify vehicles with valid labels
    df_valid_labels = filter_valid_labels(df)
    vehicles_with_labels = df_valid_labels['tracker_id'].unique()
    
    print(f"Found {len(vehicles_with_labels)} vehicles with valid labels")
    
    # Check if we have multiple source files (indicates multiple CSV files were combined)
    has_multiple_sources = 'source_file' in df.columns and df['source_file'].nunique() > 1
    
    # Split vehicles (not rows) into train/validation
    if has_multiple_sources and SPLIT_BY_SOURCE_FILE:
        # Split by source file to avoid data leakage between different recording sessions
        print("Splitting by source file to avoid data leakage...")
        source_files = sorted(df['source_file'].unique())
        split_idx = int(len(source_files) * TRAIN_VAL_SPLIT)
        
        train_sources = set(source_files[:split_idx])
        val_sources = set(source_files[split_idx:])
        
        print(f"Training sources ({len(train_sources)}): {sorted(train_sources)}")
        print(f"Validation sources ({len(val_sources)}): {sorted(val_sources)}")
        
        # Get vehicles from each source
        train_vehicles = set(df[df['source_file'].isin(train_sources)]['tracker_id'].unique())
        val_vehicles = set(df[df['source_file'].isin(val_sources)]['tracker_id'].unique())
        
        # Only use vehicles with valid labels
        train_vehicles = train_vehicles & set(vehicles_with_labels)
        val_vehicles = val_vehicles & set(vehicles_with_labels)
        
        # Split dataframes
        train_df = df[df['tracker_id'].isin(train_vehicles)].copy()
        val_df = df[df['tracker_id'].isin(val_vehicles)].copy()
        
    elif SPLIT_BY_DATE:
        print("Splitting vehicles by date (frame_index)...")
        # Get first frame for each vehicle to determine split
        vehicle_first_frames = {}
        for tracker_id in vehicles_with_labels:
            vehicle_data = df[df['tracker_id'] == tracker_id]
            vehicle_first_frames[tracker_id] = vehicle_data['frame_index'].min()
        
        # Sort vehicles by first frame
        sorted_vehicles = sorted(vehicle_first_frames.items(), key=lambda x: x[1])
        split_idx = int(len(sorted_vehicles) * TRAIN_VAL_SPLIT)
        
        train_vehicles = set([v[0] for v in sorted_vehicles[:split_idx]])
        val_vehicles = set([v[0] for v in sorted_vehicles[split_idx:]])
        
        # Split dataframes
        train_df = df[df['tracker_id'].isin(train_vehicles)].copy()
        val_df = df[df['tracker_id'].isin(val_vehicles)].copy()
    else:
        # Random split (fallback)
        print("Splitting vehicles randomly...")
        from sklearn.model_selection import train_test_split
        train_vehicles, val_vehicles = train_test_split(
            list(vehicles_with_labels),
            test_size=1 - TRAIN_VAL_SPLIT,
            random_state=RANDOM_SEED
        )
        train_df = df[df['tracker_id'].isin(train_vehicles)].copy()
        val_df = df[df['tracker_id'].isin(val_vehicles)].copy()
    
    print(f"\nSplit Summary:")
    print(f"  Training vehicles: {len(train_vehicles)}, Training rows: {len(train_df)}")
    print(f"  Validation vehicles: {len(val_vehicles)}, Validation rows: {len(val_df)}")
    
    # Verify no overlap between train and validation vehicles
    overlap = train_vehicles & val_vehicles
    if overlap:
        print(f"  WARNING: {len(overlap)} vehicles appear in both train and validation sets!")
    else:
        print(f"  ✓ No overlap between train and validation vehicles")
    
    # If using source files, verify no overlap
    if has_multiple_sources and 'source_file' in train_df.columns and 'source_file' in val_df.columns:
        train_sources = set(train_df['source_file'].unique())
        val_sources = set(val_df['source_file'].unique())
        source_overlap = train_sources & val_sources
        if source_overlap:
            print(f"  WARNING: {len(source_overlap)} source files appear in both train and validation sets!")
        else:
            print(f"  ✓ No overlap between train and validation source files")
    
    # Build sequences
    print("Building training sequences...")
    train_sequences, train_labels, train_norm_params = build_sequences_from_dataframe(
        train_df,
        sequence_length=SEQUENCE_LENGTH,
        normalize=True,
        fit_normalizer=True
    )
    
    print("Building validation sequences...")
    try:
        val_sequences, val_labels, _ = build_sequences_from_dataframe(
            val_df,
            sequence_length=SEQUENCE_LENGTH,
            normalize=True,
            fit_normalizer=False,  # Use training normalizer params
            normalizer_params=train_norm_params
        )
    except ValueError as e:
        print(f"Warning: Could not build validation sequences: {e}")
        print("Using training set for validation (small dataset).")
        val_sequences = train_sequences[-len(train_sequences)//5:]  # Use last 20% of training
        val_labels = train_labels[-len(train_labels)//5:]
    
    # Print statistics
    train_stats = get_sequence_statistics(train_sequences, train_labels)
    val_stats = get_sequence_statistics(val_sequences, val_labels)
    
    print("\nTraining set statistics:")
    for key, value in train_stats.items():
        print(f"  {key}: {value}")
    
    print("\nValidation set statistics:")
    for key, value in val_stats.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_sequences, train_labels,
        val_sequences, val_labels,
        batch_size=batch_size
    )
    
    # Create model
    print(f"\nCreating {model_type.upper()} model...")
    model = create_model(model_type=model_type)
    model = model.to(device)
    print_model_summary(model)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizer
    if OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")
    
    # Learning rate scheduler
    scheduler = None
    if USE_SCHEDULER:
        if SCHEDULER_TYPE == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=SCHEDULER_FACTOR,
                patience=SCHEDULER_PATIENCE,
                verbose=True
            )
        elif SCHEDULER_TYPE == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=10,
                gamma=SCHEDULER_FACTOR
            )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    # Initialize metadata (will be updated each epoch)
    # Note: train_stats and val_stats are already computed above
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            train_norm_params, checkpoint_path, is_best=is_best
        )
        
        # Save/update training metadata after each epoch (so it's available even if training is interrupted)
        metadata = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'num_epochs_trained': len(train_losses),
            'train_stats': train_stats,
            'val_stats': val_stats,
            'normalizer_params': train_norm_params,
            'model_config': {
                'model_type': model_type,
                'input_dim': FEATURE_DIM,
                'sequence_length': SEQUENCE_LENGTH,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        metadata_path = output_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Save final model and metadata
    print("\nSaving final model...")
    final_model_path = output_dir / 'final_model.pt'
    save_checkpoint(
        model, optimizer, num_epochs - 1, val_loss,
        train_norm_params, final_model_path, is_best=False
    )
    
    # Save final training metadata (already saved during training, but update with final stats)
    metadata = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'num_epochs_trained': len(train_losses),
        'train_stats': train_stats,
        'val_stats': val_stats,
        'normalizer_params': train_norm_params,
        'model_config': {
            'model_type': model_type,
            'input_dim': FEATURE_DIM,
            'sequence_length': SEQUENCE_LENGTH,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }
    
    metadata_path = output_dir / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"Best model: {output_dir / 'best_model.pt'}")
    print(f"Training metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Dilemma Zone Prediction Model')
    
    # Input data arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--csv_path',
        type=str,
        help='Path to single CSV file with vehicle data'
    )
    input_group.add_argument(
        '--csv_paths',
        type=str,
        nargs='+',
        help='List of paths to multiple CSV files with vehicle data'
    )
    input_group.add_argument(
        '--csv_directory',
        type=str,
        help='Directory containing CSV files to load'
    )
    parser.add_argument(
        '--csv_pattern',
        type=str,
        default='*_speed_log*.csv',
        help='Glob pattern for CSV files when using --csv_directory (default: "*_speed_log*.csv")'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default=MODEL_TYPE,
        choices=['lstm', 'cnn', 'LSTM', 'CNN'],
        help='Type of model (lstm or cnn)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=NUM_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save model and artifacts'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    train(
        csv_path=args.csv_path,
        csv_paths=args.csv_paths,
        csv_directory=args.csv_directory,
        csv_pattern=args.csv_pattern,
        model_type=args.model_type.lower(),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()


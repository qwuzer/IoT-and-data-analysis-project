"""
Training script for Dynamic Dilemma Zone Prediction Model

Main training script that loads data, builds sequences, trains model, and saves artifacts.
"""

import argparse
import os
import json
from pathlib import Path
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
    MODEL_DIR,
    MODEL_CHECKPOINT_DIR,
    RANDOM_SEED,
    MODEL_TYPE,
    SEQUENCE_LENGTH,
    FEATURE_DIM
)
from .sequence_builder import build_sequences_from_csv, build_sequences_from_dataframe, get_sequence_statistics
from .utils import load_csv_data, filter_valid_labels, split_by_date
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
    csv_path: str,
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
        csv_path: Path to CSV file with vehicle data
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
    df = load_csv_data(csv_path)
    
    # First, identify vehicles with valid labels
    df_valid_labels = filter_valid_labels(df)
    vehicles_with_labels = df_valid_labels['tracker_id'].unique()
    
    print(f"Found {len(vehicles_with_labels)} vehicles with valid labels")
    
    # Split vehicles (not rows) into train/validation
    if SPLIT_BY_DATE:
        print("Splitting vehicles by date...")
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
        from sklearn.model_selection import train_test_split
        train_vehicles, val_vehicles = train_test_split(
            list(vehicles_with_labels),
            test_size=1 - TRAIN_VAL_SPLIT,
            random_state=RANDOM_SEED
        )
        train_df = df[df['tracker_id'].isin(train_vehicles)].copy()
        val_df = df[df['tracker_id'].isin(val_vehicles)].copy()
    
    print(f"Training vehicles: {len(train_vehicles) if SPLIT_BY_DATE else len(train_vehicles)}, Training rows: {len(train_df)}")
    print(f"Validation vehicles: {len(val_vehicles) if SPLIT_BY_DATE else len(val_vehicles)}, Validation rows: {len(val_df)}")
    
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
    
    # Save training metadata
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


def main():
    parser = argparse.ArgumentParser(description='Train Dilemma Zone Prediction Model')
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV file with vehicle data'
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
        model_type=args.model_type.lower(),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()


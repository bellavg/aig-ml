import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import logging

from loss import TruthTableFeatureLoss, FeatureMetrics  # Updated import


def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        feature_dim: int,
        node_type_dim: int,
        clip_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: AIG Transformer model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        feature_dim: Dimension of node features
        node_type_dim: Dimension of node type encoding
        clip_grad_norm: Max norm for gradient clipping

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_metrics = {"mse": 0.0, "mae": 0.0, "rel_l2": 0.0, "r2": 0.0}
    total_tt_metrics = {"truth_table_accuracy": 0.0, "correct_bits": 0, "total_bits": 0}
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Extract predictions and ground truth
        pred_features = outputs['node_features']
        node_mask = batch.mask

        # Get features of masked nodes (only the truth table part)
        true_features = batch.y[node_mask, node_type_dim:]

        # Create padding mask for -1 values
        padding_mask = (true_features == -1)

        # Compute loss with padding handling
        loss = criterion(pred_features, true_features)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Update weights
        optimizer.step()

        # Compute metrics with padding handling
        metrics = FeatureMetrics.compute_metrics(pred_features, true_features, ignore_padding=True)

        # Calculate truth table accuracy (for binary values only)
        tt_metrics = FeatureMetrics.compute_truth_table_accuracy(pred_features, true_features)

        # Update totals
        total_loss += loss.item()
        for k, v in metrics.items():
            if k in total_metrics:
                total_metrics[k] += v

        for k, v in tt_metrics.items():
            total_tt_metrics[k] += v

        num_batches += 1

    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    # Calculate truth table accuracy
    if total_tt_metrics["total_bits"] > 0:
        tt_accuracy = total_tt_metrics["correct_bits"] / total_tt_metrics["total_bits"]
    else:
        tt_accuracy = 0.0

    avg_metrics["truth_table_accuracy"] = tt_accuracy

    return {"loss": avg_loss, "metrics": avg_metrics}


def validate(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        feature_dim: int,
        node_type_dim: int
) -> Dict[str, float]:
    """
    Validate the model on the validation set.

    Args:
        model: AIG Transformer model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to train on
        feature_dim: Dimension of node features
        node_type_dim: Dimension of node type encoding

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_metrics = {"mse": 0.0, "mae": 0.0, "rel_l2": 0.0, "r2": 0.0}
    total_tt_metrics = {"truth_table_accuracy": 0.0, "correct_bits": 0, "total_bits": 0}
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)

            # Extract predictions and ground truth
            pred_features = outputs['node_features']
            node_mask = batch.mask

            # Get features of masked nodes (only the truth table part)
            true_features = batch.y[node_mask, node_type_dim:]

            # Create padding mask for -1 values
            padding_mask = (true_features == -1)

            # Compute loss with padding handling
            loss = criterion(pred_features, true_features)

            # Compute metrics with padding handling
            metrics = FeatureMetrics.compute_metrics(pred_features, true_features, ignore_padding=True)

            # Calculate truth table accuracy (for binary values only)
            tt_metrics = FeatureMetrics.compute_truth_table_accuracy(pred_features, true_features)

            # Update totals
            total_loss += loss.item()
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v

            for k, v in tt_metrics.items():
                total_tt_metrics[k] += v

            num_batches += 1

    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    # Calculate truth table accuracy
    if total_tt_metrics["total_bits"] > 0:
        tt_accuracy = total_tt_metrics["correct_bits"] / total_tt_metrics["total_bits"]
    else:
        tt_accuracy = 0.0

    avg_metrics["truth_table_accuracy"] = tt_accuracy

    return {"loss": avg_loss, "metrics": avg_metrics}


def train(
        model: nn.Module,
        train_dataset,
        val_dataset,
        feature_dim: int = 256,  # Updated default for truth table size
        node_type_dim: int = 3,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        l1_weight: float = 0.1,
        feature_normalization: bool = False,  # Changed default for truth tables
        binary_loss_weight: float = 2.0,  # Added for truth table binary values
        val_interval: int = 5,
        patience: int = 10,
        clip_grad_norm: float = 1.0,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./model_checkpoints",
        log_dir: str = "./logs"
) -> Dict[str, Any]:
    """
    Train the AIG transformer model.

    Args:
        model: AIG Transformer model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        feature_dim: Dimension of node features
        node_type_dim: Dimension of node type encoding
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
        learning_rate: Initial learning rate
        weight_decay: Weight decay for L2 regularization
        l1_weight: Weight for L1 loss component
        feature_normalization: Whether to normalize features
        binary_loss_weight: Weight for binary classification loss
        val_interval: Validate every N epochs
        patience: Patience for early stopping
        clip_grad_norm: Max norm for gradient clipping
        scheduler_factor: Factor for learning rate scheduler
        scheduler_patience: Patience for learning rate scheduler
        device: Device to train on
        save_dir: Directory to save model checkpoints
        log_dir: Directory to save logs and plots

    Returns:
        Dictionary with training results
    """
    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "training.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Add console handler to see logs in stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info(f"Starting training with {num_epochs} epochs")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Move model to device
    device = torch.device(device)
    model = model.to(device)

    # Initialize loss function
    criterion = TruthTableFeatureLoss(
        l1_weight=l1_weight,
        feature_normalization=feature_normalization,
        ignore_padding=True,
        binary_loss_weight=binary_loss_weight
    ).to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'lr': []
    }

    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    no_improve_count = 0

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train for one epoch
        train_results = train_epoch(
            model, train_loader, optimizer, criterion,
            device, feature_dim, node_type_dim, clip_grad_norm
        )

        # Update history with training results
        history['train_loss'].append(train_results['loss'])
        history['train_metrics'].append(train_results['metrics'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Log training metrics
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_results['loss']:.6f}, "
                    f"MSE: {train_results['metrics']['mse']:.6f}, "
                    f"R²: {train_results['metrics']['r2']:.6f}, "
                    f"TT Accuracy: {train_results['metrics'].get('truth_table_accuracy', 0):.6f}")

        # Validate every val_interval epochs
        if (epoch + 1) % val_interval == 0 or epoch == num_epochs - 1:
            val_results = validate(
                model, val_loader, criterion, device, feature_dim, node_type_dim
            )

            # Update history with validation results
            history['val_loss'].append(val_results['loss'])
            history['val_metrics'].append(val_results['metrics'])

            # Log validation metrics
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Validation Loss: {val_results['loss']:.6f}, "
                        f"MSE: {val_results['metrics']['mse']:.6f}, "
                        f"R²: {val_results['metrics']['r2']:.6f}, "
                        f"TT Accuracy: {val_results['metrics'].get('truth_table_accuracy', 0):.6f}")

            # Update learning rate scheduler
            scheduler.step(val_results['loss'])

            # Check for improvement
            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_results['loss'],
                    'val_metrics': val_results['metrics']
                }
                best_epoch = epoch
                no_improve_count = 0

                # Save best model
                torch.save(best_model_state, os.path.join(save_dir, 'best_model.pt'))
                logger.info(f"Saved new best model with validation loss: {best_val_loss:.6f}")
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1} after {patience} epochs without improvement")
                break

        # Log epoch time
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch time: {epoch_time:.2f}s")

        # Periodically save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    # Training completed
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.2f} minutes")
    logger.info(f"Best model at epoch {best_epoch + 1} with validation loss: {best_val_loss:.6f}")

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, os.path.join(save_dir, 'final_model.pt'))

    # Plot training curves
    #plot_training_history(history, val_interval, os.path.join(log_dir, 'plots'))

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_model_path': os.path.join(save_dir, 'best_model.pt'),
        'final_model_path': os.path.join(save_dir, 'final_model.pt')
    }


def plot_training_history(history, val_interval, save_dir):
    """
    Plot and save training history.

    Args:
        history: Dictionary with training history
        val_interval: Validation interval (for x-axis alignment)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create epoch indices for plotting
    train_epochs = list(range(1, len(history['train_loss']) + 1))
    val_epochs = list(range(val_interval, len(history['train_loss']) + 1, val_interval))
    if len(val_epochs) < len(history['val_loss']):
        # Add the final epoch if it wasn't exactly on the interval
        val_epochs.append(len(history['train_loss']))

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(val_epochs[:len(history['val_loss'])], history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # Plot MSE
    plt.figure(figsize=(10, 6))
    train_mse = [metrics['mse'] for metrics in history['train_metrics']]
    val_mse = [metrics['mse'] for metrics in history['val_metrics']]
    plt.plot(train_epochs, train_mse, 'b-', label='Training MSE')
    plt.plot(val_epochs[:len(val_mse)], val_mse, 'r-', label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'mse_curve.png'))
    plt.close()

    # Plot R-squared
    plt.figure(figsize=(10, 6))
    train_r2 = [metrics['r2'] for metrics in history['train_metrics']]
    val_r2 = [metrics['r2'] for metrics in history['val_metrics']]
    plt.plot(train_epochs, train_r2, 'b-', label='Training R²')
    plt.plot(val_epochs[:len(val_r2)], val_r2, 'r-', label='Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('Training and Validation R²')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'r2_curve.png'))
    plt.close()

    # Plot Truth Table Accuracy
    if 'truth_table_accuracy' in history['train_metrics'][0]:
        plt.figure(figsize=(10, 6))
        train_tt_acc = [metrics.get('truth_table_accuracy', 0) for metrics in history['train_metrics']]
        val_tt_acc = [metrics.get('truth_table_accuracy', 0) for metrics in history['val_metrics']]
        plt.plot(train_epochs, train_tt_acc, 'b-', label='Training TT Accuracy')
        plt.plot(val_epochs[:len(val_tt_acc)], val_tt_acc, 'r-', label='Validation TT Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Truth Table Prediction Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'tt_accuracy_curve.png'))
        plt.close()

    # Plot learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
    plt.close()
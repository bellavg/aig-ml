

from collections import defaultdict
import torch
from typing import Dict, Any
from torch_geometric.data import Batch
from torch.optim import Optimizer
import torch.nn as nn


def train_epoch(args: Any, model: nn.Module, train_loader: Any,
                optimizer: Optimizer, device: torch.device,
                mask_value: float = -1.0) -> Dict[str, float]:
    """
    Run one training epoch with support for the three masking modes:
    1. "node_feature": Mask node features and predict them
    2. "edge_feature": Mask edge features and predict them
    3. "connectivity": Mask edges and predict both existence and features

    Args:
        args: Command line arguments including mask_prob and mask_mode
        model: AIGTransformer model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        mask_value: Special value to use for masking (defaults to -1.0)

    Returns:
        epoch_losses: Dictionary of average losses for this epoch
    """
    model.train()
    epoch_losses = defaultdict(float)

    # Determine masking mode from args
    mask_mode = args.mask_mode if hasattr(args, 'mask_mode') else "node_feature"

    # Get masking probability from args
    mask_prob = args.mask_prob if hasattr(args, 'mask_prob') else 0.20

    # Keep track of processed batches
    batch_count = 0

    # Import modules only once outside the loop
    from masking import create_masked_batch
    from loss import compute_loss

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        # Create masked batch with the special mask_value
        masked_batch = create_masked_batch(
            batch,
            mp=mask_prob,
            mask_mode=mask_mode,
            mask_value=mask_value
        )

        # Forward pass
        predictions = model(masked_batch)

        # Prepare target dictionary with all necessary information
        targets = {
            'x_target': masked_batch.x_target,
            'edge_index_target': masked_batch.edge_index_target,
            'edge_attr_target': masked_batch.edge_attr_target if hasattr(masked_batch, 'edge_attr_target') else None,
            'mask_mode': mask_mode
        }

        # Add masks and original values if they exist
        if hasattr(masked_batch, 'node_mask'):
            targets['node_mask'] = masked_batch.node_mask

        if hasattr(masked_batch, 'edge_mask'):
            targets['edge_mask'] = masked_batch.edge_mask

        # Add the special mask value and truth table information for node feature prediction
        if mask_mode == "node_feature" and hasattr(masked_batch, 'node_mask_value'):
            targets['mask_value'] = masked_batch.node_mask_value

        if mask_mode == "node_feature" and hasattr(masked_batch, 'truth_table_idx'):
            targets['truth_table_idx'] = masked_batch.truth_table_idx

        if mask_mode == "node_feature" and hasattr(masked_batch, 'original_truth_table_values'):
            targets['original_truth_table_values'] = masked_batch.original_truth_table_values

        # Compute loss with awareness of the masking mode
        loss, loss_dict = compute_loss(predictions, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping BEFORE optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        for key, value in loss_dict.items():
            epoch_losses[key] += value.item()

        # Increment batch counter
        batch_count += 1

        # Print progress every 10 batches
        if batch_idx % 100 == 0:
            print(f"Processed batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Average losses over all batches
    if batch_count > 0:  # Avoid division by zero
        for key in epoch_losses:
            epoch_losses[key] /= batch_count
    else:
        print("Warning: No batches were successfully processed in this epoch")

    return epoch_losses
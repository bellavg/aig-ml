from collections import defaultdict
import torch
from masking import create_masked_batch  # Import your enhanced masking function


def train_epoch(args, model, train_loader, optimizer, device):
    """
    Run one training epoch with support for gate masking.

    Args:
        args: Command line arguments including mask_prob and gate_masking flag
        model: AIGTransformer model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)

    Returns:
        epoch_losses: Dictionary of average losses for this epoch
    """
    model.train()
    epoch_losses = defaultdict(float)

    for batch in train_loader:
        batch = batch.to(device)

        # Create masked version of the batch using the specified probability and gate masking flag
        masked_batch = create_masked_batch(
            batch,
            mp=args.mask_prob,
            gate_masking=args.gate_masking
        )

        # Forward pass
        predictions = model(masked_batch)

        # Compute loss with awareness of both node and edge masking
        loss, loss_dict = model.compute_loss(
            predictions,
            {
                'node_features': masked_batch.x_target,
                'edge_index': masked_batch.edge_index_target,
                'edge_attr': masked_batch.edge_attr_target if hasattr(masked_batch, 'edge_attr_target') else None,
                'node_mask': masked_batch.node_mask,
                'edge_mask': masked_batch.edge_mask,
                'gate_masking': masked_batch.gate_masking
            }
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        for key, value in loss_dict.items():
            epoch_losses[key] += value.item()

    # Average losses over all batches
    for key in epoch_losses:
        epoch_losses[key] /= len(train_loader)

    return epoch_losses


from collections import defaultdict
import torch
from masking import create_masked_batch  # Import your enhanced masking function
from loss import compute_loss

from collections import defaultdict
import torch
from masking import create_masked_batch  # Import your enhanced masking function
from loss import compute_loss


def train_epoch(args, model, train_loader, optimizer, device):
    """
    Run one training epoch with support for all five masking modes.

    Args:
        args: Command line arguments including mask_prob and mask_mode
        model: AIGTransformer model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)

    Returns:
        epoch_losses: Dictionary of average losses for this epoch
    """
    model.train()
    epoch_losses = defaultdict(float)

    # Determine masking mode from args
    mask_mode = args.mask_mode if hasattr(args, 'mask_mode') else "node_feature"

    # For backward compatibility
    if hasattr(args, 'gate_masking') and args.gate_masking:
        mask_mode = "gate"

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        try:
            # Create masked version of the batch using the specified probability and masking mode
            masked_batch = create_masked_batch(
                batch,
                mp=args.mask_prob,
                mask_mode=mask_mode
            )

            # Forward pass
            predictions = model(masked_batch)

            # Prepare base target dictionary with common information
            targets = {
                'x_target': masked_batch.x_target,
                'edge_index_target': masked_batch.edge_index_target,
                'edge_attr_target': masked_batch.edge_attr_target if hasattr(masked_batch,
                                                                             'edge_attr_target') else None,
                'mask_mode': masked_batch.mask_mode
            }

            # Add masks if they exist
            if hasattr(masked_batch, 'node_mask'):
                targets['node_mask'] = masked_batch.node_mask
            if hasattr(masked_batch, 'edge_mask'):
                targets['edge_mask'] = masked_batch.edge_mask

            # Add mode-specific information to targets
            if mask_mode == "node_existence" and hasattr(masked_batch, 'node_existence_mask'):
                targets['node_existence_mask'] = masked_batch.node_existence_mask
                targets['node_existence_target'] = masked_batch.node_existence_target if hasattr(masked_batch,
                                                                                                 'node_existence_target') else None

            elif mask_mode == "edge_existence" and hasattr(masked_batch, 'edge_existence_mask'):
                targets['edge_existence_mask'] = masked_batch.edge_existence_mask
                targets['edge_existence_target'] = masked_batch.edge_existence_target if hasattr(masked_batch,
                                                                                                 'edge_existence_target') else None

            elif mask_mode == "removal" and hasattr(masked_batch, 'node_removal_mask'):
                targets['node_removal_mask'] = masked_batch.node_removal_mask
                targets['num_original_nodes'] = masked_batch.num_original_nodes if hasattr(masked_batch,
                                                                                           'num_original_nodes') else None
                targets['original_to_new_indices'] = masked_batch.original_to_new_indices if hasattr(masked_batch,
                                                                                                     'original_to_new_indices') else None

                # Add existence targets for removal mode
                if hasattr(masked_batch, 'node_existence_target'):
                    targets['node_existence_target'] = masked_batch.node_existence_target
                if hasattr(masked_batch, 'edge_existence_target'):
                    targets['edge_existence_target'] = masked_batch.edge_existence_target

            # Compute loss with awareness of the masking mode
            loss, loss_dict = compute_loss(predictions, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"Skipping this batch and continuing...")
            continue

    # Average losses over all batches
    batch_count = len(train_loader)
    if batch_count > 0:  # Avoid division by zero
        for key in epoch_losses:
            epoch_losses[key] /= batch_count

    return epoch_losses
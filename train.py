from collections import defaultdict
import torch
from masking import create_masked_batch
from loss import compute_loss


def train_epoch(args, model, train_loader, optimizer, device):
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

    Returns:
        epoch_losses: Dictionary of average losses for this epoch
    """
    model.train()
    epoch_losses = defaultdict(float)

    # Determine masking mode from args
    mask_mode = args.mask_mode if hasattr(args, 'mask_mode') else "node_feature"

    # For backward compatibility with old configs
    if hasattr(args, 'gate_masking') and args.gate_masking:
        print("Warning: 'gate_masking' is deprecated. Using 'node_feature' mode instead.")
        mask_mode = "node_feature"

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
            if mask_mode == "connectivity":
                # Add connectivity-specific target information
                if hasattr(masked_batch, 'all_candidate_pairs'):
                    targets['all_candidate_pairs'] = masked_batch.all_candidate_pairs
                if hasattr(masked_batch, 'all_candidate_targets'):
                    targets['all_candidate_targets'] = masked_batch.all_candidate_targets
                if hasattr(masked_batch, 'connectivity_target'):
                    targets['connectivity_target'] = masked_batch.connectivity_target
                if hasattr(masked_batch, 'masked_edge_attr_target'):
                    targets['masked_edge_attr_target'] = masked_batch.masked_edge_attr_target

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


        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()  # Print the full stack trace
            print(f"Skipping this batch and continuing...")
            continue

    # Average losses over all batches
    batch_count = len(train_loader)
    if batch_count > 0:  # Avoid division by zero
        for key in epoch_losses:
            epoch_losses[key] /= batch_count

    return epoch_losses
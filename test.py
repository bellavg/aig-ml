import torch
from collections import defaultdict
from masking import create_masked_batch
from loss import compute_loss
from prediction import reconstruct_predictions

# Fix for the edge_feature mode metrics calculation in test.py

def validate(args, model, val_loader, device):
    """
    Run validation with support for the three masking modes:
    1. "node_feature": Mask node features and predict them
    2. "edge_feature": Mask edge features and predict them
    3. "connectivity": Mask edges and predict both existence and features

    Args:
        args: Command line arguments including mask_prob and mask_mode
        model: AIGTransformer model
        val_loader: DataLoader for validation data
        device: Device to validate on (cuda/cpu)

    Returns:
        val_losses: Dictionary of average losses for validation
        metrics: Dictionary of validation metrics
    """
    model.eval()
    val_losses = defaultdict(float)
    metrics = defaultdict(float)

    # Determine masking mode from args
    mask_mode = args.mask_mode if hasattr(args, 'mask_mode') else "node_feature"

    # For backward compatibility with old configs
    if hasattr(args, 'gate_masking') and args.gate_masking:
        print("Warning: 'gate_masking' is deprecated. Using 'node_feature' mode instead.")
        mask_mode = "node_feature"

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Create masked version for validation
            masked_batch = create_masked_batch(
                batch,
                mp=args.mask_prob,
                mask_mode=mask_mode
            )

            # Forward pass
            predictions = model(masked_batch)

            # Forward pass
            predictions = model(masked_batch)
            # Prepare base target dictionary with common information
            targets = {
                'x_target': masked_batch.x_target,
                'edge_index_target': masked_batch.edge_index_target,
                'edge_attr_target': masked_batch.edge_attr_target if hasattr(masked_batch,
                                                                             'edge_attr_target') else None,
                'node_mask': masked_batch.node_mask if hasattr(masked_batch, 'node_mask') else None,
                'edge_mask': masked_batch.edge_mask if hasattr(masked_batch, 'edge_mask') else None,
                'mask_mode': mask_mode  # Use the mask_mode from args explicitly
            }

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

            # Compute validation loss
            loss, loss_dict = compute_loss(predictions, targets)

            # Accumulate losses
            for key, value in loss_dict.items():
                val_losses[key] += value.item()

            # For evaluation metrics, reconstruct the predictions to match original graph
            full_predictions = reconstruct_predictions(predictions, targets)

            # Compute metrics based on the masking mode
            if mask_mode == "node_feature":
                # Node feature prediction accuracy
                if 'node_features' in full_predictions and 'node_mask' in targets and targets[
                    'node_mask'] is not None and targets['node_mask'].sum() > 0:
                    pred_node_features = torch.sigmoid(full_predictions['node_features'])
                    pred_labels = (pred_node_features > 0.5).float()
                    node_acc = (pred_labels[targets['node_mask']] == targets['x_target'][
                        targets['node_mask']]).float().mean()
                    metrics['node_accuracy'] += node_acc.item()

            elif mask_mode == "edge_feature":
                # Edge feature prediction accuracy
                # FIXED: Check directly for 'full_edge_features' as that's what reconstruct_predictions returns
                if 'full_edge_features' in full_predictions and 'edge_mask' in targets and targets[
                    'edge_mask'] is not None and targets['edge_mask'].sum() > 0:
                    pred_edge_features = torch.sigmoid(full_predictions['full_edge_features'])
                    pred_labels = (pred_edge_features > 0.5).float()

                    # Calculate accuracy on masked edges
                    edge_acc = (pred_labels[targets['edge_mask']] == targets['edge_attr_target'][
                        targets['edge_mask']]).float().mean()
                    metrics['edge_feature_accuracy'] += edge_acc.item()

            elif mask_mode == "connectivity":
                # Both edge existence and feature prediction

                # Edge existence accuracy
                if 'edge_existence_preds' in full_predictions and 'all_candidate_targets' in targets:
                    pred_existence = (torch.sigmoid(full_predictions['edge_existence_preds']) > 0.5).float()
                    existence_acc = (pred_existence.squeeze() == targets['all_candidate_targets']).float().mean()
                    metrics['edge_existence_accuracy'] += existence_acc.item()

                # Edge feature accuracy (only for existing edges)
                if 'edge_feature_preds' in full_predictions and 'masked_edge_attr_target' in targets:
                    pred_features = torch.sigmoid(full_predictions['edge_feature_preds'])
                    pred_labels = (pred_features > 0.5).float()

                    # This is a bit tricky - we only care about feature accuracy for edges that exist
                    # For simplicity, we'll calculate on all masked edges
                    feature_acc = (pred_labels == targets['masked_edge_attr_target']).float().mean()
                    metrics['edge_feature_accuracy'] += feature_acc.item()

    # Average losses and metrics
    for key in val_losses:
        val_losses[key] /= max(1, len(val_loader))

    for key in metrics:
        metrics[key] /= max(1, len(val_loader))

    return val_losses, metrics
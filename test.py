import torch
from collections import defaultdict
from masking import create_masked_batch
from loss import compute_loss
from prediction import reconstruct_predictions


def validate(args, model, val_loader, device, mask_value: float = -1.0):
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
        mask_value: Special value to use for masking (defaults to -1.0)

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
                mask_mode=mask_mode,
                mask_value=mask_value
            )

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
            if mask_mode == "node_feature":
                # Add original truth table values if available
                if hasattr(masked_batch, 'original_truth_table_values'):
                    targets['original_truth_table_values'] = masked_batch.original_truth_table_values
                # Add truth table index if available
                if hasattr(masked_batch, 'truth_table_idx'):
                    targets['truth_table_idx'] = masked_batch.truth_table_idx
                # Add mask value if available
                if hasattr(masked_batch, 'node_mask_value'):
                    targets['node_mask_value'] = masked_batch.node_mask_value

            elif mask_mode == "edge_feature":
                # Add original edge attributes for edge feature prediction
                if hasattr(masked_batch, 'original_edge_attr'):
                    targets['original_edge_attr'] = masked_batch.original_edge_attr
                # Add edge mask value for edge feature prediction
                if hasattr(masked_batch, 'edge_mask_value'):
                    targets['edge_mask_value'] = masked_batch.edge_mask_value

            elif mask_mode == "connectivity":
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
                # Node truth table value prediction accuracy
                if 'node_features' in full_predictions and 'node_mask' in targets and targets['node_mask'].sum() > 0:
                    # Get truth table index (default is 3)
                    tt_idx = targets.get('truth_table_idx', 3)

                    # Extract predictions - handle different prediction formats
                    pred = full_predictions['node_features']

                    if pred.dim() > 1 and pred.size(1) > tt_idx:
                        # If predictions contain all features, just extract truth table values
                        pred_values = torch.sigmoid(pred[targets['node_mask'], tt_idx])
                    else:
                        # If predictions are specific to masked nodes
                        pred_values = torch.sigmoid(pred[targets['node_mask']])

                    # Get target values
                    if 'original_truth_table_values' in targets:
                        target_values = targets['original_truth_table_values']
                    else:
                        target_values = targets['x_target'][targets['node_mask'], tt_idx]

                    # Calculate accuracy
                    pred_labels = (pred_values > 0.5).float()
                    truth_table_acc = (pred_labels == target_values).float().mean()
                    metrics['truth_table_accuracy'] += truth_table_acc.item()

            elif mask_mode == "edge_feature":
                # Edge feature prediction accuracy
                if 'edge_attr_pred' in predictions and 'original_edge_attr' in targets:
                    # Get predictions for masked edges
                    edge_pred = predictions['edge_attr_pred']

                    # Get original edge attributes
                    target_edges = targets['original_edge_attr']

                    # Calculate accuracy
                    pred_labels = (torch.sigmoid(edge_pred) > 0.5).float()
                    edge_acc = (pred_labels == target_edges).all(dim=1).float().mean()
                    metrics['edge_feature_accuracy'] += edge_acc.item()

                    # Calculate per-class accuracy if we have at least 2 classes
                    if target_edges.size(1) >= 2:
                        inv_accuracy = (pred_labels[:, 0] == target_edges[:, 0]).float().mean()
                        reg_accuracy = (pred_labels[:, 1] == target_edges[:, 1]).float().mean()

                        metrics['inv_edge_accuracy'] += inv_accuracy.item()
                        metrics['reg_edge_accuracy'] += reg_accuracy.item()

                # Fallback to using full_edge_features if available
                elif 'full_edge_features' in full_predictions and 'edge_mask' in targets and targets[
                    'edge_mask'].sum() > 0:
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

                    # Only evaluate feature accuracy for positive examples (actual edges)
                    if 'all_candidate_targets' in targets:
                        # Get only positive examples (real edges that should exist)
                        positive_mask = targets['all_candidate_targets'].squeeze(-1) > 0.5

                        # Get feature predictions only for positive examples
                        positive_pred_features = pred_features[positive_mask]

                        # Apply threshold to get binary predictions
                        positive_pred_labels = (positive_pred_features > 0.5).float()

                        # Use only the first N predictions matching the target size
                        valid_count = min(positive_pred_labels.size(0), targets['masked_edge_attr_target'].size(0))

                        if valid_count > 0:
                            feature_acc = (positive_pred_labels[:valid_count] ==
                                           targets['masked_edge_attr_target'][:valid_count]).float().mean()
                            metrics['edge_feature_accuracy'] += feature_acc.item()

    # Average losses and metrics
    for key in val_losses:
        val_losses[key] /= max(1, len(val_loader))

    for key in metrics:
        metrics[key] /= max(1, len(val_loader))

    return val_losses, metrics



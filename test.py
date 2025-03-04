import argparse
from collections import defaultdict
import torch
# Import your custom modules
from masking import create_masked_batch
from loss import compute_loss

def validate(args, model, val_loader, device):
    """
    Run validation with support for all five masking modes.

    Args:
        args: Command line arguments including mask_prob and mask_mode
        model: AIGTransformer model
        val_loader: DataLoader for validation data
        device: Device to validate on (cuda/cpu)

    Returns:
        val_losses: Dictionary of average losses for validation
        metrics: Dictionary of validation metrics
    """
    from model import reconstruct_predictions  # Import the reconstruction function

    model.eval()
    val_losses = defaultdict(float)
    metrics = defaultdict(float)

    # Determine masking mode from args
    mask_mode = args.mask_mode if hasattr(args, 'mask_mode') else "node_feature"

    # For backward compatibility
    if hasattr(args, 'gate_masking') and args.gate_masking:
        mask_mode = "gate"

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

            # Prepare base target dictionary with common information
            targets = {
                'x_target': masked_batch.x_target,
                'edge_index_target': masked_batch.edge_index_target,
                'edge_attr_target': masked_batch.edge_attr_target if hasattr(masked_batch,
                                                                             'edge_attr_target') else None,
                'node_mask': masked_batch.node_mask,
                'edge_mask': masked_batch.edge_mask if hasattr(masked_batch, 'edge_mask') else None,
                'mask_mode': masked_batch.mask_mode
            }

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

            # Compute validation loss
            loss, loss_dict = compute_loss(predictions, targets)

            # Accumulate losses
            for key, value in loss_dict.items():
                val_losses[key] += value.item()

            # For evaluation metrics, reconstruct the predictions to match original graph
            full_predictions = reconstruct_predictions(predictions, targets)

            # Compute metrics based on the masking mode
            if mask_mode in ["node_feature", "node_existence", "gate"]:
                # Node feature prediction accuracy
                if 'node_features' in full_predictions and 'node_mask' in targets and targets['node_mask'].sum() > 0:
                    pred_node_features = torch.sigmoid(full_predictions['node_features'])
                    pred_labels = (pred_node_features > 0.5).float()
                    node_acc = (pred_labels[targets['node_mask']] == targets['x_target'][
                        targets['node_mask']]).float().mean()
                    metrics['node_accuracy'] += node_acc.item()

            if mask_mode in ["edge_feature", "edge_existence", "gate"]:
                # Edge feature prediction accuracy
                if 'edge_preds' in full_predictions and 'edge_features' in full_predictions['edge_preds']:
                    # This would depend on how your reconstruction function handles edge features
                    if 'masked_edges' in full_predictions['edge_preds'] and 'edge_mask' in targets:
                        # Calculate accuracy for edge features if available
                        pass  # Implement based on your model's output structure

            # Node existence accuracy (for modes that predict it)
            if mask_mode in ["node_existence", "removal"]:
                if mask_mode == "node_existence" and 'node_existence' in full_predictions:
                    pred_node_existence = (full_predictions['node_existence'] > 0.5).float()
                    if 'node_existence_mask' in targets and 'node_existence_target' in targets:
                        mask = targets['node_existence_mask']
                        target = targets['node_existence_target'][mask]
                        node_existence_acc = (pred_node_existence[mask] == target).float().mean()
                        metrics['node_existence_accuracy'] += node_existence_acc.item()

                elif mask_mode == "removal" and 'full_node_existence' in full_predictions:
                    pred_node_existence = (full_predictions['full_node_existence'] > 0.5).float()
                    if 'node_existence_target' in targets:
                        node_existence_target = targets['node_existence_target']
                        node_existence_acc = (pred_node_existence == node_existence_target).float().mean()
                        metrics['node_existence_accuracy'] += node_existence_acc.item()

            # Edge existence accuracy (for modes that predict it)
            if mask_mode in ["edge_existence", "removal"]:
                if mask_mode == "edge_existence" and 'edge_preds' in full_predictions and 'edge_existence' in \
                        full_predictions['edge_preds']:
                    pred_edge_existence = (full_predictions['edge_preds']['edge_existence'] > 0.5).float()
                    if 'edge_existence_mask' in targets and 'edge_existence_target' in targets:
                        # Get relevant subset of edge existence targets
                        edge_mask = targets['edge_mask']
                        target_existence = targets['edge_existence_target'][edge_mask]
                        # Ensure shapes match
                        valid_count = min(pred_edge_existence.size(0), target_existence.size(0))
                        if valid_count > 0:
                            edge_existence_acc = (pred_edge_existence[:valid_count] == target_existence[
                                                                                       :valid_count]).float().mean()
                            metrics['edge_existence_accuracy'] += edge_existence_acc.item()

                elif mask_mode == "removal" and 'edge_existence' in full_predictions:
                    # For removal mode, edge existence is handled differently depending on implementation
                    # This would depend on how your reconstruct_predictions function works
                    pass  # Implement based on your specific implementation

    # Average losses and metrics
    for key in val_losses:
        val_losses[key] /= max(1, len(val_loader))

    for key in metrics:
        metrics[key] /= max(1, len(val_loader))

    return val_losses, metrics
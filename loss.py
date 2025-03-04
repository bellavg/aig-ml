import torch
import torch.nn.functional as F


def compute_loss(predictions, targets):
    """
    Compute loss for all masking modes.

    Args:
        predictions: Dictionary of model predictions
        targets: Dictionary of targets and masking information

    Returns:
        total_loss: Combined loss value
        losses: Dictionary of individual loss components
    """
    losses = {}
    total_loss = 0.0

    mask_mode = targets.get('mask_mode', 'node_feature')

    # 1. Node feature prediction loss
    if 'node_features' in predictions and 'x_target' in targets:
        if mask_mode == 'removal':
            # Special handling for removal mode
            if 'original_to_new_indices' in targets:
                original_indices = targets['original_to_new_indices']
                pred_nodes = predictions['node_features']
                target_nodes = targets['x_target'][original_indices]

                # No masking needed here since we're already working with the reduced graph
                node_loss = F.binary_cross_entropy_with_logits(pred_nodes, target_nodes)
                losses['node_loss'] = node_loss
                total_loss += node_loss
        else:
            # Normal handling for other masking modes
            if 'node_mask' in targets and targets['node_mask'].sum() > 0:
                mask = targets['node_mask']
                pred_nodes = predictions['node_features'][mask]
                target_nodes = targets['x_target'][mask]
                node_loss = F.binary_cross_entropy_with_logits(pred_nodes, target_nodes)
                losses['node_loss'] = node_loss
                total_loss += node_loss

    # 2. Node existence prediction loss (for node_existence and removal modes)
    if 'node_existence' in predictions and mask_mode in ["node_existence", "removal"]:
        if mask_mode == "node_existence" and 'node_existence_mask' in targets and 'node_existence_target' in targets:
            # For node_existence mode, we only predict existence for masked nodes
            mask = targets['node_existence_mask']
            if mask.sum() > 0:
                node_existence = predictions['node_existence'][mask]
                node_existence_target = targets['node_existence_target'][mask]

                # Use with_logits if your model outputs logits rather than probabilities
                node_existence_loss = F.binary_cross_entropy_with_logits(
                    node_existence,
                    node_existence_target
                )
                losses["node_existence_loss"] = node_existence_loss
                total_loss += node_existence_loss

        elif mask_mode == "removal" and 'node_existence_target' in targets:
            if 'original_to_new_indices' in targets:
                # For removal mode, predict existence for non-removed nodes only
                original_indices = targets['original_to_new_indices']

                # The predictions are already for the non-removed nodes
                # We just need to get the corresponding targets
                node_existence_loss = F.binary_cross_entropy_with_logits(
                    predictions['node_existence'],
                    targets['node_existence_target'][original_indices]  # Should be all ones for kept nodes
                )
                losses["node_existence_loss"] = node_existence_loss
                total_loss += node_existence_loss

    # 3. Edge feature prediction loss
    if 'edge_preds' in predictions and 'edge_features' in predictions['edge_preds']:
        if mask_mode in ["edge_feature", "edge_existence"]:
            if 'edge_mask' in targets and 'edge_attr_target' in targets and targets['edge_mask'].sum() > 0:
                edge_mask = targets['edge_mask']
                pred_edges = predictions['edge_preds']['edge_features']

                # Make sure sizes match before computing loss
                if pred_edges.size(0) > 0 and edge_mask.sum() > 0:
                    target_edges = targets['edge_attr_target'][edge_mask]
                    valid_count = min(pred_edges.size(0), target_edges.size(0))

                    if valid_count > 0:
                        edge_feat_loss = F.binary_cross_entropy_with_logits(
                            pred_edges[:valid_count],
                            target_edges[:valid_count]
                        )
                        losses["edge_feature_loss"] = edge_feat_loss
                        total_loss += edge_feat_loss

    # 4. Edge existence prediction loss
    if 'edge_preds' in predictions and 'edge_existence' in predictions['edge_preds']:
        if mask_mode == "edge_existence" and 'edge_existence_mask' in targets and 'edge_existence_target' in targets:
            edge_mask = targets['edge_existence_mask']
            if edge_mask.sum() > 0:
                pred_edge_existence = predictions['edge_preds']['edge_existence']
                target_edge_existence = targets['edge_existence_target'][edge_mask]

                # Make sure sizes match
                valid_count = min(pred_edge_existence.size(0), target_edge_existence.size(0))
                if valid_count > 0:
                    edge_exist_loss = F.binary_cross_entropy_with_logits(
                        pred_edge_existence[:valid_count],
                        target_edge_existence[:valid_count]
                    )
                    losses["edge_existence_loss"] = edge_exist_loss
                    total_loss += edge_exist_loss

        elif mask_mode == "removal" and 'edge_existence_target' in targets:
            # For removal mode, edge existence is more complex
            # Here we assume edge_preds contains information about edges in the reduced graph
            if 'edge_index_mapping' in predictions['edge_preds']:
                # This would map from new edge indices to original edge indices
                edge_index_mapping = predictions['edge_preds']['edge_index_mapping']

                if edge_index_mapping.size(0) > 0:
                    pred_edge_existence = predictions['edge_preds']['edge_existence']
                    target_edge_existence = targets['edge_existence_target'][edge_index_mapping]

                    edge_exist_loss = F.binary_cross_entropy_with_logits(
                        pred_edge_existence,
                        target_edge_existence
                    )
                    losses["edge_existence_loss"] = edge_exist_loss
                    total_loss += edge_exist_loss

    # If no loss was computed, add a small constant to prevent NaNs
    if total_loss == 0.0:
        total_loss = torch.tensor(1e-5, device=next(iter(predictions.values())).device)
        losses["dummy_loss"] = total_loss

    return total_loss, losses
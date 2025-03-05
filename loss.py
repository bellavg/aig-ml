import torch
import torch.nn.functional as F


def compute_loss(predictions, targets):
    """
    Compute loss for the three masking modes:
    1. "node_feature": Mask node features and predict them
    2. "edge_feature": Mask edge features and predict them
    3. "connectivity": Mask edges and predict both existence and features

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

    # 1. Node feature prediction loss - relevant for all masking modes
    if 'node_features' in predictions and 'x_target' in targets:
        if 'node_mask' in targets and targets['node_mask'].sum() > 0:
            mask = targets['node_mask']
            pred_nodes = predictions['node_features'][mask]
            target_nodes = targets['x_target'][mask]
            node_loss = F.binary_cross_entropy_with_logits(pred_nodes, target_nodes)
            losses['node_loss'] = node_loss
            total_loss += node_loss

    # 2. Edge feature prediction loss - for edge_feature and connectivity modes
    # In the edge_feature prediction loss section
    if mask_mode in ["edge_feature", "connectivity"]:
        if 'edge_preds' in predictions and 'edge_features' in predictions['edge_preds']:
            if 'edge_mask' in targets and 'edge_attr_target' in targets and targets['edge_mask'].sum() > 0:
                # Get masked edge features from targets
                target_edges = targets['edge_attr_target'][targets['edge_mask']]

                # Get predictions
                pred_edges = predictions['edge_preds']['edge_features']

                # Make sure sizes match before computing loss
                if pred_edges.size(0) > 0 and target_edges.size(0) > 0:
                    valid_count = min(pred_edges.size(0), target_edges.size(0))

                    if valid_count > 0:
                        try:
                            # Clip predictions to prevent extreme values
                            clipped_preds = torch.clamp(pred_edges[:valid_count], -10, 10)  # More aggressive clipping

                            # Use reduction='none' to check individual loss values
                            edge_feat_loss_values = F.binary_cross_entropy_with_logits(
                                clipped_preds,
                                target_edges[:valid_count],
                                reduction='none'
                            )

                            # Filter out any extreme values
                            valid_mask = ~torch.isinf(edge_feat_loss_values) & ~torch.isnan(edge_feat_loss_values)
                            if valid_mask.any():
                                edge_feat_loss = edge_feat_loss_values[valid_mask].mean()
                            else:
                                print("WARNING: All loss values are invalid, using fallback")
                                edge_feat_loss = torch.tensor(1.0, device=pred_edges.device)

                            losses["edge_feature_loss"] = edge_feat_loss
                            total_loss += edge_feat_loss

                        except Exception as e:
                            print(f"Error in edge feature loss calculation: {e}")
                            edge_feat_loss = torch.tensor(1.0, device=pred_edges.device)
                            losses["edge_feature_loss"] = edge_feat_loss
                            total_loss += edge_feat_loss

    # 3. Edge existence loss - only for connectivity mode
    if mask_mode == "connectivity":
        if 'edge_preds' in predictions and 'edge_existence' in predictions['edge_preds']:
            # Modern format with all_candidate_pairs
            if 'all_candidate_targets' in targets and predictions['edge_preds']['edge_existence'].size(0) > 0:
                edge_exist_loss = F.binary_cross_entropy_with_logits(
                    predictions['edge_preds']['edge_existence'].squeeze(-1),
                    targets['all_candidate_targets'].squeeze(-1)
                )
                losses["edge_existence_loss"] = edge_exist_loss
                total_loss += edge_exist_loss

            # Fallback for older format
            elif 'connectivity_target' in targets:
                pred_existence = predictions['edge_preds']['edge_existence']
                target_existence = targets['connectivity_target']

                # Ensure sizes match
                valid_count = min(pred_existence.size(0), target_existence.size(0))

                if valid_count > 0:
                    edge_exist_loss = F.binary_cross_entropy_with_logits(
                        pred_existence[:valid_count],
                        target_existence[:valid_count]
                    )
                    losses["edge_existence_loss"] = edge_exist_loss
                    total_loss += edge_exist_loss

    # If no loss was computed, add a small constant to prevent NaNs
    if total_loss == 0.0:
        total_loss = torch.tensor(1e-5, device=next(iter(predictions.values())).device)
        losses["dummy_loss"] = total_loss

    return total_loss, losses
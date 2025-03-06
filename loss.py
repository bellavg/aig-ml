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

    # Helper function to safely get device from predictions
    def get_device_from_predictions(preds):
        for value in preds.values():
            if isinstance(value, torch.Tensor):
                return value.device
            elif isinstance(value, dict):
                device = get_device_from_predictions(value)
                if device is not None:
                    return device
        return torch.device('cpu')

    # Get the masking mode from targets
    mask_mode = targets.get('mask_mode')
    if mask_mode is None:
        raise ValueError("Masking mode must be specified in targets")

    # Safely get device
    device = get_device_from_predictions(predictions)

    # Only compute node feature prediction loss when in node_feature mode
    if mask_mode == "node_feature":
        if 'node_features' in predictions and 'x_target' in targets:
            if 'node_mask' in targets and targets['node_mask'].sum() > 0:
                mask = targets['node_mask']
                pred_nodes = predictions['node_features'][mask]
                target_nodes = targets['x_target'][mask]
                node_loss = F.binary_cross_entropy_with_logits(pred_nodes, target_nodes)
                losses['node_loss'] = node_loss
                total_loss += node_loss

    # Connectivity mode requires special handling
    if mask_mode == "connectivity":
        # Ensure we have candidate pairs and targets
        if 'all_candidate_pairs' not in targets or 'all_candidate_targets' not in targets:
            raise ValueError("Connectivity prediction requires candidate pairs and targets")

        # Edge existence loss (binary classification)
        try:
            # Check for edge_existence in different possible locations
            if 'edge_existence' in predictions:
                edge_existence_pred = predictions['edge_existence']
            elif 'edge_preds' in predictions and 'edge_existence' in predictions['edge_preds']:
                edge_existence_pred = predictions['edge_preds']['edge_existence']
            else:
                raise KeyError("Could not find edge existence predictions")

            existence_loss = F.binary_cross_entropy_with_logits(
                edge_existence_pred.squeeze(),
                targets['all_candidate_targets'].squeeze()
            )
            losses['edge_existence_loss'] = existence_loss
            total_loss += existence_loss
        except Exception as e:
            print(f"Error in edge existence loss: {e}")

        # Edge feature loss (only for predicted existing edges)
        try:
            # Check for edge_features in different possible locations
            if 'edge_features' in predictions:
                edge_features_pred = predictions['edge_features']
            elif 'edge_preds' in predictions and 'edge_features' in predictions['edge_preds']:
                edge_features_pred = predictions['edge_preds']['edge_features']
            else:
                raise KeyError("Could not find edge feature predictions")

            # Identify positive examples (edges that should exist)
            positive_mask = targets['all_candidate_targets'].squeeze() > 0.5

            # If we have positive examples, compute edge feature loss
            if positive_mask.sum() > 0:
                # Select positive examples
                positive_pred_features = edge_features_pred[positive_mask]

                # Get corresponding targets for positive examples
                if 'masked_edge_attr_target' in targets and targets['masked_edge_attr_target'] is not None:
                    target_features = targets['masked_edge_attr_target']

                    # Ensure sizes match
                    valid_count = min(positive_pred_features.size(0), target_features.size(0))

                    if valid_count > 0:
                        edge_feature_loss = F.binary_cross_entropy_with_logits(
                            positive_pred_features[:valid_count],
                            target_features[:valid_count]
                        )
                        losses['edge_feature_loss'] = edge_feature_loss
                        total_loss += edge_feature_loss
        except Exception as e:
            print(f"Error in edge feature loss: {e}")

    # Standard edge feature loss for edge_feature mode
    elif mask_mode == "edge_feature":
        try:
            # Check for edge_features in different possible locations
            if 'edge_features' in predictions:
                pred_edges = predictions['edge_features']
            elif 'edge_preds' in predictions and 'edge_features' in predictions['edge_preds']:
                pred_edges = predictions['edge_preds']['edge_features']
            else:
                raise KeyError("Could not find edge feature predictions")

            if 'edge_mask' in targets and 'edge_attr_target' in targets and targets['edge_mask'].sum() > 0:
                # Get masked edge features from targets
                target_edges = targets['edge_attr_target'][targets['edge_mask']]

                # Make sure sizes match before computing loss
                if pred_edges.size(0) > 0 and target_edges.size(0) > 0:
                    valid_count = min(pred_edges.size(0), target_edges.size(0))

                    if valid_count > 0:
                        # Clip predictions to prevent extreme values
                        clipped_preds = torch.clamp(pred_edges[:valid_count], -10, 10)

                        # Compute binary cross-entropy loss
                        edge_feat_loss = F.binary_cross_entropy_with_logits(
                            clipped_preds,
                            target_edges[:valid_count]
                        )

                        losses["edge_feature_loss"] = edge_feat_loss
                        total_loss += edge_feat_loss
        except Exception as e:
            print(f"Error in edge feature loss calculation: {e}")

    # If no loss was computed, add a small constant to prevent NaNs
    if total_loss == 0.0:
        total_loss = torch.tensor(1e-5, device=device)
        losses["dummy_loss"] = total_loss

    # Always add total_loss to the dictionary
    losses['total_loss'] = total_loss

    return total_loss, losses
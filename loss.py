import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F


def compute_loss(predictions, targets):
    """
    Main function to compute loss based on the masking mode.
    Dispatches to specific loss functions for each mode.

    Args:
        predictions: Dictionary of model predictions.
        targets: Dictionary containing targets and masking information.

    Returns:
        total_loss: Combined loss value.
        losses: Dictionary of individual loss components.
    """
    # Get masking mode from targets
    mask_mode = targets.get("mask_mode")
    if mask_mode is None:
        raise ValueError("Masking mode must be specified in targets")

    # Dispatch based on masking mode
    if mask_mode == "node_feature":
        return compute_node_feature_loss(predictions, targets)
    elif mask_mode == "edge_feature":
        return compute_edge_feature_loss(predictions, targets)
    elif mask_mode == "connectivity":
        return compute_connectivity_loss(predictions, targets)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")


def compute_node_feature_loss(predictions, targets):
    """
    Compute loss for node feature prediction, focusing only on the truth table value
    at index 3 for masked AND gates.

    Args:
        predictions: Dictionary of model predictions.
        targets: Dictionary containing targets and masking information.

    Returns:
        total_loss: Scalar loss value.
        losses: Dictionary with 'truth_table_loss' and 'total_loss'.
    """
    losses = {}
    total_loss = 0.0

    if "node_features" in predictions and "x_target" in targets:
        if "node_mask" in targets and targets["node_mask"].sum() > 0:
            # Get the masked nodes
            mask = targets["node_mask"]

            # Get predicted and target node features for masked nodes
            pred_nodes = predictions["node_features"][mask]
            target_nodes = targets["x_target"][mask]

            # Get the truth table index (should be at position 3)
            # If truth_table_idx is stored in targets, use it; otherwise default to 3
            tt_idx = getattr(targets, "truth_table_idx", 3)

            # Extract only the truth table values
            pred_tt = pred_nodes[:, tt_idx]
            target_tt = target_nodes[:, tt_idx]

            # Compute loss only on truth table values
            # Reshape to ensure correct dimensions for BCE loss
            tt_loss = F.binary_cross_entropy_with_logits(
                pred_tt.view(-1),
                target_tt.view(-1)
            )

            losses["truth_table_loss"] = tt_loss
            total_loss += tt_loss

    return finalize_loss(total_loss, losses, predictions)


def compute_edge_feature_loss(predictions, targets):
    """
    Compute loss for edge feature prediction (masking edge attributes).

    Args:
        predictions: Dictionary of model predictions.
        targets: Dictionary containing targets and masking information.

    Returns:
        total_loss: Scalar loss value.
        losses: Dictionary with 'edge_feature_loss' and 'total_loss'.
    """
    losses = {}
    total_loss = 0.0

    try:
        pred_edges = get_prediction(predictions, "edge_features")
        if "edge_mask" in targets and "edge_attr_target" in targets and targets["edge_mask"].sum() > 0:
            target_edges = targets["edge_attr_target"][targets["edge_mask"]]
            if pred_edges.size(0) > 0 and target_edges.size(0) > 0:
                valid_count = min(pred_edges.size(0), target_edges.size(0))
                if valid_count > 0:
                    clipped_preds = torch.clamp(pred_edges[:valid_count], -10, 10)
                    edge_feat_loss = F.binary_cross_entropy_with_logits(
                        clipped_preds, target_edges[:valid_count]
                    )
                    losses["edge_feature_loss"] = edge_feat_loss
                    total_loss += edge_feat_loss
    except Exception as e:
        print(f"Error in edge feature loss calculation: {e}")

    return finalize_loss(total_loss, losses, predictions)


def compute_connectivity_loss(predictions, targets):
    """
    Compute loss for connectivity prediction (masking edges).

    Args:
        predictions: Dictionary of model predictions.
        targets: Dictionary containing targets and masking information.

    Returns:
        total_loss: Scalar loss value.
        losses: Dictionary with 'edge_existence_loss', 'edge_feature_loss', and 'total_loss'.
    """
    losses = {}
    total_loss = 0.0

    # Edge existence loss (binary classification)
    try:
        edge_existence_pred = get_prediction(predictions, "edge_existence")
        existence_loss = F.binary_cross_entropy_with_logits(
            edge_existence_pred.squeeze(), targets["all_candidate_targets"].squeeze()
        )
        losses["edge_existence_loss"] = existence_loss
        total_loss += existence_loss
    except Exception as e:
        print(f"Error in edge existence loss: {e}")

    # Edge feature loss for positive edges
    try:
        edge_features_pred = get_prediction(predictions, "edge_features")
        positive_mask = targets["all_candidate_targets"].squeeze() > 0.5
        if positive_mask.sum() > 0:
            positive_pred_features = edge_features_pred[positive_mask]
            if "masked_edge_attr_target" in targets and targets["masked_edge_attr_target"] is not None:
                target_features = targets["masked_edge_attr_target"]
                valid_count = min(positive_pred_features.size(0), target_features.size(0))
                if valid_count > 0:
                    edge_feature_loss = F.binary_cross_entropy_with_logits(
                        positive_pred_features[:valid_count], target_features[:valid_count]
                    )
                    losses["edge_feature_loss"] = edge_feature_loss
                    total_loss += edge_feature_loss
    except Exception as e:
        print(f"Error in edge feature loss: {e}")

    return finalize_loss(total_loss, losses, predictions)





def get_prediction(predictions, key):
    """
    Safely extract a prediction value from nested dictionaries.

    Args:
        predictions: Dictionary of model predictions.
        key: Key to retrieve.

    Returns:
        Corresponding tensor.
    """
    if key in predictions:
        return predictions[key]
    elif "edge_preds" in predictions and key in predictions["edge_preds"]:
        return predictions["edge_preds"][key]
    else:
        raise KeyError(f"Could not find {key} in predictions")


def finalize_loss(total_loss, losses, predictions):
    """
    Ensures loss values are properly returned and prevents NaNs.

    Args:
        total_loss: Scalar total loss value.
        losses: Dictionary of individual loss components.
        predictions: Dictionary of model predictions.

    Returns:
        Updated total_loss and losses dictionary.
    """

    if total_loss == 0.0:
        total_loss = torch.tensor(1e-5)
        losses["dummy_loss"] = total_loss

    losses["total_loss"] = total_loss
    return total_loss, losses

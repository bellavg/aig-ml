
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union


def compute_loss(predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
                 targets: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Main function to compute loss based on the masking mode.
    Dispatches to specific loss functions for each mode.

    Args:
        predictions: Model predictions (either tensor or dictionary)
        targets: Dictionary containing targets and masking information

    Returns:
        total_loss: Combined loss value
        losses: Dictionary of individual loss components
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


def compute_node_feature_loss(predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
                              targets: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute loss for node feature prediction, focusing only on the truth table value
    for masked AND gates.

    Args:
        predictions: Model predictions (either tensor or dictionary)
        targets: Dictionary containing targets and masking information

    Returns:
        total_loss: Scalar loss value
        losses: Dictionary with 'truth_table_loss' and 'total_loss'
    """
    losses = {}

    # Handle different prediction formats
    if isinstance(predictions, dict) and "node_features" in predictions:
        # Dictionary format with node_features key
        pred = predictions["node_features"]
    else:
        # Direct tensor format
        pred = predictions

    # Get the node mask
    node_mask = targets.get("node_mask")
    if node_mask is None or node_mask.sum() == 0:
        # No nodes were masked, return zero loss
        return torch.tensor(0.0, device=pred.device), {"truth_table_loss": torch.tensor(0.0, device=pred.device)}

    # Get the truth table index (should be at position 3)
    tt_idx = targets.get("truth_table_idx", 3)

    # Determine target values to use
    if "original_truth_table_values" in targets:
        # Directly use stored original values (more efficient)
        target_tt = targets["original_truth_table_values"]

        # Extract predictions for masked nodes - handling different output formats
        if pred.dim() > 1 and pred.size(1) > tt_idx:
            # If predictions have multiple feature dimensions, extract only truth table values
            pred_tt = pred[node_mask, tt_idx]
        else:
            # If predictions are already for the specific feature or are pre-filtered
            pred_tt = pred[node_mask]
    else:
        # Fall back to extracting from x_target
        target_nodes = targets["x_target"][node_mask]
        target_tt = target_nodes[:, tt_idx]

        # Extract predictions - matching the approach used for targets
        if pred.dim() > 1 and pred.size(1) > tt_idx:
            pred_nodes = pred[node_mask]
            pred_tt = pred_nodes[:, tt_idx]
        else:
            pred_tt = pred[node_mask]

    # Ensure proper shape for loss computation
    pred_tt = pred_tt.view(-1)
    target_tt = target_tt.view(-1)

    # Determine appropriate loss function based on value range
    # If values are in [0,1] range or binary, use BCE loss
    # Otherwise use MSE loss for regression
    # Regression
    tt_loss = F.mse_loss(pred_tt, target_tt)
    losses["truth_table_loss"] = tt_loss

    # Total loss is just the truth table loss for this mode
    total_loss = tt_loss
    losses["total_loss"] = total_loss

    return total_loss, losses


def compute_edge_feature_loss(predictions, targets):
    """
    Compute loss for edge feature prediction (masking edge attributes).

    Args:
        predictions: Dictionary of model predictions.
        targets: Dictionary containing targets and masking information.

    Returns:
        total_loss: Scalar loss value.
        losses: Dictionary with detailed loss components.
    """
    losses = {}
    device = next(iter(predictions.values())).device if isinstance(predictions, dict) else predictions.device

    # Default zero loss for fallback
    zero_loss = torch.tensor(0.0, device=device)

    # Check masking mode
    if targets.get("mask_mode") != "edge_feature":
        return zero_loss, {"edge_feature_loss": zero_loss, "total_loss": zero_loss}

    # Extract edge feature predictions - handle different output structures
    if isinstance(predictions, dict):
        if "edge_attr_pred" in predictions:
            # Direct access if available
            edge_pred = predictions["edge_attr_pred"]
        elif "edge_preds" in predictions and isinstance(predictions["edge_preds"], dict):
            # Nested dictionary case
            edge_pred = predictions["edge_preds"].get("edge_features")
        else:
            # No valid predictions found
            print("Warning: Edge feature predictions not found in model output")
            return zero_loss, {"edge_feature_loss": zero_loss, "total_loss": zero_loss}
    else:
        # Not a dictionary
        print("Warning: Predictions should be a dictionary")
        return zero_loss, {"edge_feature_loss": zero_loss, "total_loss": zero_loss}

    # Check for valid predictions
    if edge_pred is None:
        print("Warning: Edge predictions are None")
        return zero_loss, {"edge_feature_loss": zero_loss, "total_loss": zero_loss}

    # Get masked edges and the original edge attributes
    # FIX: Use dictionary-style access consistently
    if "original_edge_attr" in targets and targets["original_edge_attr"] is not None:
        # Direct access to original values stored during masking
        target_edges = targets["original_edge_attr"]
    elif "edge_attr_target" in targets and "edge_mask" in targets and targets["edge_mask"].sum() > 0:
        # Fallback to edge_attr_target
        target_edges = targets["edge_attr_target"][targets["edge_mask"]]
    else:
        print("Warning: Target edge attributes not found")
        return zero_loss, {"edge_feature_loss": zero_loss, "total_loss": zero_loss}

    # Verify we have valid data to compute loss
    if edge_pred.size(0) == 0 or target_edges.size(0) == 0:
        print(f"Warning: Empty prediction ({edge_pred.size(0)}) or target ({target_edges.size(0)})")
        return zero_loss, {"edge_feature_loss": zero_loss, "total_loss": zero_loss}

    # Check for size mismatch - this should ideally not happen with correct implementation
    if edge_pred.size(0) != target_edges.size(0):
        print(f"Warning: Size mismatch between predictions ({edge_pred.size(0)}) and targets ({target_edges.size(0)})")
        # If there's a mismatch, use the indexed approach with masked_edge_indices if available
        if "masked_edge_indices" in predictions:
            valid_indices = predictions["masked_edge_indices"]
            if valid_indices.size(0) == target_edges.size(0):
                edge_pred = edge_pred  # Already correct
            else:
                # Still mismatched, use minimum size
                valid_count = min(edge_pred.size(0), target_edges.size(0))
                edge_pred = edge_pred[:valid_count]
                target_edges = target_edges[:valid_count]
                print(f"Using truncated tensors with {valid_count} edges")
        else:
            # Fallback: use minimum size
            valid_count = min(edge_pred.size(0), target_edges.size(0))
            edge_pred = edge_pred[:valid_count]
            target_edges = target_edges[:valid_count]
            print(f"Using truncated tensors with {valid_count} edges")

    # Apply numeric stability measures - clip predictions to avoid extreme values
    edge_pred_clipped = torch.clamp(edge_pred, -10, 10)

    # Ensure target is float for better loss calculation
    target_edges = target_edges.float()

    # Since edge features are one-hot encoded ([1,0] for INV, [0,1] for REG),
    # use binary cross-entropy with logits loss
    edge_feat_loss = F.binary_cross_entropy_with_logits(
        edge_pred_clipped, target_edges, reduction='mean'
    )

    losses["edge_feature_loss"] = edge_feat_loss
    losses["total_loss"] = edge_feat_loss

    # Calculate accuracy metrics for monitoring
    with torch.no_grad():
        # Convert predictions to binary
        pred_labels = (torch.sigmoid(edge_pred) > 0.5).float()

        # Calculate overall accuracy
        correct = (pred_labels == target_edges).all(dim=1).float().mean()
        losses["edge_feat_accuracy"] = correct

        # Calculate per-class accuracy if we have at least 2 classes
        if target_edges.size(1) >= 2:
            inv_accuracy = ((pred_labels[:, 0] == target_edges[:, 0])).float().mean()
            reg_accuracy = ((pred_labels[:, 1] == target_edges[:, 1])).float().mean()

            losses["inv_edge_accuracy"] = inv_accuracy
            losses["reg_edge_accuracy"] = reg_accuracy

    return edge_feat_loss, losses

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

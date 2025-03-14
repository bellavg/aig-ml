import torch
from typing import Dict, Any


def reconstruct_predictions(predictions: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct predictions to match the original graph structure
    for all three masking modes.

    Args:
        predictions: Dict with model predictions
        targets: Dict with ground truth and masking info

    Returns:
        full_pred: Dict with reconstructed predictions
    """
    mask_mode = targets.get('mask_mode', 'node_feature')
    full_pred = {}

    # Common handling: copy non-reconstructed outputs
    for key in predictions:
        if key not in ['node_features', 'edge_preds']:
            full_pred[key] = predictions[key]

    # Mode-specific reconstruction
    if mask_mode == "node_feature":
        # For node_feature mode, we only care about truth table values (at index 3)
        if 'node_features' in predictions:
            # Extract raw node feature predictions
            if isinstance(predictions['node_features'], torch.Tensor):
                raw_pred = predictions['node_features']
            else:
                # If it's a dictionary with nested structure
                raw_pred = predictions['node_features'].get('features', predictions['node_features'])

            # Store raw predictions for evaluation
            full_pred['node_features'] = raw_pred

            # Additionally, reconstruct full node features if needed
            if 'x_target' in targets and 'node_mask' in targets and targets['node_mask'].sum() > 0:
                node_mask = targets['node_mask']

                # Truth table index is position 3 by default
                truth_table_idx = targets.get('truth_table_idx', 3)

                # Create reconstructed full features tensor
                full_features = targets['x_target'].clone()

                # If predictions are for all feature dimensions
                if raw_pred.dim() > 1 and raw_pred.size(1) > truth_table_idx:
                    # Only update the truth table value at index 3
                    full_features[node_mask, truth_table_idx] = raw_pred[node_mask, truth_table_idx]
                else:
                    # If predictions are already specifically for the truth table value
                    # Apply predictions to just the masked truth table values
                    full_features[node_mask, truth_table_idx] = raw_pred[node_mask]

                full_pred['full_node_features'] = full_features

    elif mask_mode == "edge_feature":
        # Reconstruct edge features if present
        if 'edge_preds' in predictions and 'edge_features' in predictions['edge_preds']:
            edge_preds = predictions['edge_preds']

            # Create full edge features tensor with defaults from target
            if 'edge_attr_target' in targets and 'edge_mask' in targets:
                edge_features_target = targets['edge_attr_target']
                full_edge_features = edge_features_target.clone()

                # Map predictions to masked positions
                if 'masked_edges' in edge_preds:
                    masked_indices = torch.nonzero(targets['edge_mask']).squeeze(-1)

                    # Map predictions to original edge indices
                    if masked_indices.size(0) > 0 and edge_preds['edge_features'].size(0) > 0:
                        valid_count = min(masked_indices.size(0), edge_preds['edge_features'].size(0))
                        full_edge_features[masked_indices[:valid_count]] = edge_preds['edge_features'][:valid_count]

                full_pred['full_edge_features'] = full_edge_features

    elif mask_mode == "connectivity":
        # Reconstruct both edge existence and edge features
        if 'edge_preds' in predictions:
            edge_preds = predictions['edge_preds']

            # First, construct edge existence predictions
            if 'edge_existence' in edge_preds:
                # Store the raw predictions for evaluation
                full_pred['edge_existence_preds'] = edge_preds['edge_existence']

                # Also store the candidate pairs for reference
                if 'all_candidate_pairs' in edge_preds:
                    full_pred['edge_candidate_pairs'] = edge_preds['all_candidate_pairs']
                elif 'masked_edges' in edge_preds:
                    full_pred['edge_candidate_pairs'] = edge_preds['masked_edges']

            # Next, handle edge feature predictions
            if 'edge_features' in edge_preds:
                # Store the raw predictions for evaluation
                full_pred['edge_feature_preds'] = edge_preds['edge_features']

                # Optionally reconstruct full edge feature tensor
                if 'edge_attr_target' in targets and 'edge_mask' in targets:
                    edge_features_target = targets['edge_attr_target']
                    full_edge_features = edge_features_target.clone()

                    # Map predictions to masked positions - but only for edges that exist
                    # (those with predicted existence > 0.5)
                    if 'masked_edges' in edge_preds and 'edge_existence' in edge_preds:
                        masked_indices = torch.nonzero(targets['edge_mask']).squeeze(-1)
                        edge_exists = (edge_preds['edge_existence'] > 0).squeeze(-1)

                        for i, exists in enumerate(edge_exists):
                            if exists and i < len(masked_indices) and i < edge_preds['edge_features'].size(0):
                                full_edge_features[masked_indices[i]] = edge_preds['edge_features'][i]

                    full_pred['full_edge_features'] = full_edge_features

    return full_pred
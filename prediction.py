import torch


def reconstruct_predictions(predictions, targets):
    """
    Reconstruct predictions to match the original graph structure
    for all three masking modes.

    Args:
        outputs: Dict with model predictions
        targets: Dict with ground truth and masking info

    Returns:
        full_predictions: Dict with reconstructed predictions
    """
    mask_mode = targets.get('mask_mode', 'node_feature')
    full_pred = {}

    # Common handling: copy non-reconstructed outputs
    for key in predictions:
        if key not in ['node_features', 'edge_preds']:
            full_pred[key] = predictions[key]

    # Reconstruct node features (common to all masking modes)
    if 'node_features' in predictions:
        full_pred['node_features'] = predictions['node_features']

    # Mode-specific reconstruction
    if mask_mode == "node_feature":
        # Nothing special to reconstruct for node features
        pass

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
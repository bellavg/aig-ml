import torch

def reconstruct_predictions(outputs, targets):
    """
    Reconstruct predictions to match the original graph structure
    for all five masking modes.

    Args:
        outputs: Dict with model predictions
        targets: Dict with ground truth and masking info

    Returns:
        full_predictions: Dict with reconstructed predictions
    """
    mask_mode = targets['mask_mode'] if 'mask_mode' in targets else "node_feature"
    full_pred = {}

    # Common handling: copy non-reconstructed outputs
    for key in outputs:
        if key not in ['node_features', 'node_existence', 'edge_preds']:
            full_pred[key] = outputs[key]

    # Mode-specific reconstruction
    if mask_mode == "node_feature":
        # Just pass through outputs directly
        full_pred = outputs.copy()

    elif mask_mode == "edge_feature":
        # Copy outputs and handle edge feature reconstruction
        full_pred = outputs.copy()

        # Reconstruct edge features if present
        if 'edge_preds' in outputs and 'edge_features' in outputs['edge_preds']:
            edge_preds = outputs['edge_preds']

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

    elif mask_mode == "node_existence":
        # Copy outputs and handle node existence reconstruction
        full_pred = outputs.copy()

        # Create full node existence tensor if needed
        if 'node_existence' in outputs and 'node_existence_mask' in targets:
            # Default existence is 1.0 for non-masked nodes
            node_existence = torch.ones(targets['x_target'].size(0), 1, device=outputs['node_existence'].device)

            # Update with predictions for masked nodes
            mask = targets['node_existence_mask']
            node_existence[mask] = outputs['node_existence'][mask]

            full_pred['full_node_existence'] = node_existence

    elif mask_mode == "edge_existence":
        # Copy outputs and handle edge existence reconstruction
        full_pred = outputs.copy()

        # Create full edge existence tensor if needed
        if 'edge_preds' in outputs and 'edge_existence' in outputs['edge_preds']:
            edge_preds = outputs['edge_preds']

            # Default existence is 1.0 for non-masked edges
            if 'edge_index_target' in targets:
                num_edges = targets['edge_index_target'].size(1)
                edge_existence = torch.ones(num_edges, 1, device=edge_preds['edge_existence'].device)

                # Update with predictions for masked edges
                if 'edge_mask' in targets:
                    masked_indices = torch.nonzero(targets['edge_mask']).squeeze(-1)

                    # Map predictions to original edge indices
                    if masked_indices.size(0) > 0 and edge_preds['edge_existence'].size(0) > 0:
                        valid_count = min(masked_indices.size(0), edge_preds['edge_existence'].size(0))
                        edge_existence[masked_indices[:valid_count]] = edge_preds['edge_existence'][:valid_count]

                full_pred['full_edge_existence'] = edge_existence

    elif mask_mode == "removal":
        # For removal mode, reconstruct the full node and edge sets
        if 'num_original_nodes' in outputs and 'node_existence' in outputs:
            num_original_nodes = outputs['num_original_nodes']

            # Reconstruct node features
            node_feat_dim = outputs['node_features'].size(1)
            full_node_features = torch.zeros((num_original_nodes, node_feat_dim),
                                             device=outputs['node_features'].device)

            # Default to original features (from target)
            if 'x_target' in targets:
                full_node_features = targets['x_target'].clone()

            # Update with predicted features for nodes that weren't removed
            if 'original_to_new_indices' in outputs:
                original_indices = outputs['original_to_new_indices']
                full_node_features[original_indices] = outputs['node_features']

            full_pred['node_features'] = full_node_features

            # Reconstruct node existence
            full_node_existence = torch.zeros((num_original_nodes, 1),
                                              device=outputs['node_existence'].device)

            # Default existence: removed nodes = 0, non-removed nodes = 1
            if 'node_removal_mask' in targets:
                full_node_existence[~targets['node_removal_mask']] = 1.0

            # Update with predicted existence for nodes that weren't removed
            if 'original_to_new_indices' in outputs:
                original_indices = outputs['original_to_new_indices']
                full_node_existence[original_indices] = outputs['node_existence']

            full_pred['full_node_existence'] = full_node_existence

            # Edge reconstruction for removal mode is more complex
            # and would depend on how your model handles edge prediction in this mode
            if 'edge_preds' in outputs and 'removed_nodes_info' in outputs['edge_preds']:
                # This would involve reconstructing edges based on node existence
                # Implement based on your specific approach
                pass

    return full_pred
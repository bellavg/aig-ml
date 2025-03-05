import torch


def create_masked_batch(batch, mp=0.20, mask_mode="node_feature"):
    """
    Create masked batch with support for five different masking modes:
    1. "node_feature": Mask only node features
    2. "edge_feature": Mask only edge features
    3. "node_existence": Mask nodes and predict both existence and features
    4. "edge_existence": Mask edges and predict both existence and features
    5. "removal": Completely remove nodes and their edges from the graph

    Args:
        batch: PyG Data object containing the batch
        mp: Masking probability (0.0 to 1.0)
        mask_mode: One of the five masking modes

    Returns:
        masked_batch: PyG Data object with masking applied
    """
    masked_batch = batch.clone()

    # Initialize masking information
    is_and_gate, node_mask, node_existence_mask, batch_idx = _initialize_node_masks(batch)
    edge_mask, edge_existence_mask = _initialize_edge_masks(batch)

    # Store original values as targets
    _store_targets(masked_batch, batch)

    # Create masks based on the selected masking mode
    if mask_mode in ["node_feature", "node_existence", "removal"]:
        node_mask = _create_node_masks(batch, is_and_gate, batch_idx, mp)
        if mask_mode == "node_existence":
            node_existence_mask = node_mask.clone()

    if mask_mode in ["edge_feature", "edge_existence"]:
        edge_mask = _create_edge_masks(batch, mp)
        if mask_mode == "edge_existence":
            edge_existence_mask = edge_mask.clone()

    if mask_mode == "removal":
        edge_mask = _identify_connected_edges(batch, node_mask)

    # Apply the appropriate masking operation based on mode
    masking_functions = {
        "node_feature": _apply_node_feature_masking,
        "edge_feature": _apply_edge_feature_masking,
        "node_existence": _apply_node_existence_masking,
        "edge_existence": _apply_edge_existence_masking,
        "removal": _apply_node_removal_masking
    }

    masking_functions[mask_mode](
        masked_batch,
        batch,
        node_mask,
        edge_mask,
        node_existence_mask,
        edge_existence_mask
    )

    # Store masking settings
    masked_batch.mask_mode = mask_mode
    masked_batch.mask_prob = mp

    return masked_batch


def _initialize_node_masks(batch):
    """Initialize node masks and identify AND gates"""
    # Get node types (AND gates)
    is_and_gate = (batch.x[:, 0] == 0) & (batch.x[:, 1] == 1) & (batch.x[:, 2] == 0)

    # Initialize node masks
    node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)
    node_existence_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)

    # Get batch assignment for each node
    batch_idx = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long)

    return is_and_gate, node_mask, node_existence_mask, batch_idx


def _initialize_edge_masks(batch):
    """Initialize edge masks"""
    edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)
    edge_existence_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

    return edge_mask, edge_existence_mask


def _store_targets(masked_batch, batch):
    """Store original values as targets"""
    masked_batch.x_target = batch.x.clone()
    masked_batch.edge_index_target = batch.edge_index.clone()
    masked_batch.edge_attr_target = batch.edge_attr.clone() if hasattr(batch, 'edge_attr') else None


def _create_node_masks(batch, is_and_gate, batch_idx, mp):
    """Select nodes for masking based on masking probability"""
    node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)

    # Iterate through each graph in the batch
    for b in torch.unique(batch_idx):
        # Get nodes for this graph
        graph_mask = batch_idx == b
        graph_nodes = torch.nonzero(graph_mask & is_and_gate).squeeze(-1)

        # Randomly select nodes to mask
        num_to_mask = max(1, int(len(graph_nodes) * mp))
        if len(graph_nodes) > 0:
            masked_indices = graph_nodes[torch.randperm(len(graph_nodes))[:num_to_mask]]
            node_mask[masked_indices] = True

    return node_mask


def _create_edge_masks(batch, mp):
    """Select edges for masking based on masking probability"""
    edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

    # Randomly select edges to mask
    num_edges = batch.edge_index.size(1)
    num_to_mask_edges = max(1, int(num_edges * mp))

    if num_edges > 0:
        masked_edge_indices = torch.randperm(num_edges)[:num_to_mask_edges]
        edge_mask[masked_edge_indices] = True

    return edge_mask


def _identify_connected_edges(batch, node_mask):
    """Identify edges connected to masked nodes"""
    edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

    if node_mask.sum() > 0:
        src, dst = batch.edge_index
        for i in range(batch.edge_index.size(1)):
            if node_mask[src[i]] or node_mask[dst[i]]:
                edge_mask[i] = True

    return edge_mask


def _apply_node_feature_masking(masked_batch, batch, node_mask, edge_mask, node_existence_mask, edge_existence_mask):
    """Apply masking for node_feature mode"""
    # Mask only node features
    if node_mask.sum() > 0:
        masked_batch.x[node_mask] = 0.0  # Zero out masked nodes

    # Store the node mask
    masked_batch.node_mask = node_mask


def _apply_edge_feature_masking(masked_batch, batch, node_mask, edge_mask, node_existence_mask, edge_existence_mask):
    """Apply masking for edge_feature mode"""
    # Mask edge features but keep the edges
    masked_batch.edge_mask = edge_mask

    # For masked edges, zero out the features but keep the edge
    if edge_mask.sum() > 0 and hasattr(batch, 'edge_attr'):
        # Create a copy of edge attributes
        masked_edge_attr = batch.edge_attr.clone()
        # Zero out masked edge features
        masked_edge_attr[edge_mask] = 0.0
        masked_batch.edge_attr = masked_edge_attr

    # Store the node mask even though we're not masking nodes
    # This is necessary because train_epoch expects it to exist
    masked_batch.node_mask = node_mask


def _apply_node_existence_masking(masked_batch, batch, node_mask, edge_mask, node_existence_mask, edge_existence_mask):
    """Apply masking for node_existence mode"""
    # Mask node features and mark for existence prediction
    if node_mask.sum() > 0:
        masked_batch.x[node_mask] = 0.0  # Zero out masked nodes

    # Store node existence mask separately
    masked_batch.node_existence_mask = node_existence_mask
    masked_batch.node_existence_target = torch.ones(batch.x.size(0), 1, device=batch.x.device)

    # Store the node mask
    masked_batch.node_mask = node_mask


def _apply_edge_existence_masking(masked_batch, batch, node_mask, edge_mask, node_existence_mask, edge_existence_mask):
    """Apply masking for edge_existence mode"""
    # Mask edge features and mark for existence prediction
    masked_batch.edge_mask = edge_mask
    masked_batch.edge_existence_mask = edge_existence_mask

    # For masked edges, zero out features but keep the edge
    if edge_mask.sum() > 0 and hasattr(batch, 'edge_attr'):
        # Create a copy of edge attributes
        masked_edge_attr = batch.edge_attr.clone()
        # Zero out masked edge features
        masked_edge_attr[edge_mask] = 0.0
        masked_batch.edge_attr = masked_edge_attr

    # Store edge existence target: all masked edges should exist
    masked_batch.edge_existence_target = torch.ones(batch.edge_index.size(1), 1, device=batch.edge_index.device)

    # Store the node mask even though we're not masking nodes
    # This is necessary because train_epoch expects it to exist
    masked_batch.node_mask = node_mask


def _apply_node_removal_masking(masked_batch, batch, node_mask, edge_mask, node_existence_mask, edge_existence_mask):
    """Apply masking for removal mode"""
    # Store original node removal mask for reconstruction
    masked_batch.node_removal_mask = node_mask.clone()
    masked_batch.edge_mask = edge_mask.clone()

    # For removal, we actually remove the nodes and their edges
    if node_mask.sum() > 0:
        # Get indices of nodes to keep
        keep_nodes = ~node_mask
        nodes_to_keep = torch.nonzero(keep_nodes).squeeze(-1)

        # Create a mapping from old indices to new indices
        old_to_new = torch.full((batch.x.size(0),), -1, dtype=torch.long, device=batch.x.device)
        for new_idx, old_idx in enumerate(nodes_to_keep):
            old_to_new[old_idx] = new_idx

        # Update node features and batch assignment
        masked_batch.x = masked_batch.x[keep_nodes]
        if hasattr(batch, 'batch'):
            masked_batch.batch = batch.batch[keep_nodes]

        # Track original node indices for reconstruction
        masked_batch.original_to_new_indices = torch.arange(batch.x.size(0), device=batch.x.device)[keep_nodes]
        masked_batch.old_to_new_mapping = old_to_new
        masked_batch.num_original_nodes = batch.x.size(0)

        # Keep only edges that don't involve removed nodes
        keep_edges = ~edge_mask
        masked_batch.edge_index = batch.edge_index[:, keep_edges]
        if hasattr(batch, 'edge_attr'):
            masked_batch.edge_attr = batch.edge_attr[keep_edges]

        # Remap edge indices to the new node indices
        if masked_batch.edge_index.size(1) > 0:
            for i in range(2):  # For both source and target nodes
                masked_batch.edge_index[i] = old_to_new[masked_batch.edge_index[i]]

        # Create a new node mask that matches the reduced graph size
        # Since we've removed nodes, no nodes are masked in the reduced graph
        new_node_mask = torch.zeros(masked_batch.x.size(0), dtype=torch.bool, device=masked_batch.x.device)
        masked_batch.node_mask = new_node_mask

    # Create existence targets
    masked_batch.node_existence_target = torch.ones(batch.x.size(0), 1, device=batch.x.device)
    masked_batch.node_existence_target[node_mask] = 0.0  # Removed nodes should not exist

    masked_batch.edge_existence_target = torch.ones(batch.edge_index.size(1), 1, device=batch.edge_index.device)
    masked_batch.edge_existence_target[edge_mask] = 0.0  # Edges to removed nodes should not exist
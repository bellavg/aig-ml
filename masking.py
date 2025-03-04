import torch


def create_masked_batch(batch, mp=0.20, mask_mode="node_feature"):
    """
    Create masked batch with support for three masking modes:
    1. "node_feature": Mask only node features
    2. "edge_feature": Mask only edge features
    3. "connectivity": Mask edges and predict both existence and features

    Args:
        batch: PyG Data object containing the batch
        mp: Masking probability (0.0 to 1.0)
        mask_mode: One of the three masking modes

    Returns:
        masked_batch: PyG Data object with masking applied
    """
    masked_batch = batch.clone()

    # Initialize masking information
    is_and_gate, node_mask = _initialize_node_masks(batch)
    edge_mask = _initialize_edge_masks(batch)

    # Store original values as targets
    _store_targets(masked_batch, batch)

    # Create masks based on the selected masking mode
    if mask_mode == "node_feature":
        node_mask = _create_node_masks(batch, is_and_gate, mp)

    elif mask_mode in ["edge_feature", "connectivity"]:
        edge_mask = _create_edge_masks(batch, mp)

    # Apply the appropriate masking operation based on mode
    if mask_mode == "node_feature":
        _apply_node_feature_masking(masked_batch, batch, node_mask)

    elif mask_mode == "edge_feature":
        _apply_edge_feature_masking(masked_batch, batch, edge_mask)

    elif mask_mode == "connectivity":
        _apply_connectivity_masking(masked_batch, batch, edge_mask)

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

    return is_and_gate, node_mask


def _initialize_edge_masks(batch):
    """Initialize edge masks"""
    edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)
    return edge_mask


def _store_targets(masked_batch, batch):
    """Store original values as targets"""
    masked_batch.x_target = batch.x.clone()
    masked_batch.edge_index_target = batch.edge_index.clone()
    masked_batch.edge_attr_target = batch.edge_attr.clone() if hasattr(batch, 'edge_attr') else None


def _create_node_masks(batch, is_and_gate, mp):
    """Select nodes for masking based on masking probability"""
    node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)

    # Get batch assignment for each node
    batch_idx = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long)

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
    """Select edges for masking based on masking probability with a minimum of 2 edges when possible"""
    edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

    # Randomly select edges to mask
    num_edges = batch.edge_index.size(1)

    # Calculate how many edges to mask based on probability, with minimum of 2 if possible
    num_to_mask_edges = int(num_edges * mp)
    num_to_mask_edges = max(num_to_mask_edges, 2 if num_edges >= 2 else 1)
    num_to_mask_edges = min(num_to_mask_edges, num_edges)  # Don't try to mask more edges than exist

    if num_edges > 0:
        masked_edge_indices = torch.randperm(num_edges)[:num_to_mask_edges]
        edge_mask[masked_edge_indices] = True


    return edge_mask

def _apply_node_feature_masking(masked_batch, batch, node_mask):
    """Apply masking for node_feature mode"""
    # Mask only node features
    if node_mask.sum() > 0:
        masked_batch.x[node_mask] = 0.0  # Zero out masked nodes

    # Store the node mask
    masked_batch.node_mask = node_mask


def _apply_edge_feature_masking(masked_batch, batch, edge_mask):
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

    # Store an empty node mask for consistency
    # This is necessary because train_epoch expects it to exist
    masked_batch.node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)


def _apply_connectivity_masking(masked_batch, batch, edge_mask):
    """Apply connectivity masking using candidate edge strategy with negative examples."""
    # Store original edge information
    masked_batch.edge_index_target = batch.edge_index.clone()
    masked_batch.edge_attr_target = batch.edge_attr.clone() if hasattr(batch, 'edge_attr') else None

    # Store which edges are masked
    masked_batch.edge_mask = edge_mask
    masked_batch.masked_edge_indices = torch.nonzero(edge_mask).squeeze(-1)

    # Remove masked edges from the graph
    keep_edges = ~edge_mask
    masked_batch.edge_index = batch.edge_index[:, keep_edges]
    if hasattr(batch, 'edge_attr'):
        masked_batch.edge_attr = batch.edge_attr[keep_edges]

    # Store source and destination nodes of masked edges
    src_nodes = batch.edge_index[0, edge_mask]
    dst_nodes = batch.edge_index[1, edge_mask]

    # For each node pair in masked edges, store the information needed for prediction
    masked_batch.masked_edge_node_pairs = torch.stack([src_nodes, dst_nodes], dim=0)
    masked_batch.connectivity_target = torch.ones(edge_mask.sum(), device=batch.edge_index.device)

    if hasattr(batch, 'edge_attr'):
        masked_batch.masked_edge_attr_target = batch.edge_attr[edge_mask]

    # Create validation candidates - node pairs that could have edges but don't in the original
    negative_pairs = []
    batch_idx = batch.batch if hasattr(batch, 'batch') else None

    if batch_idx is not None:
        # For each graph in the batch
        for b in torch.unique(batch_idx):
            # Get nodes in this graph
            graph_nodes = torch.nonzero(batch_idx == b).squeeze(-1)

            if len(graph_nodes) > 5:  # Only for graphs with enough nodes
                # Sample a few random node pairs that don't have edges
                num_neg_samples = min(10, edge_mask.sum().item())  # Add same number as masked edges or max 10

                for _ in range(num_neg_samples):
                    attempts = 0
                    while attempts < 20:  # Limit attempts to avoid infinite loop
                        attempts += 1
                        idx1 = torch.randint(0, len(graph_nodes), (1,)).item()
                        idx2 = torch.randint(0, len(graph_nodes), (1,)).item()

                        if idx1 == idx2:  # Skip self-loops
                            continue

                        src = graph_nodes[idx1]
                        dst = graph_nodes[idx2]

                        # Check if this edge already exists in the original graph
                        edge_exists = False
                        for e in range(batch.edge_index.size(1)):
                            if (batch.edge_index[0, e] == src and batch.edge_index[1, e] == dst):
                                edge_exists = True
                                break

                        if not edge_exists:
                            negative_pairs.append((src.item(), dst.item()))
                            break

    # Add negative examples if we found any
    if len(negative_pairs) > 0:
        # Convert list of tuples to tensor
        neg_src = [p[0] for p in negative_pairs]
        neg_dst = [p[1] for p in negative_pairs]
        neg_src_tensor = torch.tensor(neg_src, device=batch.edge_index.device)
        neg_dst_tensor = torch.tensor(neg_dst, device=batch.edge_index.device)

        negative_pair_tensor = torch.stack([neg_src_tensor, neg_dst_tensor], dim=0)
        masked_batch.negative_edge_pairs = negative_pair_tensor
        masked_batch.negative_edge_targets = torch.zeros(len(negative_pairs), device=batch.edge_index.device)

        # Store combined positive and negative examples for training
        # Positive examples (masked edges that should exist)
        pos_pairs = masked_batch.masked_edge_node_pairs
        pos_targets = masked_batch.connectivity_target

        # Combine positive and negative examples
        all_pairs = torch.cat([pos_pairs, negative_pair_tensor], dim=1)
        all_targets = torch.cat([pos_targets, masked_batch.negative_edge_targets], dim=0)

        masked_batch.all_candidate_pairs = all_pairs
        masked_batch.all_candidate_targets = all_targets

    # Store an empty node mask for consistency
    # This is necessary because train_epoch expects it to exist
    masked_batch.node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)
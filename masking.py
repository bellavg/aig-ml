import torch


def create_masked_batch(batch, mp=0.20, gate_masking=False):
    """
    Create masked batch with support for both node-only and gate masking.

    Args:
        batch: PyG Data object containing the batch
        mp: Masking probability (0.0 to 1.0)
        gate_masking: If True, mask entire gates (node + connected edges)

    Returns:
        masked_batch: PyG Data object with masking applied
    """
    masked_batch = batch.clone()

    # Get node types (AND gates)
    is_and_gate = (batch.x[:, 0] == 0) & (batch.x[:, 1] == 1) & (batch.x[:, 2] == 0)

    # Initialize node mask (only for AND gates)
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

    # Store original values as targets
    masked_batch.x_target = batch.x.clone()
    masked_batch.edge_index_target = batch.edge_index.clone()
    masked_batch.edge_attr_target = batch.edge_attr.clone() if hasattr(batch, 'edge_attr') else None

    # Apply node masking
    if node_mask.sum() > 0:
        masked_batch.x[node_mask] = 0.0  # Zero out masked nodes

    # Edge masking for gate masking mode
    if gate_masking and node_mask.sum() > 0:
        # Create edge mask tensor
        edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

        # Get source and destination nodes for each edge
        src, dst = batch.edge_index

        # Identify edges connected to masked nodes
        for i in range(batch.edge_index.size(1)):
            if node_mask[src[i]] or node_mask[dst[i]]:
                edge_mask[i] = True

        # Store original edge info for prediction targets
        masked_batch.edge_mask = edge_mask

        # For easier access to just the masked edges
        masked_batch.masked_edge_indices = torch.nonzero(edge_mask).squeeze(-1)
        masked_batch.masked_edge_attr = batch.edge_attr[edge_mask] if hasattr(batch, 'edge_attr') else None

        # Remove masked edges from the graph for the forward pass
        keep_edges = ~edge_mask
        masked_batch.edge_index = batch.edge_index[:, keep_edges]
        if hasattr(batch, 'edge_attr'):
            masked_batch.edge_attr = batch.edge_attr[keep_edges]
    else:
        # Initialize empty edge mask for consistency
        masked_batch.edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)
        masked_batch.masked_edge_indices = torch.tensor([], dtype=torch.long, device=batch.edge_index.device)
        masked_batch.masked_edge_attr = None

    # Store masking settings
    masked_batch.node_mask = node_mask
    masked_batch.gate_masking = gate_masking
    masked_batch.mask_prob = mp

    return masked_batch




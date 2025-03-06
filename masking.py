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
    is_and_gate = _identify_and_gates(batch)
    node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)
    edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

    # Store original values as targets
    _store_targets(masked_batch, batch)

    # Create masks based on the selected masking mode
    if mask_mode == "node_feature":
        node_mask = _create_node_masks(batch, is_and_gate, mp)

    elif mask_mode in ["edge_feature", "connectivity"]:
        if mask_mode == "connectivity":
            # Use more strategic edge masking for connectivity prediction
            edge_mask = _create_strategic_edge_masks(batch, mp, is_and_gate)
        else:
            edge_mask = _create_edge_masks(batch, mp)

    # Apply the appropriate masking operation based on mode
    if mask_mode == "node_feature":
        _apply_node_feature_masking(masked_batch, node_mask)

    elif mask_mode == "edge_feature":
        _apply_edge_feature_masking(masked_batch, batch, edge_mask)

    elif mask_mode == "connectivity":
        _apply_connectivity_masking(masked_batch, batch, edge_mask)

    # Store masking settings
    masked_batch.mask_mode = mask_mode
    masked_batch.mask_prob = mp

    return masked_batch


def _identify_and_gates(batch):
    """Identify AND gates in the graph"""
    # Get node types (AND gates) - [0, 1, 0] encoding
    is_and_gate = (batch.x[:, 0] == 0) & (batch.x[:, 1] == 1) & (batch.x[:, 2] == 0)
    return is_and_gate


def _identify_input_nodes(batch, is_and_gate=None):
    """Identify input nodes in the AIG (PI nodes)"""
    # Primary inputs have type encoding [1, 0, 0]
    is_input = (batch.x[:, 0] == 1) & (batch.x[:, 1] == 0) & (batch.x[:, 2] == 0)
    return is_input


def _identify_output_nodes(batch, is_and_gate=None):
    """Identify output nodes in the AIG (PO nodes)"""
    # Primary outputs have type encoding [0, 0, 1]
    is_output = (batch.x[:, 0] == 0) & (batch.x[:, 1] == 0) & (batch.x[:, 2] == 1)
    return is_output


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


def _create_strategic_edge_masks(batch, mp, is_and_gate=None):
    """
    Select edges for masking in a more strategic way for AIGs
    - Prefer to mask edges between gates
    - Avoid masking input and output connections
    """
    edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

    if is_and_gate is None:
        is_and_gate = _identify_and_gates(batch)

    # Identify input and output nodes
    is_input = _identify_input_nodes(batch, is_and_gate)
    is_output = _identify_output_nodes(batch, is_and_gate)

    # Get batch assignment for edges
    batch_idx = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long)

    # Process each graph separately
    for b in torch.unique(batch_idx):
        # Find edges within this graph
        if hasattr(batch, 'batch'):
            graph_node_mask = batch_idx == b
            graph_edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

            # An edge is in this graph if both source and target nodes are in the graph
            for e in range(batch.edge_index.size(1)):
                src, dst = batch.edge_index[0, e], batch.edge_index[1, e]
                if graph_node_mask[src] and graph_node_mask[dst]:
                    graph_edge_mask[e] = True
        else:
            # Single graph case
            graph_edge_mask = torch.ones(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)

        graph_edges = torch.nonzero(graph_edge_mask).squeeze(-1)

        if len(graph_edges) > 0:
            # Calculate number of edges to mask
            num_to_mask = max(2, int(len(graph_edges) * mp))
            num_to_mask = min(num_to_mask, len(graph_edges) - 1)  # Leave at least one edge

            # Categorize edges
            internal_edges = []  # Edges between internal gates
            input_edges = []  # Edges from inputs
            output_edges = []  # Edges to outputs
            other_edges = []  # Other edges

            for e_idx in graph_edges:
                src, dst = batch.edge_index[0, e_idx], batch.edge_index[1, e_idx]

                if is_input[src]:
                    input_edges.append(e_idx.item())
                elif is_output[dst]:
                    output_edges.append(e_idx.item())
                elif is_and_gate[src] and is_and_gate[dst]:
                    internal_edges.append(e_idx.item())
                else:
                    other_edges.append(e_idx.item())

            # Prioritize masking internal edges, then other edges, then input/output edges
            edges_to_mask = []
            remaining = num_to_mask

            # Take as many internal edges as possible (up to 60% of mask budget)
            internal_budget = min(len(internal_edges), int(num_to_mask * 0.6))
            if internal_budget > 0:
                selected = torch.randperm(len(internal_edges))[:internal_budget]
                edges_to_mask.extend([internal_edges[i] for i in selected])
                remaining -= internal_budget

            # Then take other edges (up to 30% of mask budget)
            if remaining > 0 and len(other_edges) > 0:
                other_budget = min(len(other_edges), int(num_to_mask * 0.3))
                selected = torch.randperm(len(other_edges))[:other_budget]
                edges_to_mask.extend([other_edges[i] for i in selected])
                remaining -= other_budget

            # Finally, if we still need more, take input/output edges
            if remaining > 0:
                combined = input_edges + output_edges
                if len(combined) > 0:
                    selected = torch.randperm(len(combined))[:remaining]
                    edges_to_mask.extend([combined[i] for i in selected])
                    remaining -= len(selected)

            # If we still haven't filled our quota (rare), fall back to random selection
            if remaining > 0 and len(graph_edges) > 0:
                all_edges = graph_edges.tolist()
                for edge in edges_to_mask:
                    if edge in all_edges:
                        all_edges.remove(edge)

                if len(all_edges) >= remaining:
                    selected = torch.randperm(len(all_edges))[:remaining]
                    edges_to_mask.extend([all_edges[i] for i in selected])

            # Mark selected edges in the mask
            for e in edges_to_mask:
                edge_mask[e] = True

    return edge_mask


def _apply_node_feature_masking(masked_batch, node_mask):
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
    """Apply connectivity masking with strategic negative examples"""
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

    # Create negative examples (node pairs that don't have edges)
    negative_pairs = _generate_strategic_negative_examples(batch, edge_mask)

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
    else:
        # If no negative examples, just use the positive ones
        masked_batch.all_candidate_pairs = masked_batch.masked_edge_node_pairs
        masked_batch.all_candidate_targets = masked_batch.connectivity_target

    # Ensure we don't break AIG structural constraints
    _validate_aig_constraints(masked_batch)

    # Store an empty node mask for consistency
    # This is necessary because train_epoch expects it to exist
    masked_batch.node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)


def _generate_strategic_negative_examples(batch, edge_mask):
    """
    Generate negative examples (node pairs with no edge) in a more strategic way for AIGs
    """
    negative_pairs = []
    batch_idx = batch.batch if hasattr(batch, 'batch') else None

    # Identify AND gates
    is_and_gate = _identify_and_gates(batch)

    # Identify input and output nodes
    is_input = _identify_input_nodes(batch, is_and_gate)
    is_output = _identify_output_nodes(batch, is_and_gate)

    # For each graph in the batch
    unique_batch_indices = [0] if batch_idx is None else torch.unique(batch_idx)

    for b in unique_batch_indices:
        # Get nodes for this graph
        if batch_idx is not None:
            graph_mask = batch_idx == b
            graph_nodes = torch.nonzero(graph_mask).squeeze(-1)
        else:
            graph_nodes = torch.arange(batch.x.size(0), device=batch.x.device)

        if len(graph_nodes) <= 5:  # Skip if too few nodes
            continue

        # Determine how many negative examples to generate
        pos_examples = edge_mask.sum().item() if batch_idx is None else edge_mask[
            torch.isin(batch.edge_index[0], graph_nodes) &
            torch.isin(batch.edge_index[1], graph_nodes)
            ].sum().item()

        num_neg_samples = min(20, max(pos_examples, 10))  # At least 10, at most 20

        # Get nodes by type for this graph
        graph_and_gates = graph_nodes[is_and_gate[graph_nodes]]
        graph_inputs = graph_nodes[is_input[graph_nodes]]
        graph_outputs = graph_nodes[is_output[graph_nodes]]

        # Generate strategic negative examples
        neg_pairs_added = 0

        # Strategy 1: Input to output (these rarely connect directly in AIGs)
        if len(graph_inputs) > 0 and len(graph_outputs) > 0:
            for _ in range(min(3, num_neg_samples // 4)):
                input_idx = torch.randint(0, len(graph_inputs), (1,)).item()
                output_idx = torch.randint(0, len(graph_outputs), (1,)).item()
                src, dst = graph_inputs[input_idx], graph_outputs[output_idx]

                if not _edge_exists(batch, src, dst) and (src.item(), dst.item()) not in negative_pairs:
                    negative_pairs.append((src.item(), dst.item()))
                    neg_pairs_added += 1

        # Strategy 2: Output to AND gate (reverse direction, should never happen in AIGs)
        if len(graph_outputs) > 0 and len(graph_and_gates) > 0:
            for _ in range(min(3, num_neg_samples // 4)):
                output_idx = torch.randint(0, len(graph_outputs), (1,)).item()
                gate_idx = torch.randint(0, len(graph_and_gates), (1,)).item()
                src, dst = graph_outputs[output_idx], graph_and_gates[gate_idx]

                if not _edge_exists(batch, src, dst) and (src.item(), dst.item()) not in negative_pairs:
                    negative_pairs.append((src.item(), dst.item()))
                    neg_pairs_added += 1

        # Strategy 3: AND gate to input (reverse direction, should never happen in AIGs)
        if len(graph_and_gates) > 0 and len(graph_inputs) > 0:
            for _ in range(min(3, num_neg_samples // 4)):
                gate_idx = torch.randint(0, len(graph_and_gates), (1,)).item()
                input_idx = torch.randint(0, len(graph_inputs), (1,)).item()
                src, dst = graph_and_gates[gate_idx], graph_inputs[input_idx]

                if not _edge_exists(batch, src, dst) and (src.item(), dst.item()) not in negative_pairs:
                    negative_pairs.append((src.item(), dst.item()))
                    neg_pairs_added += 1

        # Strategy 4: Random node pairs (to fill remaining quota)
        remaining = num_neg_samples - neg_pairs_added
        attempts = 0
        max_attempts = 50

        while neg_pairs_added < num_neg_samples and attempts < max_attempts:
            attempts += 1
            idx1 = torch.randint(0, len(graph_nodes), (1,)).item()
            idx2 = torch.randint(0, len(graph_nodes), (1,)).item()

            if idx1 == idx2:  # Skip self-loops
                continue

            src, dst = graph_nodes[idx1], graph_nodes[idx2]

            if not _edge_exists(batch, src, dst) and (src.item(), dst.item()) not in negative_pairs:
                negative_pairs.append((src.item(), dst.item()))
                neg_pairs_added += 1

    return negative_pairs


def _edge_exists(batch, src, dst):
    """Check if an edge exists between src and dst in the batch"""
    for e in range(batch.edge_index.size(1)):
        if batch.edge_index[0, e] == src and batch.edge_index[1, e] == dst:
            return True
    return False


def _validate_aig_constraints(masked_batch):
    """
    Ensure that the masked graph still maintains basic AIG constraints:
    - AND gates should have at least one input
    - Critical paths should not be broken
    """
    # Ensure AND gates have at least one input after masking
    is_and_gate = _identify_and_gates(masked_batch)
    and_gates = torch.nonzero(is_and_gate).squeeze(-1)

    # Current edge indices after masking
    edge_index = masked_batch.edge_index

    # Check in-degree for each AND gate
    for gate in and_gates:
        # Count incoming edges to this gate in the masked graph
        in_edges = (edge_index[1] == gate).sum().item()

        # If no inputs, we need to add back at least one
        if in_edges == 0:
            # Check original graph for potential edges to restore
            original_edge_index = masked_batch.edge_index_target
            edge_mask = masked_batch.edge_mask

            # Find an incoming edge to restore
            for e in range(original_edge_index.size(1)):
                if original_edge_index[1, e] == gate and edge_mask[e]:
                    # Unmask this edge
                    edge_mask[e] = False
                    break

            # Reconstruct the graph with updated mask
            keep_edges = ~edge_mask
            masked_batch.edge_index = masked_batch.edge_index_target[:, keep_edges]
            if hasattr(masked_batch, 'edge_attr_target'):
                masked_batch.edge_attr = masked_batch.edge_attr_target[keep_edges]

            # Update masked edge pairs and targets
            masked_edges = masked_batch.edge_index_target[:, edge_mask]
            masked_batch.masked_edge_node_pairs = masked_edges
            masked_batch.connectivity_target = torch.ones(edge_mask.sum(), device=edge_mask.device)

            if hasattr(masked_batch, 'edge_attr_target'):
                masked_batch.masked_edge_attr_target = masked_batch.edge_attr_target[edge_mask]
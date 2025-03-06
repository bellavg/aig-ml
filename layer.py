import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=2, num_heads=8, dropout=0.1):
        super(EdgeAwareGraphTransformerLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, num_heads)

        # Edge gating mechanism
        self.edge_gate = nn.Linear(edge_dim, num_heads)

        # Multi-head attention with edge features
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        # Relative positional encoding projection
        self.rel_pos_proj = nn.Parameter(torch.randn(1, num_heads, 1))

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.ReLU(),
            nn.Linear(4 * out_dim, out_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # Projection if dimensions change
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def relative_attention(self, graph_edges, num_nodes, device):
        """
        Create a relative positional encoding matrix based on graph structure.

        Args:
            graph_edges: Edge indices [2, num_edges]
            num_nodes: Number of nodes in the graph
            device: Device to create tensor on

        Returns:
            Relative positional encoding [1, num_nodes, num_nodes]
        """
        # Initialize relative positional encoding matrix
        rel_pos = torch.zeros((num_nodes, num_nodes), device=device)

        # Fill in direct connections
        if graph_edges.size(1) > 0:
            for i in range(graph_edges.size(1)):
                src, dst = graph_edges[0, i].item(), graph_edges[1, i].item()
                if src < num_nodes and dst < num_nodes:  # Ensure indices are valid
                    rel_pos[src, dst] = 1  # Direct connection

        # Add to attention scores
        return rel_pos.unsqueeze(0)  # [1, nodes, nodes]

    def forward(self, x, edge_index, edge_attr, batch=None, mask=None):
        """
        x: Node features [num_nodes, in_dim]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge features [num_edges, edge_dim]
        batch: Batch assignment for nodes [num_nodes]
        mask: Attention mask [num_nodes]
        """
        # Apply layer normalization
        x_norm = self.norm1(x)

        # Project queries, keys, values
        q = self.q_proj(x_norm)  # [num_nodes, out_dim]
        k = self.k_proj(x_norm)  # [num_nodes, out_dim]
        v = self.v_proj(x_norm)  # [num_nodes, out_dim]

        # Apply edge projection with gating
        if edge_attr is not None:
            # Calculate edge feature projection
            raw_edge_weights = self.edge_proj(edge_attr)

            # Calculate gating factors (values between 0 and 1)
            edge_influence = torch.sigmoid(self.edge_gate(edge_attr))

            # Apply gate to control how much of each edge feature passes through
            edge_weights = raw_edge_weights * edge_influence
        else:
            edge_weights = None

        # Process each graph in the batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        out = torch.zeros_like(v)

        for g in range(num_graphs):
            # Get nodes for this graph
            graph_mask = batch == g
            graph_nodes = torch.nonzero(graph_mask).squeeze(1)

            if len(graph_nodes) == 0:
                continue

            # Get node indices in the original tensor
            node_indices = {n.item(): i for i, n in enumerate(graph_nodes)}

            # Get node features for this graph
            graph_q = q[graph_nodes].view(len(graph_nodes), self.num_heads, self.head_dim)
            graph_k = k[graph_nodes].view(len(graph_nodes), self.num_heads, self.head_dim)
            graph_v = v[graph_nodes].view(len(graph_nodes), self.num_heads, self.head_dim)

            # Reshape for multi-head attention
            graph_q = graph_q.permute(1, 0, 2)  # [heads, nodes, head_dim]
            graph_k = graph_k.permute(1, 0, 2)  # [heads, nodes, head_dim]
            graph_v = graph_v.permute(1, 0, 2)  # [heads, nodes, head_dim]

            # Compute attention scores
            attn_scores = torch.matmul(graph_q, graph_k.transpose(-2, -1)) / (
                    self.head_dim ** 0.5)  # [heads, nodes, nodes]

            # Find edges for this graph
            if edge_index.size(1) > 0:  # Check that there are edges
                graph_edge_mask = (edge_index[0, :] >= graph_nodes.min()) & \
                                  (edge_index[0, :] <= graph_nodes.max()) & \
                                  (edge_index[1, :] >= graph_nodes.min()) & \
                                  (edge_index[1, :] <= graph_nodes.max())

                # Get graph-specific edges
                graph_edges = edge_index[:, graph_edge_mask] if graph_edge_mask.sum() > 0 else edge_index[:, :0]

                # Calculate relative positional encodings
                rel_pos = self.relative_attention(graph_edges, len(graph_nodes), x.device)

                # Project and add relative positional encodings to attention scores
                # Reshape to match attention heads dimension [heads, nodes, nodes]
                rel_pos_influence = (self.rel_pos_proj.unsqueeze(-1) * rel_pos.unsqueeze(1)).squeeze(0)
                attn_scores = attn_scores + rel_pos_influence

                # Add edge weights to attention scores
                if graph_edge_mask.sum() > 0 and edge_weights is not None:
                    graph_edge_weights = edge_weights[graph_edge_mask]

                    # Adjust edge indices to local graph indices
                    local_src = torch.tensor([node_indices[n.item()] for n in graph_edges[0]], device=x.device)
                    local_dst = torch.tensor([node_indices[n.item()] for n in graph_edges[1]], device=x.device)

                    # Add edge weights to attention scores
                    for h in range(self.num_heads):
                        attn_scores[h, local_src, local_dst] += graph_edge_weights[:, h]

            # Apply mask if provided
            if mask is not None:
                graph_node_mask = mask[graph_nodes]
                attn_mask = ~graph_node_mask.bool().unsqueeze(0).unsqueeze(-1)
                attn_scores = attn_scores.masked_fill(attn_mask, -1e9)

            # Apply softmax and dropout
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Apply attention to values
            graph_out = torch.matmul(attn_probs, graph_v)  # [heads, nodes, head_dim]
            graph_out = graph_out.permute(1, 0, 2).reshape(len(graph_nodes), self.out_dim)  # [nodes, out_dim]

            # Store output for this graph
            out[graph_nodes] = graph_out

        # Residual connection and projection
        x_out = self.proj(x) + self.dropout(out)

        # Feed-forward with residual
        x_out = x_out + self.dropout(self.ff(self.norm2(x_out)))

        return x_out
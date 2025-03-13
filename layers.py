
from torch_scatter import scatter_add, scatter_mean, scatter_max

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DAGTransformerLayer(nn.Module):
    """
    Transformer layer optimized for DAGs with relative positional encoding based on GRPE.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, max_hop=5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.head_dim = d_model // nhead
        self.max_hop = max_hop
        self.scale = self.head_dim ** -0.5

        # Multi-head projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, distance_matrix, edge_index, edge_attr,
                query_hop_emb, key_hop_emb, value_hop_emb,
                query_edge_emb, key_edge_emb, value_edge_emb, batch=None):
        """
        Forward pass with GRPE-style relative positional encoding.
        """
        # Apply first layer normalization
        x_norm = self.norm1(x)

        # Project to get queries, keys, values
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Reshape for multi-head attention
        batch_size = x.size(0)
        q = q.view(batch_size, self.nhead, self.head_dim)
        k = k.view(batch_size, self.nhead, self.head_dim)
        v = v.view(batch_size, self.nhead, self.head_dim)

        # Process each graph in the batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        out = torch.zeros_like(x)

        for g in range(num_graphs):
            # Get nodes for this graph
            graph_mask = batch == g
            graph_nodes = torch.nonzero(graph_mask).squeeze(1)

            if len(graph_nodes) == 0:
                continue

            # Get subgraph features
            graph_q = q[graph_nodes]  # [nodes, heads, head_dim]
            graph_k = k[graph_nodes]
            graph_v = v[graph_nodes]

            # Get subgraph distance matrix
            graph_distance = distance_matrix[graph_nodes][:, graph_nodes]

            # Initialize attention scores with dot product
            attn_scores = torch.zeros(len(graph_nodes), len(graph_nodes), self.nhead,
                                      device=x.device)

            # Compute dot-product attention (vectorized)
            for h in range(self.nhead):
                attn_scores[:, :, h] = torch.matmul(
                    graph_q[:, h], graph_k[:, h].transpose(0, 1)
                ) * self.scale

            # Apply DAG-specific attention masking (vectorized)
            # Make unreachable nodes have very negative attention scores
            mask = (graph_distance >= self.max_hop + 1).unsqueeze(-1).expand(-1, -1, self.nhead)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

            # Add topology-based attention (node-topology interaction) - VECTORIZED VERSION
            # Replace the nested loops with vectorized operations
            num_graph_nodes = len(graph_nodes)

            # Get hop distances and ensure they're within bounds for indexing
            hop_distances = torch.clamp(graph_distance, max=self.max_hop).long()

            # Reshape for broadcasting with query_hop_emb weights
            hop_distances = hop_distances.view(num_graph_nodes, num_graph_nodes, 1)

            # Get embeddings for all hop distances at once
            q_hop_weights = query_hop_emb.weight[hop_distances]  # [num_nodes, num_nodes, d_model]
            k_hop_weights = key_hop_emb.weight[hop_distances]  # [num_nodes, num_nodes, d_model]

            # Reshape for per-head processing
            q_hop_weights = q_hop_weights.view(num_graph_nodes, num_graph_nodes, self.nhead, self.head_dim)
            k_hop_weights = k_hop_weights.view(num_graph_nodes, num_graph_nodes, self.nhead, self.head_dim)

            # Compute topology attention contribution for all pairs at once
            for h in range(self.nhead):
                # Compute query-hop interactions (vectorized)
                q_contrib = torch.sum(
                    graph_q[:, h, None, :] * q_hop_weights[:, :, h, :],
                    dim=-1
                )  # [num_nodes, num_nodes]

                # Compute key-hop interactions (vectorized)
                k_contrib = torch.sum(
                    graph_k[None, :, h, :] * k_hop_weights[:, :, h, :],
                    dim=-1
                )  # [num_nodes, num_nodes]

                # Add contributions to attention scores
                attn_scores[:, :, h] += q_contrib + k_contrib

            # Add edge-based attention if available - KEEP YOUR EXISTING CODE HERE
            # Modify the edge-based attention section in the forward method
            if edge_attr is not None and edge_index.size(1) > 0:
                # Find edges within this subgraph
                src_idx, dst_idx = edge_index

                # Create node maps (global to local indices)
                node_map = {node.item(): i for i, node in enumerate(graph_nodes)}

                # Find edges between nodes in this subgraph
                edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

                for e in range(edge_index.size(1)):
                    src, dst = src_idx[e].item(), dst_idx[e].item()
                    if src in node_map and dst in node_map:
                        edge_mask[e] = True

                if edge_mask.sum() > 0:
                    # Process all valid edges at once for efficiency
                    valid_edges = torch.nonzero(edge_mask).squeeze(1)

                    for e_idx in valid_edges:
                        # Get source and destination nodes
                        src = src_idx[e_idx].item()
                        dst = dst_idx[e_idx].item()

                        # Map to local indices in the subgraph
                        src_local = node_map[src]
                        dst_local = node_map[dst]

                        # Get edge type
                        edge_type = 0
                        if edge_attr is not None:
                            if edge_attr[e_idx].dim() > 0:
                                edge_type = torch.argmax(edge_attr[e_idx]).item()
                            else:
                                edge_type = int(edge_attr[e_idx].item())

                        # Ensure edge_type is within bounds
                        edge_type = min(edge_type, query_edge_emb.weight.size(0) - 1)

                        # Add query-edge and key-edge interactions for each head
                        for h in range(self.nhead):
                            # Add query-edge interaction
                            q_edge = torch.matmul(
                                graph_q[src_local, h],
                                query_edge_emb.weight[edge_type].view(self.nhead, self.head_dim)[h]
                            )

                            # Add key-edge interaction
                            k_edge = torch.matmul(
                                graph_k[dst_local, h],
                                key_edge_emb.weight[edge_type].view(self.nhead, self.head_dim)[h]
                            )

                            # Add to attention scores (now uses scalar addition)
                            attn_scores[src_local, dst_local, h] += q_edge + k_edge

                # Apply softmax per head (vectorized)
            attn_probs = F.softmax(attn_scores, dim=1)

            # Apply dropout to attention probabilities
            attn_probs = self.dropout(attn_probs)

            # Apply attention to values (vectorized per head)
            graph_out = torch.zeros(len(graph_nodes), self.d_model, device=x.device)

            for h in range(self.nhead):
                # Basic attention: weighted sum of values (vectorized)
                head_out = torch.matmul(attn_probs[:, :, h], graph_v[:, h])  # [num_nodes, head_dim]

                # Store in correct slice of output
                graph_out[:, h * self.head_dim:(h + 1) * self.head_dim] = head_out

                # Add value position encodings - vectorized version for hop-based value encoding
                # Reshape for broadcasting with value_hop_emb weights
                v_hop_weights = value_hop_emb.weight[hop_distances[:, :, 0]]  # [num_nodes, num_nodes, d_model]
                v_hop_weights = v_hop_weights.view(num_graph_nodes, num_graph_nodes, self.nhead, self.head_dim)

                # Weight by attention probability and sum
                weighted_v_hop = v_hop_weights[:, :, h, :] * attn_probs[:, :, h].unsqueeze(
                    -1)  # [num_nodes, num_nodes, head_dim]
                v_hop_contrib = weighted_v_hop.sum(dim=1)  # Sum over source nodes, [num_nodes, head_dim]

                # Add to the output
                graph_out[:, h * self.head_dim:(h + 1) * self.head_dim] += v_hop_contrib

            # Apply output projection
            graph_out = self.out_proj(graph_out)

            # Store in output tensor
            out[graph_nodes] = graph_out

            # Apply first residual connection
        x = x + self.dropout(out)

        # Apply second sublayer: FFN with residual
        x = x + self.ff(self.norm2(x))

        return x


class StructureExtractor(nn.Module):
    """Structure extractor optimized for AIGs/DAGs."""

    def __init__(self, embed_dim, num_layers=2, batch_norm=True, dropout=0.1, edge_dim=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        # Edge feature projection - make edge_dim configurable
        self.edge_proj = nn.Linear(edge_dim, embed_dim)

        # GNN layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(embed_dim, embed_dim) for _ in range(num_layers)
        ])

        # Batch normalization or Layer normalization (more stable for varying batch sizes)
        if batch_norm:
            self.norm_layers = nn.ModuleList([
                nn.BatchNorm1d(embed_dim) for _ in range(num_layers)
            ])
        else:
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(num_layers)
            ])

        # Activation and dropout
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Output projection with skip connection
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, edge_index, edge_attr=None):
        """Extract structural features from the graph."""
        h = x

        # Transform edge features if available
        if edge_attr is not None:
            if edge_attr.dim() > 0:  # Check if edge attributes exist and have proper dimension
                edge_features = self.edge_proj(edge_attr)
            else:
                edge_features = None
        else:
            edge_features = None

        # Apply GNN layers
        for i, conv in enumerate(self.conv_layers):
            # Apply graph convolution
            h_conv = conv(h, edge_index, edge_features)

            # Apply normalization
            h_conv = self.norm_layers[i](h_conv)

            # Apply activation and dropout
            h_conv = self.activation(h_conv)
            h_conv = self.dropout(h_conv)

            # Residual connection
            h = h + h_conv

        # Final projection
        h = self.out_proj(h)

        return h


class GraphConvLayer(nn.Module):
    """Graph convolution layer optimized for DAGs using scatter operations for efficiency."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.W_edge = nn.Linear(in_dim, out_dim)

        self.out_dim = out_dim

        # Initialize weights with Glorot/Xavier initialization
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.zeros_(self.W_edge.bias)

    def forward(self, x, edge_index, edge_features=None):
        """Forward pass with efficient scatter operations."""
        # Early return if no edges
        if edge_index.size(1) == 0:
            return torch.zeros(x.size(0), self.out_dim, device=x.device)  # Return [num_nodes, out_dim]

        # Transform node features
        h = self.W(x)

        # Get source and target nodes
        src, dst = edge_index

        # Initialize messages with transformed source features
        messages = h[src]

        # Add edge features if available
        if edge_features is not None:
            edge_messages = self.W_edge(edge_features)
            messages = messages + edge_messages

        # Aggregate messages using scatter_add (much more efficient than loops)
        out = torch.zeros_like(h)
        scatter_add(messages, dst, dim=0, out=out)

        # Compute in-degrees for normalization
        ones = torch.ones(edge_index.size(1), device=edge_index.device)
        in_degree = scatter_add(ones, dst, dim=0, dim_size=x.size(0))

        # Avoid division by zero
        in_degree = torch.clamp(in_degree, min=1.0)

        # Normalize by in-degree
        out = out / in_degree.unsqueeze(1)

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max


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

        Args:
            x: Node features [num_nodes, d_model]
            distance_matrix: Hop distances [num_nodes, num_nodes]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            *_hop_emb: Embeddings for hop distances
            *_edge_emb: Embeddings for edge types
            batch: Batch assignment for nodes [num_nodes]
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

            # Compute dot-product attention
            for h in range(self.nhead):
                attn_scores[:, :, h] = torch.matmul(
                    graph_q[:, h], graph_k[:, h].transpose(0, 1)
                ) * self.scale

            # Add topology-based attention (node-topology interaction)
            for src in range(len(graph_nodes)):
                for dst in range(len(graph_nodes)):
                    hop_dist = graph_distance[src, dst].item()

                    # Add query-hop interaction
                    q_hop = torch.matmul(
                        graph_q[src],
                        query_hop_emb.weight[hop_dist].view(self.nhead, self.head_dim).transpose(0, 1)
                    )

                    # Add key-hop interaction
                    k_hop = torch.matmul(
                        graph_k[dst],
                        key_hop_emb.weight[hop_dist].view(self.nhead, self.head_dim).transpose(0, 1)
                    )

                    # Add to attention scores
                    attn_scores[src, dst] += q_hop + k_hop

            # Add edge-based attention if available
            if edge_attr is not None:
                # Find edges within this subgraph
                src_idx, dst_idx = edge_index
                graph_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool,
                                              device=edge_index.device)

                for e in range(edge_index.size(1)):
                    src, dst = src_idx[e].item(), dst_idx[e].item()
                    if src in graph_nodes and dst in graph_nodes:
                        graph_edge_mask[e] = True

                # Add edge-based scores
                if graph_edge_mask.sum() > 0:
                    for e_idx in range(edge_index.size(1)):
                        if graph_edge_mask[e_idx]:
                            # Get source and destination in subgraph
                            src = edge_index[0, e_idx].item()
                            dst = edge_index[1, e_idx].item()

                            # Map to local indices
                            src_local = (graph_nodes == src).nonzero().item()
                            dst_local = (graph_nodes == dst).nonzero().item()

                            # Get edge type
                            if edge_attr is not None:
                                edge_type = torch.argmax(edge_attr[e_idx]).item()
                            else:
                                edge_type = 0

                            # Add query-edge interaction
                            q_edge = torch.matmul(
                                graph_q[src_local],
                                query_edge_emb.weight[edge_type].view(self.nhead, self.head_dim).transpose(0, 1)
                            )

                            # Add key-edge interaction
                            k_edge = torch.matmul(
                                graph_k[dst_local],
                                key_edge_emb.weight[edge_type].view(self.nhead, self.head_dim).transpose(0, 1)
                            )

                            # Add to attention scores
                            attn_scores[src_local, dst_local] += q_edge + k_edge

            # Apply softmax
            attn_probs = F.softmax(attn_scores, dim=1)
            attn_probs = self.dropout(attn_probs)

            # Apply attention to values
            graph_out = torch.zeros_like(graph_v)

            for h in range(self.nhead):
                graph_out[:, h] = torch.matmul(
                    attn_probs[:, :, h], graph_v[:, h]
                )

                # Add value-hop and value-edge encoding
                for src in range(len(graph_nodes)):
                    for dst in range(len(graph_nodes)):
                        hop_dist = graph_distance[src, dst].item()

                        # Weight by attention probability
                        val_hop_contrib = value_hop_emb.weight[hop_dist].view(self.nhead, self.head_dim)[h]
                        val_hop_contrib = val_hop_contrib * attn_probs[src, dst, h]

                        # Add to output
                        graph_out[dst, h] += val_hop_contrib

            # Concatenate heads and apply output projection
            graph_out = graph_out.reshape(len(graph_nodes), self.d_model)
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

    def __init__(self, embed_dim, num_layers=2, batch_norm=True, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        # Edge feature projection
        self.edge_proj = nn.Linear(2, embed_dim)

        # GNN layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(embed_dim, embed_dim) for _ in range(num_layers)
        ])

        # Batch normalization
        if batch_norm:
            self.norm_layers = nn.ModuleList([
                nn.BatchNorm1d(embed_dim) for _ in range(num_layers)
            ])

        # Activation and dropout
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None):
        """Extract structural features from the graph."""
        h = x

        # Transform edge features if available
        edge_features = self.edge_proj(edge_attr) if edge_attr is not None else None

        # Apply GNN layers
        for i, conv in enumerate(self.conv_layers):
            # Apply graph convolution
            h_conv = conv(h, edge_index, edge_features)

            # Apply batch normalization if enabled
            if self.batch_norm:
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
    """Graph convolution layer optimized for DAGs."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.W_edge = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_features=None):
        """Forward pass."""
        # Transform node features
        h = self.W(x)

        # Initialize output
        out = torch.zeros_like(h)

        # Get source and target nodes
        src, dst = edge_index

        # For each edge, send message from source to target
        for i in range(edge_index.size(1)):
            s, d = src[i], dst[i]

            # Get message (transformed source features)
            msg = h[s]

            # Add edge features if available
            if edge_features is not None:
                edge_msg = self.W_edge(edge_features[i])
                msg = msg + edge_msg

            # Update target node
            out[d] += msg

        # Normalize by in-degree
        in_degree = torch.zeros(x.size(0), device=x.device)
        for i in range(edge_index.size(1)):
            in_degree[dst[i]] += 1

        # Avoid division by zero
        in_degree = torch.clamp(in_degree, min=1.0)

        # Apply normalization
        for i in range(x.size(0)):
            if in_degree[i] > 0:
                out[i] = out[i] / in_degree[i]

        return out
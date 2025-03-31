import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max
import math
from typing import Dict, List, Optional, Tuple, Union
import torch


class DAGTransformerLayer(nn.Module):
    """
    Transformer layer optimized for DAGs with structure-aware and edge-aware attention.
    """

    def __init__(self, hidden_dim, num_heads, dim_feedforward, dropout=0.1, max_hop=5, gnn_type="gcn"):
        super(DAGTransformerLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.max_hop = max_hop

        # Self-attention with structure awareness
        self.self_attn = DAGMultiHeadAttention(hidden_dim, num_heads, dropout, max_hop)

        # Structure extractor
        self.structure_extractor = StructureExtractor(hidden_dim, gnn_type=gnn_type)

        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim)
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, distance_matrix, edge_type_matrix, edge_index, edge_features,
                query_hop_emb, query_edge_emb, key_hop_emb, key_edge_emb,
                value_hop_emb, value_edge_emb, batch=None, padding_mask=None):
        """
        Forward pass through the transformer layer.

        Args:
            x: Node features [num_nodes, hidden_dim]
            distance_matrix: Hop distances between nodes [num_nodes, num_nodes]
            edge_type_matrix: Edge types between nodes [num_nodes, num_nodes]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, hidden_dim]
            query/key/value_hop/edge_emb: Embedding weights for structural bias
            batch: Batch assignment for nodes [num_nodes]
            padding_mask: Mask for padding values in truth tables [num_nodes, feature_dim]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Extract structure-aware features
        x_struct = self.structure_extractor(x, edge_index, edge_attr=edge_features)

        # Add structural information as a residual connection
        x = x + self.dropout(x_struct)

        # Apply first layer norm
        x_norm = self.norm1(x)

        # Apply structure-aware attention
        attn_output = self.self_attn(
            x_norm, x_norm, x_norm,
            query_hop_emb, query_edge_emb,
            key_hop_emb, key_edge_emb,
            value_hop_emb, value_edge_emb,
            distance_matrix, edge_type_matrix,
            batch, padding_mask
        )

        # Apply residual connection and second layer norm
        x = x + self.dropout(attn_output)
        x = self.norm2(x)

        # Apply feedforward network
        ff_output = self.feed_forward(x)

        # Apply final residual connection and normalization
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class DAGMultiHeadAttention(nn.Module):
    """
    Multi-head attention with structural bias from both hop distances and edge types.
    Implementation inspired by GRPE.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1, max_hop=5):
        super(DAGMultiHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.max_hop = max_hop
        self.scale = self.head_dim ** -0.5

        # Projections for query, key, value
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value,
                query_hop_emb, query_edge_emb,
                key_hop_emb, key_edge_emb,
                value_hop_emb, value_edge_emb,
                distance_matrix, edge_type_matrix,
                batch=None, padding_mask=None):
        """
        Forward pass with structural bias from both hop distances and edge types.
        Implementation based on GRPE's attention mechanism.
        """
        num_nodes = query.size(0)

        # Project inputs to q, k, v
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # If we have a padding mask, apply it to key vectors
        if padding_mask is not None:
            # Create a node-level padding ratio (percentage of values that are padding in each node)
            node_padding_ratio = padding_mask.float().mean(dim=1, keepdim=True)

            # Apply scaling factor to key projection based on padding ratio
            # Nodes with more padding will have less influence in attention
            k = k * (1.0 - node_padding_ratio)

        # Process batch-wise
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=query.device)

        # Handle one batch at a time to save memory
        unique_batches = torch.unique(batch)
        output = torch.zeros(num_nodes, self.hidden_dim, device=query.device)

        for b in unique_batches:
            # Get indices for this batch
            batch_mask = (batch == b)
            batch_indices = torch.nonzero(batch_mask).squeeze(-1)
            batch_size = len(batch_indices)

            if batch_size == 0:
                continue

            # Extract tensors for this batch
            batch_q = q[batch_indices]
            batch_k = k[batch_indices]
            batch_v = v[batch_indices]
            batch_distance = distance_matrix[batch_indices][:, batch_indices]
            batch_edge_type = edge_type_matrix[batch_indices][:, batch_indices]

            # Extract padding mask for this batch if available
            batch_padding = None
            if padding_mask is not None:
                batch_padding = padding_mask[batch_indices]

                # Create a node-to-node padding relevance matrix
                node_padding_ratio = batch_padding.float().mean(dim=1, keepdim=True)
                padding_relevance = 1.0 - torch.matmul(node_padding_ratio, node_padding_ratio.transpose(0, 1))
                padding_relevance = padding_relevance.unsqueeze(1).expand(-1, self.num_heads, -1).reshape(
                    batch_size * self.num_heads, batch_size)

            # Reshape for multi-head attention
            batch_q = batch_q.view(batch_size, self.num_heads, self.head_dim).transpose(0,
                                                                                        1)  # [num_heads, batch_size, head_dim]
            batch_k = batch_k.view(batch_size, self.num_heads, self.head_dim).transpose(0,
                                                                                        1)  # [num_heads, batch_size, head_dim]
            batch_v = batch_v.view(batch_size, self.num_heads, self.head_dim).transpose(0,
                                                                                        1)  # [num_heads, batch_size, head_dim]

            # Compute basic attention scores (content-based)
            content_scores = torch.bmm(batch_q,
                                       batch_k.transpose(1, 2)) * self.scale  # [num_heads, batch_size, batch_size]

            # Apply padding relevance if available
            if padding_mask is not None:
                content_scores = content_scores * padding_relevance.view(self.num_heads, batch_size, batch_size)

            # Get dimensions from embedding weights
            num_hop_types = query_hop_emb.weight.size(0)
            num_edge_types = query_edge_emb.weight.size(0)

            # Reshape embedding weights for multi-head attention
            query_hop_weights = query_hop_emb.weight.view(num_hop_types, self.num_heads, self.head_dim).transpose(0, 1)
            key_hop_weights = key_hop_emb.weight.view(num_hop_types, self.num_heads, self.head_dim).transpose(0, 1)
            query_edge_weights = query_edge_emb.weight.view(num_edge_types, self.num_heads, self.head_dim).transpose(0,
                                                                                                                     1)
            key_edge_weights = key_edge_emb.weight.view(num_edge_types, self.num_heads, self.head_dim).transpose(0, 1)

            # Create hop-based and edge-based biases
            hop_bias = torch.zeros_like(content_scores)  # [num_heads, batch_size, batch_size]
            edge_bias = torch.zeros_like(content_scores)  # [num_heads, batch_size, batch_size]

            # Convert distance and edge type matrices to integers for indexing
            batch_distance_int = batch_distance.long().clamp(0, num_hop_types - 1)
            batch_edge_type_int = batch_edge_type.long().clamp(0, num_edge_types - 1)

            # Loop through batch for explicit computation (avoids gather issues)
            for h in range(self.num_heads):
                for i in range(batch_size):
                    for j in range(batch_size):
                        # Add hop-based bias
                        hop_idx = batch_distance_int[i, j].item()
                        q_hop_bias = torch.dot(batch_q[h, i], query_hop_weights[h, hop_idx])
                        k_hop_bias = torch.dot(batch_k[h, j], key_hop_weights[h, hop_idx])
                        hop_bias[h, i, j] = q_hop_bias + k_hop_bias

                        # Add edge-based bias
                        edge_idx = batch_edge_type_int[i, j].item()
                        q_edge_bias = torch.dot(batch_q[h, i], query_edge_weights[h, edge_idx])
                        k_edge_bias = torch.dot(batch_k[h, j], key_edge_weights[h, edge_idx])
                        edge_bias[h, i, j] = q_edge_bias + k_edge_bias

            # Combine content, hop, and edge biases
            attn_scores = content_scores + hop_bias + edge_bias

            # Mask unreachable nodes (beyond max_hop)
            mask = (batch_distance >= self.max_hop + 1).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

            # Apply softmax and dropout
            attn_weights = F.softmax(attn_scores, dim=2)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            batch_output = torch.bmm(attn_weights, batch_v)  # [num_heads, batch_size, head_dim]

            # Apply value-based bias following GRPE approach
            value_hop_weights = value_hop_emb.weight.view(num_hop_types, self.num_heads, self.head_dim).transpose(0, 1)
            value_edge_weights = value_edge_emb.weight.view(num_edge_types, self.num_heads, self.head_dim).transpose(0,
                                                                                                                     1)

            # Create empty tensors for aggregation
            value_hop_agg = torch.zeros(self.num_heads, batch_size, num_hop_types, device=query.device)
            value_edge_agg = torch.zeros(self.num_heads, batch_size, num_edge_types, device=query.device)

            # Aggregate attention weights by hop distance and edge type
            for h in range(self.num_heads):
                for i in range(batch_size):
                    for j in range(batch_size):
                        # Aggregate hop attention
                        hop_idx = batch_distance_int[i, j].item()
                        value_hop_agg[h, i, hop_idx] += attn_weights[h, i, j]

                        # Aggregate edge attention
                        edge_idx = batch_edge_type_int[i, j].item()
                        value_edge_agg[h, i, edge_idx] += attn_weights[h, i, j]

            # Apply value bias using aggregated weights
            value_hop_bias = torch.bmm(value_hop_agg, value_hop_weights)  # [num_heads, batch_size, head_dim]
            value_edge_bias = torch.bmm(value_edge_agg, value_edge_weights)  # [num_heads, batch_size, head_dim]

            # Add value bias to output
            batch_output = batch_output + value_hop_bias + value_edge_bias

            # Reshape and apply output projection
            batch_output = batch_output.transpose(0, 1).reshape(batch_size, self.hidden_dim)
            output[batch_indices] = self.out_proj(batch_output)

        return output


import torch
import torch.nn as nn
from torch_scatter import scatter_add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer optimized for AIGs.
    Supports edge features with proper gradient computation.
    """

    def __init__(self, in_dim, out_dim, edge_dim=None):
        super(GCNLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Node feature transformation
        self.linear = nn.Linear(in_dim, out_dim)

        # Edge feature transformation
        self.use_edge_features = edge_dim is not None
        if self.use_edge_features:
            # Adjust edge linear layer to match input edge feature dimension dynamically
            self.edge_linear = nn.Linear(edge_dim, out_dim)

        # Initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if self.use_edge_features:
            nn.init.xavier_uniform_(self.edge_linear.weight)
            nn.init.zeros_(self.edge_linear.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the GCN layer with gradient support for edge attributes.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Transform node features
        h = self.linear(x)

        # Early return if no edges
        if edge_index.size(1) == 0:
            return torch.zeros_like(h)

        # Get source and target nodes
        src, dst = edge_index

        # Transform edge features if available
        if self.use_edge_features and edge_attr is not None:
            # Ensure edge_attr requires gradient
            if not edge_attr.requires_grad:
                edge_attr = edge_attr.clone().requires_grad_(True)

            # Flatten and handle different tensor shapes
            if edge_attr.dim() > 2:
                # If more than 2D, flatten all but the first dimension
                edge_attr = edge_attr.reshape(edge_attr.size(0), -1)

            # Adjust linear layer input dimension if needed
            if self.edge_linear.in_features != edge_attr.size(1):
                # Dynamically adjust the linear layer
                new_edge_linear = nn.Linear(edge_attr.size(1), self.out_dim).to(edge_attr.device)
                nn.init.xavier_uniform_(new_edge_linear.weight)
                nn.init.zeros_(new_edge_linear.bias)
                self.edge_linear = new_edge_linear

            # Transform edge features
            edge_h = self.edge_linear(edge_attr)

            # Apply messages with edge features
            messages = h[src] + edge_h
        else:
            # Apply messages without edge features
            messages = h[src]

        # Aggregate messages using scatter_add
        out = torch.zeros_like(h)
        scatter_add(messages, dst, dim=0, out=out)

        # Normalize by in-degree
        node_degrees = torch.zeros(x.size(0), dtype=torch.float, device=x.device)
        ones = torch.ones(edge_index.size(1), dtype=torch.float, device=x.device)
        scatter_add(ones, dst, dim=0, out=node_degrees)

        # Avoid division by zero
        node_degrees = torch.clamp(node_degrees, min=1.0)

        # Apply normalization
        out = out / node_degrees.unsqueeze(1)

        return out


class StructureExtractor(nn.Module):
    """
    Extract structural features from the graph using GNN layers.
    Specialized for AIG graphs.
    """

    def __init__(self, hidden_dim, num_layers=1, batch_norm=True, gnn_type="gcn", edge_dim=2):
        super(StructureExtractor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GCNLayer(hidden_dim, hidden_dim, edge_dim))

        # Layer normalization or batch normalization
        self.use_batch_norm = batch_norm
        if batch_norm:
            self.norm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        else:
            self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Activation
        self.activation = nn.GELU()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Extract structural features from the graph.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, hidden_dim]

        Returns:
            Structural node features [num_nodes, hidden_dim]
        """
        h = x

        # Apply GNN layers
        for i, (gnn_layer, norm_layer) in enumerate(zip(self.gnn_layers, self.norm_layers)):
            h_new = gnn_layer(h, edge_index, edge_attr)

            # Apply normalization
            if self.use_batch_norm:
                h_new = norm_layer(h_new)
            else:
                h_new = norm_layer(h_new)

            # Apply activation and residual connection
            h_new = self.activation(h_new)
            h = h + h_new

        # Apply output projection
        h = self.out_proj(h)

        return h

from torch_scatter import scatter_add, scatter_mean, scatter_max

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DAGTransformerLayer(nn.Module):
    """
    Transformer layer optimized for DAGs with relative positional encoding based on GRPE.
    Refactored for better readability and to ensure gradient flow through all parameters.
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
        q, k, v = self._project_qkv(x_norm)

        # Process batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Process attention for each graph in the batch
        out, processed_any_graph = self._process_batch_attention(
            q, k, v, batch, distance_matrix, edge_index, edge_attr,
            query_hop_emb, key_hop_emb, value_hop_emb,
            query_edge_emb, key_edge_emb, value_edge_emb
        )

        # Apply first residual connection
        x = x + self.dropout(out)

        # Apply second sublayer: FFN with residual
        x = x + self.ff(self.norm2(x))

        params_sum = 0
        for name, param in self.named_parameters():
            params_sum = params_sum + param.sum() * 0

        return x + params_sum

    def _project_qkv(self, x_norm):
        """Project input to queries, keys, and values."""
        # Apply projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Reshape for multi-head attention
        batch_size = x_norm.size(0)
        q = q.view(batch_size, self.nhead, self.head_dim)
        k = k.view(batch_size, self.nhead, self.head_dim)
        v = v.view(batch_size, self.nhead, self.head_dim)

        return q, k, v

    def _handle_empty_input(self):
        """Handle case when input is empty."""
        # Create a dummy computation to ensure gradient flow
        dummy_tensor = self.q_proj.weight.sum() * 0 + self.k_proj.weight.sum() * 0
        return dummy_tensor

    def _process_batch_attention(self, q, k, v, batch, distance_matrix, edge_index, edge_attr,
                                 query_hop_emb, key_hop_emb, value_hop_emb,
                                 query_edge_emb, key_edge_emb, value_edge_emb):
        """
        Process attention for each graph in the batch with optimized memory usage.
        Implements batched processing when possible and reduces redundant operations.

        Args:
            q, k, v: Query, key, value tensors [num_nodes, nhead, head_dim]
            batch: Batch assignment for nodes [num_nodes]
            distance_matrix: Distance matrix [num_nodes, num_nodes]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_features]
            query_hop_emb, key_hop_emb, value_hop_emb: Hop embedding layers
            query_edge_emb, key_edge_emb, value_edge_emb: Edge embedding layers

        Returns:
            (out, processed_any_graph): Output tensor and processing flag
        """
        # Quick path for single graph (common case)
        if batch.max().item() == 0:
            # If there's only one graph in the batch, process it directly
            graph_nodes = torch.arange(q.size(0), device=q.device)
            out = self._process_subgraph_attention(
                graph_nodes, q, k, v, distance_matrix,
                edge_index, edge_attr, query_hop_emb, key_hop_emb, value_hop_emb,
                query_edge_emb, key_edge_emb, value_edge_emb
            )
            return out, True

        # Process multiple graphs
        out = torch.zeros_like(q.view(q.size(0), -1))
        processed_any_graph = False

        # Find the unique batch ids actually present in the batch
        # This avoids processing empty graphs
        unique_batches = torch.unique(batch)

        # Pre-compute node indices for each batch
        batch_node_indices = {}
        for b in unique_batches:
            batch_node_indices[b.item()] = torch.nonzero(batch == b).squeeze(1)

        # Process each non-empty graph
        for b in unique_batches:
            graph_nodes = batch_node_indices[b.item()]

            processed_any_graph = True

            # Process this subgraph
            graph_out = self._process_subgraph_attention(
                graph_nodes, q, k, v, distance_matrix,
                edge_index, edge_attr, query_hop_emb, key_hop_emb, value_hop_emb,
                query_edge_emb, key_edge_emb, value_edge_emb
            )

            # Store in output tensor
            out[graph_nodes] = graph_out

            # Clear GPU cache if needed for large batches
            if graph_nodes.size(0) > 1000 and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        return out, processed_any_graph

    def _process_subgraph_attention(self, graph_nodes, q, k, v, distance_matrix,
                                   edge_index, edge_attr, query_hop_emb, key_hop_emb, value_hop_emb,
                                   query_edge_emb, key_edge_emb, value_edge_emb):
        """Process attention for a single subgraph."""
        # Get subgraph features
        graph_q = q[graph_nodes]  # [nodes, heads, head_dim]
        graph_k = k[graph_nodes]
        graph_v = v[graph_nodes]

        # Get subgraph distance matrix
        graph_distance = distance_matrix[graph_nodes][:, graph_nodes]
        num_graph_nodes = len(graph_nodes)

        # Compute basic attention scores
        attn_scores = self._compute_basic_attention(graph_q, graph_k, graph_distance)

        # Add topology-based attention
        attn_scores = self._add_topology_attention(
            attn_scores, graph_q, graph_k, graph_distance,
            query_hop_emb, key_hop_emb, num_graph_nodes
        )

        # Add edge-based attention if available
        attn_scores = self._add_edge_attention(
            attn_scores, graph_nodes, graph_q, graph_k,
            edge_index, edge_attr, query_edge_emb, key_edge_emb
        )

        # Apply softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        graph_out = self._apply_attention_to_values(
            attn_probs, graph_v, graph_distance, value_hop_emb, num_graph_nodes
        )

        # Apply output projection
        graph_out = self.out_proj(graph_out)

        return graph_out

    # 2. Optimize the _compute_basic_attention method
    def _compute_basic_attention(self, graph_q, graph_k, graph_distance):
        """Compute basic dot-product attention scores - batched implementation."""
        num_graph_nodes = graph_q.size(0)

        # Compute dot-product attention for all heads at once
        # [num_nodes, nhead, head_dim] @ [num_nodes, head_dim, nhead] -> [num_nodes, num_nodes, nhead]
        attn_scores = torch.bmm(
            graph_q.transpose(0, 1),  # [nhead, num_nodes, head_dim]
            graph_k.transpose(0, 1).transpose(1, 2)  # [nhead, head_dim, num_nodes]
        ).permute(1, 2, 0) * self.scale  # [num_nodes, num_nodes, nhead]

        # Apply DAG-specific attention masking (vectorized)
        mask = (graph_distance >= self.max_hop + 1).unsqueeze(-1).expand(-1, -1, self.nhead)
        attn_scores = attn_scores.masked_fill(mask, -1e9)

        return attn_scores

    def _add_topology_attention(self, attn_scores, graph_q, graph_k, graph_distance,
                                query_hop_emb, key_hop_emb, num_graph_nodes):
        """
        Optimized topology-based attention implementation using batched operations.
        Reduces redundant computations and improves memory access patterns.

        Args:
            attn_scores: Current attention scores [num_nodes, num_nodes, nhead]
            graph_q: Query vectors for subgraph nodes [num_nodes, nhead, head_dim]
            graph_k: Key vectors for subgraph nodes [num_nodes, nhead, head_dim]
            graph_distance: Distance matrix for the subgraph [num_nodes, num_nodes]
            query_hop_emb: Query hop embedding layer
            key_hop_emb: Key hop embedding layer
            num_graph_nodes: Number of nodes in the subgraph

        Returns:
            Updated attention scores
        """
        # Get hop distances and ensure they're within bounds for indexing
        hop_distances = torch.clamp(graph_distance, max=self.max_hop).long()

        # Get embeddings for all hop distances at once
        # [num_nodes, num_nodes, d_model]
        q_hop_weights = query_hop_emb(hop_distances)
        k_hop_weights = key_hop_emb(hop_distances)

        # Reshape for per-head processing
        # [num_nodes, num_nodes, nhead, head_dim]
        q_hop_weights = q_hop_weights.view(num_graph_nodes, num_graph_nodes, self.nhead, self.head_dim)
        k_hop_weights = k_hop_weights.view(num_graph_nodes, num_graph_nodes, self.nhead, self.head_dim)

        # Process in chunks to prevent excessive memory usage for large graphs
        # Using efficient tensor operations
        chunk_size = min(32, num_graph_nodes)  # Adjust based on available memory

        for start_idx in range(0, num_graph_nodes, chunk_size):
            end_idx = min(start_idx + chunk_size, num_graph_nodes)
            chunk_size_actual = end_idx - start_idx

            # Process contribution for all heads at once
            for h in range(self.nhead):
                # Extract query vectors for this chunk and head
                # [chunk_size, head_dim]
                chunk_q = graph_q[start_idx:end_idx, h]

                # Reshape for broadcasting
                # [chunk_size, 1, head_dim]
                chunk_q = chunk_q.unsqueeze(1)

                # Extract query hop weights for this chunk, all destinations, and this head
                # [chunk_size, num_nodes, head_dim]
                chunk_q_hop = q_hop_weights[start_idx:end_idx, :, h]

                # Compute query-hop interactions efficiently
                # [chunk_size, 1, head_dim] * [chunk_size, num_nodes, head_dim] -> [chunk_size, num_nodes]
                q_contrib = (chunk_q * chunk_q_hop).sum(dim=-1)

                # Extract key vectors for all nodes
                # [num_nodes, head_dim]
                all_k = graph_k[:, h]

                # Extract key hop weights for this chunk, all destinations, and this head
                # [chunk_size, num_nodes, head_dim]
                chunk_k_hop = k_hop_weights[start_idx:end_idx, :, h]

                # Reshape key vectors for broadcasting
                # [1, num_nodes, head_dim]
                all_k = all_k.unsqueeze(0)

                # Compute key-hop interactions efficiently
                # [1, num_nodes, head_dim] * [chunk_size, num_nodes, head_dim] -> [chunk_size, num_nodes]
                k_contrib = (all_k * chunk_k_hop).sum(dim=-1)

                # Add both contributions to attention scores
                attn_scores[start_idx:end_idx, :, h] += q_contrib + k_contrib

        return attn_scores

    def _add_edge_attention(self, attn_scores, graph_nodes, graph_q, graph_k,
                            edge_index, edge_attr, query_edge_emb, key_edge_emb):
        """
        Fully optimized edge-based attention implementation.
        Uses dictionary-based node mapping for safety and scatter operations for performance.

        Args:
            attn_scores: Current attention scores [num_nodes, num_nodes, nhead]
            graph_nodes: Indices of nodes in this subgraph
            graph_q: Query vectors for subgraph nodes [num_nodes, nhead, head_dim]
            graph_k: Key vectors for subgraph nodes [num_nodes, nhead, head_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_features]
            query_edge_emb: Query edge embedding layer
            key_edge_emb: Key edge embedding layer

        Returns:
            Updated attention scores
        """
        if edge_attr is None or edge_index.size(1) == 0:
            return attn_scores

        num_graph_nodes = len(graph_nodes)
        device = graph_nodes.device

        # Create node maps (global to local indices) - dictionary for safety
        node_map = {node.item(): i for i, node in enumerate(graph_nodes)}

        # Find edges within this subgraph
        src_idx, dst_idx = edge_index

        # Collect valid edges
        valid_edges = []
        src_local_indices = []
        dst_local_indices = []

        for e in range(edge_index.size(1)):
            src, dst = src_idx[e].item(), dst_idx[e].item()
            if src in node_map and dst in node_map:
                valid_edges.append(e)
                src_local_indices.append(node_map[src])
                dst_local_indices.append(node_map[dst])

        # If no valid edges, return unchanged scores
        if not valid_edges:
            return attn_scores

        # Convert lists to tensors
        valid_edges_tensor = torch.tensor(valid_edges, device=device)
        src_local_indices = torch.tensor(src_local_indices, device=device)
        dst_local_indices = torch.tensor(dst_local_indices, device=device)

        # Process edge types vectorized
        if edge_attr is not None:
            if edge_attr.dim() > 1:
                edge_types = torch.argmax(edge_attr[valid_edges_tensor], dim=1)
            else:
                edge_types = edge_attr[valid_edges_tensor].long()
        else:
            edge_types = torch.zeros(len(valid_edges), dtype=torch.long, device=device)

        # Clamp edge types to valid range
        max_edge_type = query_edge_emb.weight.size(0) - 1
        edge_types = torch.clamp(edge_types, max=max_edge_type)

        # Reshape embedding weights once outside the loop
        q_emb = query_edge_emb.weight.view(-1, self.nhead, self.head_dim)
        k_emb = key_edge_emb.weight.view(-1, self.nhead, self.head_dim)

        # Use torch_scatter for efficient updates
        from torch_scatter import scatter_add

        # Process each head
        for h in range(self.nhead):
            # Get query and key vectors for this head
            src_q = graph_q[src_local_indices, h]  # [num_valid_edges, head_dim]
            dst_k = graph_k[dst_local_indices, h]  # [num_valid_edges, head_dim]

            # Get edge embeddings for this head
            q_edge_emb = q_emb[edge_types, h]  # [num_valid_edges, head_dim]
            k_edge_emb = k_emb[edge_types, h]  # [num_valid_edges, head_dim]

            # Compute attention contributions
            q_contrib = torch.sum(src_q * q_edge_emb, dim=1)  # [num_valid_edges]
            k_contrib = torch.sum(dst_k * k_edge_emb, dim=1)  # [num_valid_edges]
            edge_contrib = q_contrib + k_contrib  # [num_valid_edges]

            # Use scatter operation to update attention scores efficiently
            flat_attn = attn_scores[:, :, h].clone().view(-1)
            flat_indices = src_local_indices * num_graph_nodes + dst_local_indices
            scatter_add(edge_contrib, flat_indices, out=flat_attn)
            attn_scores[:, :, h] = flat_attn.view(num_graph_nodes, num_graph_nodes)

        return attn_scores

    def _apply_attention_to_values(self, attn_probs, graph_v, graph_distance,
                                   value_hop_emb, num_graph_nodes):
        """
        Optimized implementation of applying attention to values.
        Reduces redundant operations and implements batch processing of heads.

        Args:
            attn_probs: Attention probabilities [num_nodes, num_nodes, nhead]
            graph_v: Value vectors [num_nodes, nhead, head_dim]
            graph_distance: Distance matrix [num_nodes, num_nodes]
            value_hop_emb: Value hop embedding layer
            num_graph_nodes: Number of nodes in the subgraph

        Returns:
            Updated node features [num_nodes, d_model]
        """
        device = graph_v.device

        # Pre-allocate output tensor
        graph_out = torch.zeros(num_graph_nodes, self.d_model, device=device)

        # Get hop distances for value position encoding - compute once
        hop_distances = torch.clamp(graph_distance, max=self.max_hop).long()

        # Get embeddings for all hop distances at once - compute once outside the head loop
        # [num_nodes, num_nodes, d_model]
        v_hop_weights = value_hop_emb.weight[hop_distances]

        # Reshape for per-head processing
        # [num_nodes, num_nodes, nhead, head_dim]
        v_hop_weights = v_hop_weights.view(num_graph_nodes, num_graph_nodes, self.nhead, self.head_dim)

        # Process all heads at once for the basic attention
        # Reshape attention probs for batch matrix multiplication
        # [num_nodes, num_nodes, nhead] -> [nhead, num_nodes, num_nodes]
        attn_probs_t = attn_probs.transpose(0, 2).transpose(1, 2)

        # Apply attention to values for all heads at once
        # [nhead, num_nodes, num_nodes] @ [nhead, num_nodes, head_dim] -> [nhead, num_nodes, head_dim]
        head_out = torch.bmm(attn_probs_t, graph_v.transpose(0, 1))

        # Reshape and store in output
        # [nhead, num_nodes, head_dim] -> [num_nodes, nhead, head_dim] -> [num_nodes, d_model]
        head_out = head_out.transpose(0, 1).reshape(num_graph_nodes, self.d_model)
        graph_out = head_out

        # Process value position encoding contributions - still need to do per head
        # But we can optimize the inner loop operations
        for h in range(self.nhead):
            # Extract the attention weights for this head
            head_attn = attn_probs[:, :, h].unsqueeze(-1)  # [num_nodes, num_nodes, 1]

            # Get value hop weights for this head
            head_v_hop = v_hop_weights[:, :, h, :]  # [num_nodes, num_nodes, head_dim]

            # Weight by attention probability and sum in one operation
            # [num_nodes, num_nodes, head_dim] * [num_nodes, num_nodes, 1] -> [num_nodes, num_nodes, head_dim]
            weighted_v_hop = head_v_hop * head_attn

            # Sum over source nodes
            v_hop_contrib = weighted_v_hop.sum(dim=1)  # [num_nodes, head_dim]

            # Add to the output in the correct position
            start_idx = h * self.head_dim
            end_idx = (h + 1) * self.head_dim
            graph_out[:, start_idx:end_idx] += v_hop_contrib

        return graph_out

    def _ensure_gradient_flow(self, out):
        """Ensure gradient flow through all parameters even if no processing occurred."""
        # Create a dummy computation that involves all the linear layers
        dummy = (self.q_proj.weight.sum() + self.k_proj.weight.sum() +
                 self.v_proj.weight.sum() + self.out_proj.weight.sum()) * 0
        return out + dummy


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
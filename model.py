import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


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

        # Multi-head attention with edge features
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

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

        # Project edge features
        edge_weights = self.edge_proj(edge_attr) if edge_attr is not None else None  # [num_edges, num_heads]

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

            # Get edges for this graph
            if edge_index.size(1) > 0:  # Check that there are edges
                graph_edge_mask = (edge_index[0, :] >= graph_nodes.min()) & \
                                  (edge_index[0, :] <= graph_nodes.max()) & \
                                  (edge_index[1, :] >= graph_nodes.min()) & \
                                  (edge_index[1, :] <= graph_nodes.max())

                if graph_edge_mask.sum() > 0 and edge_weights is not None:
                    graph_edges = edge_index[:, graph_edge_mask]
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


class AIGTransformer(nn.Module):
    def __init__(
            self,
            node_features=4,
            edge_features=2,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            max_nodes=120
    ):
        super(AIGTransformer, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Node feature embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)

        # Positional encoding (will be applied per graph)
        self.pos_encoding = nn.Parameter(torch.randn(max_nodes, hidden_dim))

        # Graph transformer layers
        self.layers = nn.ModuleList([
            EdgeAwareGraphTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                edge_dim=edge_features,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Node existence prediction
        self.node_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Task-specific head for node feature prediction
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features)
        )

        # Edge existence prediction
        self.edge_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Edge feature prediction
        self.edge_feature_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_features)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass for batched PyG Data objects with support for all masking modes

        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_features]
                - batch: Batch assignment [num_nodes]
                - node_mask: Node feature masking info [num_nodes]
                - edge_mask: Edge masking info [num_edges in target]
                - mask_mode: One of the five masking modes
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None
        node_mask = data.node_mask if hasattr(data, 'node_mask') else None
        edge_mask = data.edge_mask if hasattr(data, 'edge_mask') else None
        mask_mode = data.mask_mode if hasattr(data, 'mask_mode') else "node_feature"

        # Node feature embedding
        x = self.node_embedding(x)

        # Add positional encoding based on node position within each graph
        if batch is not None:
            for g in range(batch.max().item() + 1):
                graph_mask = batch == g
                num_nodes = graph_mask.sum().item()
                x[graph_mask] = x[graph_mask] + self.pos_encoding[:num_nodes]
        else:
            # Single graph case
            x = x + self.pos_encoding[:x.size(0)]

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch, node_mask)

        # Predict node features
        node_out = self.node_predictor(x)

        # Initialize results dictionary
        results = {
            'node_features': node_out,
            'mask': node_mask
        }

        # Handle different masking modes
        if mask_mode == "node_feature":
            # Only predict node features - standard behavior
            pass

        elif mask_mode == "edge_feature":
            # Predict edge features
            if hasattr(data, 'edge_index_target') and hasattr(data, 'edge_mask') and edge_mask.sum() > 0:
                edge_index_target = data.edge_index_target
                masked_edges = edge_index_target[:, edge_mask]

                # Ensure source and target nodes are within our indices
                valid_edges_mask = (masked_edges[0] < x.size(0)) & (masked_edges[1] < x.size(0))

                if valid_edges_mask.sum() > 0:
                    masked_edges = masked_edges[:, valid_edges_mask]

                    # Get node embeddings for source and target nodes
                    src_embeddings = x[masked_edges[0]]
                    dst_embeddings = x[masked_edges[1]]

                    # Concatenate embeddings
                    edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

                    # Predict edge features
                    edge_features = self.edge_feature_predictor(edge_embeddings)

                    # Store edge predictions
                    results['edge_preds'] = {
                        'masked_edges': masked_edges,
                        'edge_features': edge_features
                    }

        elif mask_mode == "node_existence":
            # Predict node features and node existence
            if hasattr(data, 'node_existence_mask') and data.node_existence_mask.sum() > 0:
                node_existence = self.node_existence_predictor(x)
                results['node_existence'] = node_existence
                results['node_existence_mask'] = data.node_existence_mask

        elif mask_mode == "edge_existence":
            # Predict edge features and edge existence
            if hasattr(data, 'edge_index_target') and hasattr(data, 'edge_mask') and edge_mask.sum() > 0:
                edge_index_target = data.edge_index_target
                masked_edges = edge_index_target[:, edge_mask]

                # Ensure source and target nodes are within our indices
                valid_edges_mask = (masked_edges[0] < x.size(0)) & (masked_edges[1] < x.size(0))

                if valid_edges_mask.sum() > 0:
                    masked_edges = masked_edges[:, valid_edges_mask]

                    # Get node embeddings for source and target nodes
                    src_embeddings = x[masked_edges[0]]
                    dst_embeddings = x[masked_edges[1]]

                    # Concatenate embeddings
                    edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

                    # Predict edge existence and features
                    edge_existence = self.edge_existence_predictor(edge_embeddings)
                    edge_features = self.edge_feature_predictor(edge_embeddings)

                    # Store edge predictions
                    results['edge_preds'] = {
                        'masked_edges': masked_edges,
                        'edge_existence': edge_existence,
                        'edge_features': edge_features
                    }

        elif mask_mode == "removal":
            # We need to map predictions back to original node indices
            # Predict existence for all nodes in the reduced graph
            node_existence = self.node_existence_predictor(x)
            results['node_existence'] = node_existence

            # Store mapping info for reconstruction
            if hasattr(data, 'original_to_new_indices'):
                results['original_to_new_indices'] = data.original_to_new_indices

            if hasattr(data, 'node_removal_mask'):
                results['node_removal_mask'] = data.node_removal_mask

            if hasattr(data, 'num_original_nodes'):
                results['num_original_nodes'] = data.num_original_nodes

            # Predict edge existence and features for reconstruction
            if hasattr(data, 'edge_index_target') and hasattr(data, 'edge_mask') and edge_mask.sum() > 0:
                # For edges that were connected to removed nodes
                # We need to map these to the remaining nodes
                edge_index_target = data.edge_index_target

                # We can't directly use the masked edges because some nodes are removed
                # Instead, we'll generate potential edge candidates from the current nodes
                if hasattr(data, 'old_to_new_mapping'):
                    # Generate new edge candidates for prediction
                    edge_preds = {}

                    # This is a complex case - in practice, you might want a more
                    # efficient approach than generating all possible edges
                    # For now, just store information needed for reconstruction
                    edge_preds['removed_nodes_info'] = {
                        'old_to_new_mapping': data.old_to_new_mapping,
                        'node_removal_mask': data.node_removal_mask,
                        'original_edges': edge_index_target
                    }

                    results['edge_preds'] = edge_preds

        return results



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
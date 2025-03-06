import torch
import torch.nn as nn
from layer import EdgeAwareGraphTransformerLayer
from torch.nn import functional as F

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

        # Improved type-specific positional encoding (for 4 node types)
        self.type_specific_pos_encoding = nn.Parameter(
            torch.randn(4, max_nodes, hidden_dim)
        )

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

        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        # Improved node feature prediction head
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, node_features)
        )

        # Improved edge feature prediction with components for skip connections
        self.edge_feat_down = nn.Linear(hidden_dim * 2, hidden_dim)
        self.edge_feat_norm1 = nn.LayerNorm(hidden_dim)
        self.edge_feat_mid = nn.Linear(hidden_dim, hidden_dim // 2)
        self.edge_feat_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.edge_feat_out = nn.Linear(hidden_dim // 2, edge_features)

        # Edge existence prediction (for connectivity mode)
        self.edge_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # Extract data attributes
        x, edge_index, batch, mask_mode = self._extract_data_attributes(data)
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        node_mask = data.node_mask if hasattr(data, 'node_mask') else None
        edge_mask = data.edge_mask if hasattr(data, 'edge_mask') else None

        # Node feature embedding
        x = self.node_embedding(x)

        # Apply type-specific positional encoding
        x = self._apply_positional_encoding(x, data, batch)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch, node_mask)

        # Add global context using multihead attention
        x = self._apply_global_context(x, batch)
        # Initialize an empty results dictionary
        results = {}

        # Handle different masking modes
        if mask_mode == "node_feature":
            # Predict node features only when in node_feature mode
            node_out = self.node_predictor(x)
            results['node_features'] = node_out
            # Only include node_mask in node_feature mode
            if node_mask is not None:
                results['mask'] = node_mask

        elif mask_mode == "edge_feature":
            self._handle_edge_feature_mode(data, x, edge_mask, results)

        elif mask_mode == "connectivity":
            self._handle_connectivity(data, x, edge_mask, results)
        else:
            raise ValueError(f"Unknown masking mode: {mask_mode}")

        return results

    def _extract_data_attributes(self, data):
        """Extract and return common data attributes."""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        mask_mode = data.mask_mode if hasattr(data, 'mask_mode') else "node_feature"
        return x, edge_index, batch, mask_mode

    def _apply_positional_encoding(self, x, data, batch):
        """Apply type-specific positional encoding to node embeddings."""
        # Extract the type from the first 3 bits of each node feature
        node_types = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for i in range(3):
            node_types += (data.x[:, i].long() * (2 ** (2 - i)))  # Convert binary to decimal

        # Apply type-specific positional encoding
        if batch is not None:
            for g in range(batch.max().item() + 1):
                graph_mask = batch == g
                num_nodes = graph_mask.sum().item()
                for type_idx in range(4):  # For each node type
                    type_mask = (node_types == type_idx) & graph_mask
                    if type_mask.sum() > 0:  # Only if we have nodes of this type
                        node_indices = torch.where(type_mask)[0] - torch.where(graph_mask)[0][0]  # Relative positions
                        pos_idx = torch.clamp(node_indices,
                                              max=num_nodes - 1)  # Ensure we don't exceed available positions
                        x[type_mask] = x[type_mask] + self.type_specific_pos_encoding[type_idx, pos_idx]
        else:
            # Single graph case
            num_nodes = x.size(0)
            for type_idx in range(4):
                type_mask = node_types == type_idx
                if type_mask.sum() > 0:
                    pos_idx = torch.arange(type_mask.sum(), device=x.device)
                    pos_idx = torch.clamp(pos_idx, max=num_nodes - 1)
                    x[type_mask] = x[type_mask] + self.type_specific_pos_encoding[type_idx, pos_idx]

        return x

    def _predict_edge_features(self, src_embeddings, dst_embeddings):
        """
        Improved edge feature prediction with deeper network and normalization.
        """
        # Concatenate embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

        # Apply the improved edge feature predictor with skip connections
        edge_feat = self.edge_feat_down(edge_embeddings)
        edge_feat = self.edge_feat_norm1(edge_feat)
        edge_feat = F.relu(edge_feat)
        edge_feat_mid = self.edge_feat_mid(edge_feat)
        edge_feat_mid = self.edge_feat_norm2(edge_feat_mid)
        edge_feat_mid = F.relu(edge_feat_mid)

        # Final prediction
        edge_features = self.edge_feat_out(edge_feat_mid)

        return edge_features

    def _handle_edge_feature_mode(self, data, x, edge_mask, results):
        """Handle edge feature prediction mode."""
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

                # Predict edge features using the improved predictor
                edge_features = self._predict_edge_features(src_embeddings, dst_embeddings)

                # Store edge predictions
                results['edge_preds'] = {
                    'masked_edges': masked_edges,
                    'edge_features': edge_features
                }

    def _handle_connectivity(self, data, x, edge_mask, results):
        """
        Handle connectivity prediction mode with candidate pairs.

        Args:
            data: PyG data object containing the masked graph
            x: Node embeddings
            edge_mask: Boolean mask indicating which edges are masked
            results: Dictionary to store predictions
        """
        if hasattr(data, 'all_candidate_pairs'):
            all_candidate_pairs = data.all_candidate_pairs

            # Get node embeddings for source and target nodes of all candidates
            src_embeddings = x[all_candidate_pairs[0]]
            dst_embeddings = x[all_candidate_pairs[1]]

            # Concatenate embeddings
            edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

            # Predict edge existence
            edge_existence = self.edge_existence_predictor(edge_embeddings)

            # Predict edge features using the improved predictor
            edge_features = self._predict_edge_features(src_embeddings, dst_embeddings)

            # Store edge predictions
            results['edge_preds'] = {
                'all_candidate_pairs': all_candidate_pairs,
                'edge_existence': edge_existence,
                'edge_features': edge_features
            }
        else:
            # Fallback for when all_candidate_pairs is not available
            # This might happen during testing or in older data formats
            if hasattr(data, 'masked_edge_node_pairs') and hasattr(data, 'connectivity_target'):
                masked_pairs = data.masked_edge_node_pairs

                # Get node embeddings for masked pairs
                src_embeddings = x[masked_pairs[0]]
                dst_embeddings = x[masked_pairs[1]]

                # Predict edge existence and features
                edge_existence = self.edge_existence_predictor(
                    torch.cat([src_embeddings, dst_embeddings], dim=1))
                edge_features = self._predict_edge_features(src_embeddings, dst_embeddings)

                # Store predictions
                results['edge_preds'] = {
                    'masked_edges': masked_pairs,
                    'edge_existence': edge_existence,
                    'edge_features': edge_features
                }


    def _apply_global_context(self, x, batch):
        """
        Apply global context to node embeddings using multihead attention.

        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Updated node embeddings with global context
        """
        if batch is None:
            # Single graph case - create a dummy batch
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Process each graph in the batch
        num_graphs = batch.max().item() + 1
        updated_x = x.clone()

        for g in range(num_graphs):
            # Get nodes for this graph
            graph_mask = batch == g
            graph_nodes = torch.nonzero(graph_mask).squeeze(1)

            if len(graph_nodes) == 0:
                continue

            # Get node features for this graph
            graph_x = x[graph_nodes]

            # Apply self-attention (all nodes attend to all other nodes)
            # Rearrange for nn.MultiheadAttention: [seq_len, batch_size, embedding_dim]
            graph_x_t = graph_x.unsqueeze(1).transpose(0, 1)  # [1, nodes, hidden_dim]

            # Apply global attention
            attn_output, _ = self.global_attention(
                query=graph_x_t,
                key=graph_x_t,
                value=graph_x_t
            )

            # Reshape back to [nodes, hidden_dim]
            attn_output = attn_output.transpose(0, 1).squeeze(1)

            # Update node embeddings with global context
            updated_x[graph_nodes] = attn_output

        return updated_x

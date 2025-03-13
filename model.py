import torch
import torch.nn as nn
from layers import *
from torch.nn import functional as F


class AIGTransformer(nn.Module):
    def __init__(
            self,
            node_features=4,
            edge_features=2,
            hidden_dim=128,  # [YOURS - Increased from 64]
            num_layers=3,  # [YOURS - Increased from 2]
            num_heads=4,  # [YOURS - Increased from 2]
            dropout=0.2,  # [YOURS - Increased from 0.1]
            max_nodes=120,
            max_hop=5  # [GRPE - New parameter for max hop distance]
    ):
        super(AIGTransformer, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.max_hop = max_hop
        self.num_heads = num_heads

        # [YOURS with improvements] Node feature embedding with batch normalization
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()  # [YOURS - GELU instead of ReLU]
        )

        # [GRPE] Task token for better graph-level representations
        self.task_token = nn.Parameter(torch.randn(1, hidden_dim))

        # [GRPE] Topology and edge encodings
        self.query_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)  # +3 for self, unreachable, task
        self.key_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)
        self.value_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)

        self.query_edge_emb = nn.Embedding(edge_features + 4, hidden_dim)  # +4 for special edges
        self.key_edge_emb = nn.Embedding(edge_features + 4, hidden_dim)
        self.value_edge_emb = nn.Embedding(edge_features + 4, hidden_dim)

        # [COMBINED] Transformer layers based on your design but with GRPE concepts
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(  # This is a custom layer you'll need to implement
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,  # [YOURS - Wider FFN]
                dropout=dropout,
                activation="gelu",  # [YOURS - GELU instead of ReLU]
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # [DAGformer] Structure extractor
        self.structure_extractor = StructureExtractor(
            hidden_dim,
            num_layers=2,
            batch_norm=True
        )

        # [YOURS - improved] Node feature prediction head
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, node_features)
        )

        # [YOURS] Edge feature prediction components
        self.edge_feat_down = nn.Linear(hidden_dim * 2, hidden_dim)
        self.edge_feat_norm1 = nn.LayerNorm(hidden_dim)
        self.edge_feat_mid = nn.Linear(hidden_dim, hidden_dim // 2)
        self.edge_feat_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.edge_feat_out = nn.Linear(hidden_dim // 2, edge_features)

        # [YOURS with improvements] Edge existence prediction
        self.edge_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # [YOURS - GELU instead of ReLU]
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),  # [YOURS - GELU instead of ReLU]
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # [YOURS - addition] Final layer norm for better training stability
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        # [YOURS] Extract data attributes
        x, edge_index, batch, mask_mode = self._extract_data_attributes(data)
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        node_mask = data.node_mask if hasattr(data, 'node_mask') else None
        edge_mask = data.edge_mask if hasattr(data, 'edge_mask') else None

        # [YOURS] Node feature embedding
        x = self.node_embedding(x)

        # [GRPE] Compute distance/hop matrix for relative positional encoding
        distance_matrix = self._compute_hop_distances(edge_index, x.size(0), batch)

        # Apply transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # [GRPE] Extract structural features
            x_struct = self.structure_extractor(x, edge_index, edge_attr)
            x = x + x_struct  # Residual connection with structural features

            # [COMBINED] Apply transformer layer with relative positional encoding
            x = layer(
                x=x,
                distance_matrix=distance_matrix,
                edge_index=edge_index,
                edge_attr=edge_attr,
                query_hop_emb=self.query_hop_emb,
                key_hop_emb=self.key_hop_emb,
                value_hop_emb=self.value_hop_emb,
                query_edge_emb=self.query_edge_emb,
                key_edge_emb=self.key_edge_emb,
                value_edge_emb=self.value_edge_emb,
                batch=batch
            )

        # [YOURS] Apply final normalization
        x = self.final_norm(x)

        # [YOURS] Initialize results dictionary
        results = {}

        # [YOURS] Handle different masking modes
        if mask_mode == "node_feature":
            node_out = self.node_predictor(x)
            results['node_features'] = node_out
            if node_mask is not None:
                results['mask'] = node_mask

        elif mask_mode == "edge_feature":
            self._handle_edge_feature_mode(data, x, edge_mask, results)

        elif mask_mode == "connectivity":
            self._handle_connectivity(data, x, edge_mask, results)

        else:
            raise ValueError(f"Unknown masking mode: {mask_mode}")

        return results

    # [YOURS] Extract data attributes
    def _extract_data_attributes(self, data):
        """Extract and return common data attributes."""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        mask_mode = data.mask_mode if hasattr(data, 'mask_mode') else "node_feature"
        return x, edge_index, batch, mask_mode

    # [GRPE] Compute hop distances between nodes
    def _compute_hop_distances(self, edge_index, num_nodes, batch=None):
        """
        Compute shortest path distances between all pairs of nodes.
        Returns a matrix of size [num_nodes, num_nodes].
        """
        device = edge_index.device

        # Initialize with "unreachable"
        distance_matrix = torch.full((num_nodes, num_nodes), self.max_hop + 2,
                                     dtype=torch.long, device=device)

        # Set diagonal to 0 (self-connections)
        indices = torch.arange(num_nodes, device=device)
        distance_matrix[indices, indices] = 0

        # Set direct connections to 1
        if edge_index.size(1) > 0:  # Check that there are edges
            distance_matrix[edge_index[0], edge_index[1]] = 1

        # Floyd-Warshall algorithm for shortest paths
        if batch is None:
            # Single graph case - standard Floyd-Warshall
            for k in range(num_nodes):
                # Use broadcasting for efficient computation
                update = distance_matrix[:, k:k + 1] + distance_matrix[k:k + 1, :]
                distance_matrix = torch.minimum(distance_matrix, update)
        else:
            # Process each graph in the batch separately
            for b in range(batch.max().item() + 1):
                b_mask = batch == b
                b_indices = torch.nonzero(b_mask).squeeze(1)
                if len(b_indices) == 0:
                    continue

                # Get submatrix for this graph
                for k in b_indices:
                    # Only update distances within the same graph
                    update_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)
                    update_mask[b_indices, :] = True
                    update_mask[:, b_indices] = True

                    # Use broadcasting only on the relevant submatrix
                    update = distance_matrix[:, k:k + 1] + distance_matrix[k:k + 1, :]

                    # Only update where the mask is True
                    distance_matrix = torch.where(
                        update_mask & (update < distance_matrix),
                        update,
                        distance_matrix
                    )

        # Clamp to max_hop
        distance_matrix = torch.clamp(distance_matrix, max=self.max_hop + 1)

        return distance_matrix

    # [YOURS with GELU improvement] Edge feature prediction
    def _predict_edge_features(self, src_embeddings, dst_embeddings):
        """Edge feature prediction with GELU activation."""
        # Concatenate embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

        # Apply predictor with skip connections
        edge_feat = self.edge_feat_down(edge_embeddings)
        edge_feat = self.edge_feat_norm1(edge_feat)
        edge_feat = F.gelu(edge_feat)  # [YOURS - GELU instead of ReLU]
        edge_feat_mid = self.edge_feat_mid(edge_feat)
        edge_feat_mid = self.edge_feat_norm2(edge_feat_mid)
        edge_feat_mid = F.gelu(edge_feat_mid)  # [YOURS - GELU instead of ReLU]

        # Final prediction
        edge_features = self.edge_feat_out(edge_feat_mid)

        return edge_features

    # [YOURS] Handle edge feature prediction mode
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

    # [YOURS] Handle connectivity prediction mode
    def _handle_connectivity(self, data, x, edge_mask, results):
        """Handle connectivity prediction mode."""
        if hasattr(data, 'all_candidate_pairs'):
            all_candidate_pairs = data.all_candidate_pairs

            # Get node embeddings for source and target nodes of all candidates
            src_embeddings = x[all_candidate_pairs[0]]
            dst_embeddings = x[all_candidate_pairs[1]]

            # Predict edge existence
            edge_existence = self.edge_existence_predictor(
                torch.cat([src_embeddings, dst_embeddings], dim=1))

            # Predict edge features
            edge_features = self._predict_edge_features(src_embeddings, dst_embeddings)

            # Store edge predictions
            results['edge_preds'] = {
                'all_candidate_pairs': all_candidate_pairs,
                'edge_existence': edge_existence,
                'edge_features': edge_features
            }
        else:
            # Fallback for when all_candidate_pairs is not available
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

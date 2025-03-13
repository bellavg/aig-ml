import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_add, scatter_mean, scatter_max
from layers import *



class AIGTransformer(nn.Module):
    def __init__(
            self,
            node_features=4,
            edge_features=2,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.2,
            max_nodes=120,
            max_hop=5
    ):
        super(AIGTransformer, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.max_hop = max_hop
        self.num_heads = num_heads

        # Node feature embedding with batch normalization
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )



        # Topology and edge encodings
        self.query_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)  # +3 for self, unreachable, task
        self.key_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)
        self.value_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)

        self.query_edge_emb = nn.Embedding(edge_features + 4, hidden_dim)  # +4 for special edges
        self.key_edge_emb = nn.Embedding(edge_features + 4, hidden_dim)
        self.value_edge_emb = nn.Embedding(edge_features + 4, hidden_dim)

        # Transformer layers with DAG-specific attention
        self.layers = nn.ModuleList([
            DAGTransformerLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                max_hop=max_hop
            )
            for _ in range(num_layers)
        ])

        # Structure extractor
        self.structure_extractor = StructureExtractor(
            hidden_dim,
            num_layers=2,
            batch_norm=True
        )

        # Node feature prediction head
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

        # Edge feature prediction components
        self.edge_feat_down = nn.Linear(hidden_dim * 2, hidden_dim)
        self.edge_feat_norm1 = nn.LayerNorm(hidden_dim)
        self.edge_feat_mid = nn.Linear(hidden_dim, hidden_dim // 2)
        self.edge_feat_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.edge_feat_out = nn.Linear(hidden_dim // 2, edge_features)

        # Edge existence prediction
        self.edge_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Final layer norm for better training stability
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        # Extract data attributes
        x, edge_index, batch, mask_mode = self._extract_data_attributes(data)
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        node_mask = data.node_mask if hasattr(data, 'node_mask') else None
        edge_mask = data.edge_mask if hasattr(data, 'edge_mask') else None

        # Node feature embedding
        x = self.node_embedding(x)

        # Compute distance/hop matrix for relative positional encoding
        distance_matrix = self._compute_hop_distances(edge_index, x.size(0), batch)

        # Apply transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Extract structural features
            x_struct = self.structure_extractor(x, edge_index, edge_attr)
            x = x + x_struct  # Residual connection with structural features

            # Apply transformer layer with relative positional encoding
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

        # Apply final normalization
        x = self.final_norm(x)

        # Initialize results dictionary
        results = {}

        # Handle different masking modes
        if mask_mode == "node_feature":
            node_out = self.node_predictor(x)
            results['node_features'] = node_out
            if node_mask is not None:
                results['mask'] = node_mask

        elif mask_mode == "edge_feature":
            self._handle_edge_feature_mode(data, x, edge_mask, results)

            # EXPLICITLY use edge feature components to ensure gradient flow
            if 'edge_preds' in results and results['edge_preds']:
                edge_embeddings = results['edge_preds'].get('edge_features')
                if edge_embeddings is not None:
                    # Force gradient computation by applying dropout or other operation
                    edge_embeddings = self.dropout(edge_embeddings)

        elif mask_mode == "connectivity":
            self._handle_connectivity(data, x, edge_mask, results)

            # EXPLICITLY use edge existence predictor components
            if 'edge_preds' in results and results['edge_preds']:
                edge_existence = results['edge_preds'].get('edge_existence')
                if edge_existence is not None:
                    # Force gradient computation
                    edge_existence = self.dropout(edge_existence)

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

    def _compute_hop_distances(self, edge_index, num_nodes, batch=None):
        """
        Compute shortest path distances between all pairs of nodes in a DAG.
        This version assumes that node IDs are in topological order (i.e. edges only
        go from lower to higher indices). It computes distances only in the forward
        (reachable) direction, then clamps unreachable distances.

        Returns:
            distance_matrix: [num_nodes, num_nodes] tensor of hop distances.
        """
        device = edge_index.device
        # Use max_val as a temporary "infinity" value.
        max_val = self.max_hop + 2
        distance_matrix = torch.full((num_nodes, num_nodes), max_val, dtype=torch.float32, device=device)

        # Set self-distances to 0.
        for i in range(num_nodes):
            distance_matrix[i, i] = 0.0

        # Build a successor list: for each node, list its direct successors.
        successors = [[] for _ in range(num_nodes)]
        num_edges = edge_index.size(1)
        for e in range(num_edges):
            src = edge_index[0, e].item()
            dst = edge_index[1, e].item()
            successors[src].append(dst)

        # For each node, propagate hop distances forward along successors.
        for i in range(num_nodes):
            # Initialize distances for this source node.
            # (We use a Python list for simplicity; num_nodes is assumed to be moderate.)
            dist = [max_val] * num_nodes
            dist[i] = 0  # Distance from i to itself is 0.

            # Since nodes are topologically ordered, we iterate from i up to num_nodes.
            for j in range(i, num_nodes):
                # Only process j if it is reachable from i.
                if dist[j] < max_val:
                    # Update distances for each successor of j.
                    for k in successors[j]:
                        # Because of the ordering, we expect k > j.
                        if k >= i and dist[k] > dist[j] + 1:
                            dist[k] = dist[j] + 1
            # Write computed distances for source i into the distance matrix.
            distance_matrix[i] = torch.tensor(dist, dtype=torch.float32, device=device)

        # Optionally, if you need an undirected (symmetric) distance, you could symmetrize:
        # distance_matrix = torch.min(distance_matrix, distance_matrix.t())

        # Clamp any distance that exceeds max_hop to max_hop+1.
        distance_matrix = torch.clamp(distance_matrix, max=float(self.max_hop + 1))
        return distance_matrix


    def _predict_edge_features(self, src_embeddings, dst_embeddings):
        """Edge feature prediction with GELU activation."""
        # Concatenate embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

        # Apply predictor with skip connections
        edge_feat = self.edge_feat_down(edge_embeddings)
        edge_feat = self.edge_feat_norm1(edge_feat)
        edge_feat = F.gelu(edge_feat)
        edge_feat_mid = self.edge_feat_mid(edge_feat)
        edge_feat_mid = self.edge_feat_norm2(edge_feat_mid)
        edge_feat_mid = F.gelu(edge_feat_mid)

        # Final prediction
        edge_features = self.edge_feat_out(edge_feat_mid)

        return edge_features

    def _handle_edge_feature_mode(self, data, x, edge_mask, results):
        if hasattr(data, 'edge_index_target') and hasattr(data, 'edge_mask') and edge_mask.sum() > 0:
            edge_index_target = data.edge_index_target
            masked_edges = edge_index_target[:, edge_mask]

            valid_edges_mask = (masked_edges[0] < x.size(0)) & (masked_edges[1] < x.size(0))

            if valid_edges_mask.sum() > 0:
                masked_edges = masked_edges[:, valid_edges_mask]

                src_embeddings = x[masked_edges[0]]
                dst_embeddings = x[masked_edges[1]]

                # EXPLICITLY use all edge feature layers
                edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
                edge_feat = self.edge_feat_down(edge_embeddings)
                edge_feat = self.edge_feat_norm1(edge_feat)
                edge_feat = F.gelu(edge_feat)

                edge_feat_mid = self.edge_feat_mid(edge_feat)
                edge_feat_mid = self.edge_feat_norm2(edge_feat_mid)
                edge_feat_mid = F.gelu(edge_feat_mid)

                edge_features = self.edge_feat_out(edge_feat_mid)

                results['edge_preds'] = {
                    'masked_edges': masked_edges,
                    'edge_features': edge_features
                }

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
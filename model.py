import torch
import torch.nn as nn
from layer import EdgeAwareGraphTransformerLayer


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

        # Task-specific head for node feature prediction
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features)
        )

        # Edge feature prediction
        self.edge_feature_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_features)
        )

        # Edge existence prediction (for connectivity mode)
        self.edge_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass for batched PyG Data objects with support for three masking modes:
        1. "node_feature": Mask node features and predict them
        2. "edge_feature": Mask edge features and predict them
        3. "connectivity": Mask edges and predict both existence and features

        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_features]
                - batch: Batch assignment [num_nodes]
                - node_mask: Node feature masking info [num_nodes]
                - edge_mask: Edge masking info [num_edges in target]
                - mask_mode: One of the three masking modes
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

        elif mask_mode == "connectivity":
            # Predict both edge existence and features
            if hasattr(data, 'masked_edge_node_pairs') and hasattr(data, 'all_candidate_pairs'):
                # For positive examples (masked edges that should exist)
                all_candidate_pairs = data.all_candidate_pairs

                # Get node embeddings for source and target nodes of all candidates
                src_embeddings = x[all_candidate_pairs[0]]
                dst_embeddings = x[all_candidate_pairs[1]]

                # Concatenate embeddings
                edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

                # Predict edge existence
                edge_existence = self.edge_existence_predictor(edge_embeddings)

                # Predict edge features (only matters for existing edges, but predict for all)
                edge_features = self.edge_feature_predictor(edge_embeddings)

                # Store edge predictions
                results['edge_preds'] = {
                    'all_candidate_pairs': all_candidate_pairs,
                    'edge_existence': edge_existence,
                    'edge_features': edge_features
                }

            # Fallback for older data format (for backward compatibility)
            elif hasattr(data, 'edge_index_target') and hasattr(data, 'edge_mask') and edge_mask.sum() > 0:
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

        return results
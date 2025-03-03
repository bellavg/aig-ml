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

        # Task-specific head for node prediction
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features)
        )

        # Edge existence prediction (new)
        self.edge_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Edge feature prediction (new)
        self.edge_feature_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_features)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass for batched PyG Data objects with support for gate masking

        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_features]
                - batch: Batch indices [num_nodes]
                - node_mask: Masking information [num_nodes]
                - gate_masking: Boolean flag for gate masking mode
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None
        node_mask = data.node_mask if hasattr(data, 'node_mask') else None
        gate_masking = hasattr(data, 'gate_masking') and data.gate_masking

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

        # Basic result dictionary
        results = {
            'node_features': node_out,
            'mask': node_mask
        }

        # For gate masking, also predict edges
        if gate_masking:
            # Get masked edges
            edge_index_target = data.edge_index_target
            edge_mask = data.edge_mask

            # Create all possible pairs of nodes for edge prediction
            num_nodes = x.size(0)
            edge_preds = {}

            # If we have masked edges, predict their existence and features
            if edge_mask.sum() > 0:
                # Get edges that were masked
                masked_edges = edge_index_target[:, edge_mask]

                # Get node embeddings for source and target nodes
                src_embeddings = x[masked_edges[0]]
                dst_embeddings = x[masked_edges[1]]

                # Concatenate embeddings
                edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

                # Predict edge existence
                edge_existence = self.edge_existence_predictor(edge_embeddings)

                # Predict edge features
                edge_features = self.edge_feature_predictor(edge_embeddings)

                # Store edge predictions
                edge_preds['masked_edges'] = masked_edges
                edge_preds['edge_existence'] = edge_existence
                edge_preds['edge_features'] = edge_features

                results['edge_preds'] = edge_preds

        return results

    def compute_loss(self, outputs, targets):
        """
        Compute loss for masked node and edge prediction task

        Args:
            outputs: Dict with 'node_features', 'mask', and optionally 'edge_preds'
            targets: Dict with node and edge targets, masking info
        """
        # Extract predictions and targets
        pred_nodes = outputs['node_features']
        mask = outputs['mask'] if 'mask' in outputs else targets['node_mask']
        target_nodes = targets['node_features']

        # Initialize losses dictionary
        losses = {}

        # Node feature prediction loss
        node_loss = F.binary_cross_entropy_with_logits(
            pred_nodes[mask],
            target_nodes[mask]
        )
        losses["node_loss"] = node_loss
        total_loss = node_loss

        # Edge prediction losses (for gate masking)
        if 'edge_preds' in outputs and 'edge_mask' in targets and targets['gate_masking']:
            edge_preds = outputs['edge_preds']

            # Edge existence prediction loss
            if 'edge_existence' in edge_preds:
                # For simplicity, we use a binary prediction target (1.0 for all masked edges)
                # In a more advanced implementation, you might want to predict which edges should exist
                edge_existence_targets = torch.ones_like(edge_preds['edge_existence'])

                edge_existence_loss = F.binary_cross_entropy(
                    edge_preds['edge_existence'],
                    edge_existence_targets
                )
                losses["edge_existence_loss"] = edge_existence_loss
                total_loss += edge_existence_loss

            # Edge feature prediction loss
            if 'edge_features' in edge_preds and 'edge_attr' in targets:
                # Get edge features for masked edges
                target_edge_attr = targets['edge_attr']
                edge_mask = targets['edge_mask']
                masked_edge_attr = target_edge_attr[edge_mask]

                edge_feature_loss = F.mse_loss(
                    edge_preds['edge_features'],
                    masked_edge_attr
                )
                losses["edge_feature_loss"] = edge_feature_loss
                total_loss += edge_feature_loss

        # Always return total loss and individual losses
        losses["total_loss"] = total_loss
        return total_loss, losses
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max
import math
import torch
from typing import Dict, List, Optional, Tuple, Union

from layers import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import your custom modules
from loss import TruthTableFeatureLoss, FeatureMetrics


class AIGTransformerLightning(pl.LightningModule):
    """
    PyTorch Lightning implementation of the AIG Transformer model.
    """

    def __init__(
            self,
            node_type_dim: int = 3,  # Dimension of node type one-hot encoding
            feature_dim: int = 256,  # Target feature dimension (truth table size)
            edge_type_dim: int = 2,  # Dimension of edge type one-hot encoding
            hidden_dim: int = 128,  # Hidden dimension
            num_layers: int = 4,  # Number of transformer layers
            num_heads: int = 4,  # Number of attention heads
            dropout: float = 0.1,  # Dropout rate
            max_hop: int = 5,  # Maximum hop distance to consider
            gnn_type: str = "gcn",  # Type of GNN to use for structure extraction
            max_tt_length: int = 256,  # Maximum truth table length
            prediction_tasks: List[str] = ["node_feature"],  # List of prediction tasks
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            l1_weight: float = 0.1,
            feature_normalization: bool = False,
            binary_loss_weight: float = 2.0,
            scheduler_factor: float = 0.5,
            scheduler_patience: int = 5,
            clip_grad_norm: float = 1.0,
            batch_size: int = 32,
    ):
        """
        Initialize the AIG Transformer Lightning model.

        Args:
            node_type_dim: Dimension of node type one-hot encoding
            feature_dim: Target feature dimension (truth table size)
            edge_type_dim: Dimension of edge type one-hot encoding
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_hop: Maximum hop distance to consider
            gnn_type: Type of GNN to use for structure extraction
            max_tt_length: Maximum truth table length
            prediction_tasks: List of prediction tasks
            learning_rate: Initial learning rate
            weight_decay: Weight decay for L2 regularization
            l1_weight: Weight for L1 loss component
            feature_normalization: Whether to normalize features
            binary_loss_weight: Weight for binary classification loss
            scheduler_factor: Factor for learning rate scheduler
            scheduler_patience: Patience for learning rate scheduler
            clip_grad_norm: Max norm for gradient clipping
        """
        super(AIGTransformerLightning, self).__init__()
        self.save_hyperparameters()

        # Import AIG Transformer model
        from model import AIGTransformer

        # Create the PyTorch model
        self.model = AIGTransformer(
            node_type_dim=node_type_dim,
            feature_dim=feature_dim,
            edge_type_dim=edge_type_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_hop=max_hop,
            gnn_type=gnn_type,
            max_tt_length=max_tt_length,
            prediction_tasks=prediction_tasks
        )

        # Initialize loss function
        self.criterion = TruthTableFeatureLoss(
            l1_weight=l1_weight,
            feature_normalization=feature_normalization,
            ignore_padding=True,
            binary_loss_weight=binary_loss_weight
        )

        # Store hyperparameters needed for optimizer and scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.clip_grad_norm = clip_grad_norm
        self.node_type_dim = node_type_dim
        self.batch_size = batch_size

    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch: PyG Data object containing the graph

        Returns:
            Model output dictionary
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: PyG Data object containing the graph
            batch_idx: Index of the batch

        Returns:
            Loss value
        """
        # Forward pass
        outputs = self(batch)

        # Extract predictions and ground truth
        pred_features = outputs['node_features']
        node_mask = batch.mask

        # Get features of masked nodes (only the truth table part)
        true_features = batch.y[node_mask, self.node_type_dim:]

        # Compute loss
        loss = self.criterion(pred_features, true_features)

        # Compute metrics with padding handling
        metrics = FeatureMetrics.compute_metrics(pred_features, true_features, ignore_padding=True)

        # Calculate truth table accuracy
        tt_metrics = FeatureMetrics.compute_truth_table_accuracy(pred_features, true_features)

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size,sync_dist=True)
        self.log('train_mse', metrics['mse'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('train_mae', metrics['mae'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('train_r2', metrics['r2'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('train_tt_accuracy', tt_metrics['truth_table_accuracy'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: PyG Data object containing the graph
            batch_idx: Index of the batch

        Returns:
            Dictionary with validation metrics
        """
        # Forward pass
        outputs = self(batch)

        # Extract predictions and ground truth
        pred_features = outputs['node_features']
        node_mask = batch.mask

        # Get features of masked nodes (only the truth table part)
        true_features = batch.y[node_mask, self.node_type_dim:]

        # Compute loss
        loss = self.criterion(pred_features, true_features)

        # Compute metrics with padding handling
        metrics = FeatureMetrics.compute_metrics(pred_features, true_features, ignore_padding=True)

        # Calculate truth table accuracy
        tt_metrics = FeatureMetrics.compute_truth_table_accuracy(pred_features, true_features)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size,sync_dist=True)
        self.log('val_mse', metrics['mse'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('val_mae', metrics['mae'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('val_r2', metrics['r2'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('val_tt_accuracy', tt_metrics['truth_table_accuracy'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)

        return {'val_loss': loss, 'metrics': metrics, 'tt_metrics': tt_metrics}

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: PyG Data object containing the graph
            batch_idx: Index of the batch

        Returns:
            Dictionary with test metrics
        """
        # Forward pass
        outputs = self(batch)

        # Extract predictions and ground truth
        pred_features = outputs['node_features']
        node_mask = batch.mask

        # Get features of masked nodes (only the truth table part)
        true_features = batch.y[node_mask, self.node_type_dim:]

        # Compute metrics with padding handling
        metrics = FeatureMetrics.compute_metrics(pred_features, true_features, ignore_padding=True)

        # Calculate truth table accuracy
        tt_metrics = FeatureMetrics.compute_truth_table_accuracy(pred_features, true_features)

        # Compute loss
        loss = self.criterion(pred_features, true_features)

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=self.batch_size,sync_dist=True)
        self.log('test_mse', metrics['mse'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('test_mae', metrics['mae'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('test_r2', metrics['r2'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log('test_tt_accuracy', tt_metrics['truth_table_accuracy'], on_step=False, on_epoch=True,batch_size=self.batch_size,sync_dist=True)

        # Store predictions and targets for later analysis
        return {
            'test_loss': loss,
            'metrics': metrics,
            'tt_metrics': tt_metrics,
            'pred_features': pred_features.detach().cpu(),
            'true_features': true_features.detach().cpu(),
            'node_mask': node_mask.detach().cpu()
        }

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Configured optimizer and scheduler
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Define scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            ),
            'monitor': 'val_loss',  # Metric to monitor
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        """
        Clip gradients before optimizer step.

        Args:
            optimizer: The optimizer
        """
        if self.clip_grad_norm > 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)

    def test_epoch_end(self, outputs):
        """
        Process and log aggregated test results at the end of the test epoch.

        Args:
            outputs: List of outputs from test_step

        Returns:
            Dictionary with aggregated metrics
        """
        # Aggregate metrics across all batches
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_mse = np.mean([x['metrics']['mse'] for x in outputs])
        test_mae = np.mean([x['metrics']['mae'] for x in outputs])
        test_r2 = np.mean([x['metrics']['r2'] for x in outputs])

        # Aggregate truth table metrics
        correct_bits = sum(x['tt_metrics']['correct_bits'] for x in outputs)
        total_bits = sum(x['tt_metrics']['total_bits'] for x in outputs)
        tt_accuracy = correct_bits / total_bits if total_bits > 0 else 0.0

        # Log final metrics
        self.log('test_loss_final', test_loss)
        self.log('test_mse_final', test_mse)
        self.log('test_mae_final', test_mae)
        self.log('test_r2_final', test_r2)
        self.log('test_tt_accuracy_final', tt_accuracy)

        return {
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_tt_accuracy': tt_accuracy,
            'correct_bits': correct_bits,
            'total_bits': total_bits
        }


class AIGTransformer(nn.Module):
    """
    Structure-aware transformer for AND-Inverter Graphs with multi-task capability.
    Can predict node features, edge features, and links.

    This model:
    1. Uses DAG-specific attention that respects the graph structure
    2. Combines both hop-based and edge-type-based positional encoding
    3. Supports different prediction tasks through task-specific heads
    4. Includes truth table positional encoding to maintain feature order
    5. Handles padding masks for truth table values
    """

    def __init__(
            self,
            node_type_dim=3,  # Dimension of node type one-hot encoding
            feature_dim=256,  # Target feature dimension (truth table size)
            edge_type_dim=2,  # Dimension of edge type one-hot encoding
            hidden_dim=128,  # Hidden dimension
            num_layers=4,  # Number of transformer layers
            num_heads=4,  # Number of attention heads
            dropout=0.1,  # Dropout rate
            max_hop=5,  # Maximum hop distance to consider
            gnn_type="gcn",  # Type of GNN to use for structure extraction
            max_tt_length=256,  # Maximum truth table length
            prediction_tasks=["node_feature"]  # List of prediction tasks
    ):
        super(AIGTransformer, self).__init__()

        self.prediction_tasks = prediction_tasks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_hop = max_hop
        self.feature_dim = feature_dim
        self.edge_type_dim = edge_type_dim
        self.node_type_dim = node_type_dim
        self.max_tt_length = max_tt_length

        # Constants for special hop/edge types
        self.TASK_DISTANCE = max_hop + 1
        self.UNREACHABLE_DISTANCE = max_hop + 2
        self.TASK_EDGE = edge_type_dim + 1
        self.SELF_EDGE = edge_type_dim + 2
        self.NO_EDGE = edge_type_dim + 3

        # Truth table positional encoding
        self.register_buffer(
            "tt_pos_encoding",
            self._create_tt_positional_encoding(max_tt_length, hidden_dim // 4)
        )

        # Node feature processing
        self.node_type_embedding = nn.Linear(node_type_dim, hidden_dim // 2)
        self.tt_feature_embedding = nn.Linear(feature_dim, hidden_dim // 2)

        # Combine node type, tt features, and positional encoding
        self.node_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Edge feature embedding
        self.edge_embedding = nn.Linear(edge_type_dim, hidden_dim)

        # Positional encoding (hop-based and edge-type-based)
        self.query_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)
        self.key_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)
        self.value_hop_emb = nn.Embedding(max_hop + 3, hidden_dim)

        self.query_edge_emb = nn.Embedding(edge_type_dim + 4, hidden_dim)
        self.key_edge_emb = nn.Embedding(edge_type_dim + 4, hidden_dim)
        self.value_edge_emb = nn.Embedding(edge_type_dim + 4, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            DAGTransformerLayer(
                hidden_dim,
                num_heads,
                hidden_dim * 4,
                dropout,
                max_hop=max_hop,
                gnn_type=gnn_type
            ) for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Task-specific prediction heads
        self.prediction_heads = nn.ModuleDict()

        # Node feature prediction head
        if "node_feature" in prediction_tasks:
            self.prediction_heads["node_feature"] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim)
            )

        # Edge feature prediction head
        if "edge_feature" in prediction_tasks:
            self.prediction_heads["edge_feature"] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, edge_type_dim)
            )

        # Link prediction head
        if "link" in prediction_tasks:
            self.prediction_heads["link"] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

    def _create_tt_positional_encoding(self, max_length: int, dim: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding for truth table values.

        Args:
            max_length: Maximum length of truth table
            dim: Dimension of positional encoding

        Returns:
            Positional encoding tensor of shape [max_length, dim]
        """
        position = torch.arange(max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pos_encoding = torch.zeros(max_length, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, node_type_dim + feature_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge attributes [num_edges, edge_type_dim]
                - node_mask: Boolean mask for nodes to predict features for [num_nodes]
                - edge_mask: Boolean mask for edges to predict features for [num_edges]
                - candidate_edges: Edges to consider for link prediction [2, num_candidates]
                - batch: Batch assignment for nodes [num_nodes]
                - mask_mode: Which prediction task to perform

        Returns:
            Dictionary of prediction results
        """
        # Extract data attributes
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        node_mask = data.mask if hasattr(data, 'mask') else None
        edge_mask = data.edge_mask if hasattr(data, 'edge_mask') else None
        candidate_edges = data.candidate_edges if hasattr(data, 'candidate_edges') else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        mask_mode = data.mask_mode if hasattr(data, 'mask_mode') else "node_feature"

        # Split node features into node type and truth table features
        node_type = x[:, :self.node_type_dim]
        tt_features = x[:, self.node_type_dim:]

        # Create padding mask for truth table values (value == -1)
        padding_mask = (tt_features == -1)

        # Process node type
        node_type_emb = self.node_type_embedding(node_type)

        # Process truth table features with positional encoding
        # First handle padding by zeroing out -1 values
        tt_features_masked = tt_features.clone()
        tt_features_masked[padding_mask] = 0.0

        # Embed truth table features
        tt_emb = self.tt_feature_embedding(tt_features_masked)

        # Combine with positional encoding for valid (non-padded) positions
        batch_size = x.size(0)
        pos_encoding = self.tt_pos_encoding[:self.max_tt_length, :].unsqueeze(0).expand(batch_size, -1, -1)

        # Combine embeddings
        h = torch.cat([node_type_emb, tt_emb], dim=-1)
        h = self.node_embedding(h)

        # Prepare edge features
        if edge_attr is not None:
            edge_features = self.edge_embedding(edge_attr)
        else:
            edge_features = None

        # Compute distance/hop matrix for structural bias
        distance_matrix = self._compute_hop_distances(edge_index, x.size(0), batch)

        # Compute edge_attr matrix for edge type bias
        edge_type_matrix = self._compute_edge_type_matrix(edge_index, edge_attr, x.size(0), batch)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(
                h,
                distance_matrix,
                edge_type_matrix,
                edge_index,
                edge_features,
                self.query_hop_emb,  # Pass embedding module, not weight
                self.query_edge_emb,
                self.key_hop_emb,
                self.key_edge_emb,
                self.value_hop_emb,
                self.value_edge_emb,
                batch
            )

        # Apply final normalization
        h = self.final_norm(h)

        # Return predictions based on mask_mode
        results = {}

        if mask_mode == "node_feature" and "node_feature" in self.prediction_heads:
            node_features = self.prediction_heads["node_feature"](h)

            # Apply node mask if provided
            if node_mask is not None:
                masked_indices = torch.nonzero(node_mask).squeeze(-1)
                if masked_indices.numel() > 0:
                    masked_features = node_features[masked_indices]
                    results["node_features"] = masked_features
                    results["mask"] = node_mask
            else:
                results["node_features"] = node_features

        elif mask_mode == "edge_feature" and "edge_feature" in self.prediction_heads:
            # Handle edge feature prediction
            if edge_mask is not None and edge_mask.sum() > 0:
                # Get masked edges
                masked_edge_indices = torch.nonzero(edge_mask).squeeze(-1)
                masked_edges = edge_index[:, masked_edge_indices]

                # Get node embeddings for source and target nodes
                src_embeddings = h[masked_edges[0]]
                tgt_embeddings = h[masked_edges[1]]

                # Concatenate embeddings
                edge_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=1)

                # Predict edge features
                edge_features = self.prediction_heads["edge_feature"](edge_embeddings)

                results["edge_features"] = edge_features
                results["masked_edge_indices"] = masked_edge_indices

        elif mask_mode == "link" and "link" in self.prediction_heads:
            # Handle link prediction
            if candidate_edges is not None:
                # Get node embeddings for source and target nodes
                src_embeddings = h[candidate_edges[0]]
                tgt_embeddings = h[candidate_edges[1]]

                # Concatenate embeddings
                pair_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=1)

                # Predict link existence
                link_scores = self.prediction_heads["link"](pair_embeddings)

                results["link_scores"] = link_scores
                results["candidate_edges"] = candidate_edges

        return results

    def _compute_hop_distances(self, edge_index, num_nodes, batch):
        """
        Compute shortest path distances between all pairs of nodes in a DAG.

        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes in the graph
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            distance_matrix: [num_nodes, num_nodes] tensor of hop distances.
        """
        device = edge_index.device
        # Use max_val as a "infinity" value
        max_val = self.max_hop + 2
        distance_matrix = torch.full((num_nodes, num_nodes), max_val, dtype=torch.long, device=device)

        # Set self-distances to 0
        for i in range(num_nodes):
            distance_matrix[i, i] = 0

        # Build successor lists
        successors = [[] for _ in range(num_nodes)]
        num_edges = edge_index.size(1)
        for e in range(num_edges):
            src = edge_index[0, e].item()
            dst = edge_index[1, e].item()
            successors[src].append(dst)

        # For each node, propagate hop distances forward through the DAG
        for i in range(num_nodes):
            # Only process nodes within the same batch
            batch_i = batch[i].item()
            same_batch_mask = (batch == batch_i)

            # Initialize distances for this source node
            dist = [max_val] * num_nodes
            dist[i] = 0

            # Find nodes reachable from i (only forward direction)
            for j in range(num_nodes):
                if same_batch_mask[j].item() and dist[j] < max_val:
                    # Update distances for each successor of j
                    for k in successors[j]:
                        if same_batch_mask[k].item() and dist[k] > dist[j] + 1:
                            dist[k] = dist[j] + 1

            # Write computed distances into the matrix
            distance_matrix[i] = torch.tensor(dist, dtype=torch.long, device=device)

        # Clamp distances to max_hop+1 (unreachable)
        distance_matrix = torch.clamp(distance_matrix, max=self.UNREACHABLE_DISTANCE)
        return distance_matrix

    def _compute_edge_type_matrix(self, edge_index, edge_attr, num_nodes, batch):
        """
        Compute a matrix of edge types between all pairs of nodes.

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            num_nodes: Number of nodes in the graph
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            edge_type_matrix: [num_nodes, num_nodes] tensor of edge types.
        """
        device = edge_index.device

        # Initialize with NO_EDGE for all pairs
        edge_type_matrix = torch.full((num_nodes, num_nodes), self.NO_EDGE, dtype=torch.long, device=device)

        # Set diagonal to self-edge type
        for i in range(num_nodes):
            edge_type_matrix[i, i] = self.SELF_EDGE

        # Fill in actual edge types from edge_index and edge_attr
        if edge_attr is not None:
            num_edges = edge_index.size(1)
            for e in range(num_edges):
                src = edge_index[0, e].item()
                dst = edge_index[1, e].item()

                # For binary edge attributes (e.g., INV/REG), use argmax to get type
                if edge_attr.dim() > 1 and edge_attr.size(1) > 1:
                    edge_type = torch.argmax(edge_attr[e]).item()
                else:
                    edge_type = edge_attr[e].item()

                edge_type_matrix[src, dst] = edge_type

        return edge_type_matrix

    def compute_loss(self, pred, target, padding_mask=None):
        """
        Compute loss for the model predictions, handling padding values.

        Args:
            pred: Predicted features [batch_size, feature_dim]
            target: Target features [batch_size, feature_dim]
            padding_mask: Boolean mask for padding values [batch_size, feature_dim]

        Returns:
            Loss value
        """
        if padding_mask is None:
            # If no padding mask provided, create one based on target values of -1
            padding_mask = (target == -1)

        # Apply mask to predictions and targets
        valid_mask = ~padding_mask

        # Calculate MSE loss only on valid (non-padding) positions
        if valid_mask.sum() > 0:
            # Only compute loss on non-padded values
            mse_loss = F.mse_loss(
                pred[valid_mask],
                target[valid_mask]
            )
            return mse_loss
        else:
            # If all values are padded, return zero loss
            return torch.tensor(0.0, device=pred.device)
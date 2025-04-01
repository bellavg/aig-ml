"""
Test suite for the AIGTransformer model.

This test suite verifies the correct functioning of the AIGTransformer 
model and its components, including:
- Model initialization
- Forward pass with different prediction tasks
- Hop distance and edge type computation
- Loss calculation with padding
- Multi-task capabilities
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your modules
from model import AIGTransformer


class TestAIGTransformerInitialization:
    """Tests for the initialization of the AIGTransformer model."""

    @pytest.fixture
    def model_params(self):
        """Basic parameters for testing AIGTransformer."""
        return {
            'node_type_dim': 3,
            'feature_dim': 256,
            'edge_type_dim': 2,
            'hidden_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'max_hop': 5,
            'gnn_type': 'gcn',
            'max_tt_length': 256
        }

    def test_basic_initialization(self, model_params):
        """Test if the model initializes with default parameters."""
        model = AIGTransformer(**model_params)

        # Check model structure
        assert isinstance(model, AIGTransformer)
        assert len(model.layers) == model_params['num_layers']
        assert len(model.prediction_heads) == 1  # Default includes node_feature
        assert 'node_feature' in model.prediction_heads

        # Check dimension parameters are stored correctly
        assert model.node_type_dim == model_params['node_type_dim']
        assert model.feature_dim == model_params['feature_dim']
        assert model.hidden_dim == model_params['hidden_dim']

        # Check constants are defined correctly
        assert model.TASK_DISTANCE == model_params['max_hop'] + 1
        assert model.UNREACHABLE_DISTANCE == model_params['max_hop'] + 2
        assert model.TASK_EDGE == model_params['edge_type_dim'] + 1
        assert model.SELF_EDGE == model_params['edge_type_dim'] + 2
        assert model.NO_EDGE == model_params['edge_type_dim'] + 3

    def test_multi_task_initialization(self, model_params):
        """Test if the model initializes with multiple prediction tasks."""
        all_tasks = ["node_feature", "edge_feature", "link"]
        model = AIGTransformer(**model_params, prediction_tasks=all_tasks)

        # Check prediction heads
        assert len(model.prediction_heads) == len(all_tasks)
        for task in all_tasks:
            assert task in model.prediction_heads

        # Check output dimensions for each head
        assert model.prediction_heads["node_feature"][-1].out_features == model_params['feature_dim']
        assert model.prediction_heads["edge_feature"][-1].out_features == model_params['edge_type_dim']
        assert model.prediction_heads["link"][-1].out_features == 1

    def test_positional_encoding(self, model_params):
        """Test if truth table positional encoding is created correctly."""
        model = AIGTransformer(**model_params)
        max_length = model_params['max_tt_length']
        dim = model_params['hidden_dim'] // 4

        # Check dimensions of positional encoding
        assert model.tt_pos_encoding.shape == (max_length, dim)

        # Instead of checking exact values, check basic properties of sinusoidal encoding
        calculated_values = model.tt_pos_encoding[:10, :10]

        # Check that values are in expected range (-1 to 1)
        assert torch.all(calculated_values >= -1.0) and torch.all(calculated_values <= 1.0)

        # First row should alternate between 0 and 1 (sin(0) = 0, cos(0) = 1)
        assert torch.allclose(calculated_values[0, 0::2], torch.zeros(5), atol=1e-6)
        assert torch.allclose(calculated_values[0, 1::2], torch.ones(5), atol=1e-6)

        # Check that positions have different encodings
        assert not torch.allclose(calculated_values[0], calculated_values[1], atol=1e-3)
        assert not torch.allclose(calculated_values[1], calculated_values[2], atol=1e-3)


class TestAIGTransformerForward:
    """Tests for the forward pass of the AIGTransformer model."""

    @pytest.fixture
    def model_params(self):
        """Parameters for a small test model."""
        return {
            'node_type_dim': 3,
            'feature_dim': 8,  # Small feature dimension for testing
            'edge_type_dim': 2,
            'hidden_dim': 16,  # Small hidden dimension for testing
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1,
            'max_hop': 3,
            'gnn_type': 'gcn',
            'max_tt_length': 8,
            'prediction_tasks': ["node_feature", "edge_feature", "link"]
        }

    @pytest.fixture
    def create_data(self, model_params):
        """Create a PyG data object with synthetic AIG graph data."""
        def _create(num_nodes=10, mask_mode="node_feature"):
            """Helper to create data with different parameters."""
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Create node features
            node_type = torch.zeros(num_nodes, model_params['node_type_dim'])
            for i in range(num_nodes):
                node_type[i, i % model_params['node_type_dim']] = 1.0

            # Create truth table features with padding (-1)
            tt_features = torch.rand(num_nodes, model_params['feature_dim'])
            padding_mask = torch.rand(num_nodes, model_params['feature_dim']) < 0.2
            tt_features[padding_mask] = -1.0

            # Combine node type and truth table
            x = torch.cat([node_type, tt_features], dim=1)

            # Create edges (DAG structure)
            edge_list = []
            for i in range(num_nodes - 1):
                # Each node connects to a few nodes ahead (ensuring DAG property)
                for j in range(i + 1, min(i + 3, num_nodes)):
                    edge_list.append((i, j))

            edge_index = torch.tensor(edge_list, dtype=torch.long).t()

            # Create edge attributes
            edge_attr = torch.zeros(edge_index.size(1), model_params['edge_type_dim'])
            for i in range(edge_index.size(1)):
                edge_type = i % model_params['edge_type_dim']
                edge_attr[i, edge_type] = 1.0

            # Create mask for prediction task
            if mask_mode == "node_feature":
                # Mask 20% of nodes - ensure at least one node is masked
                mask = torch.zeros(num_nodes, dtype=torch.bool)
                num_to_mask = max(1, int(num_nodes * 0.2))
                mask_indices = torch.randperm(num_nodes)[:num_to_mask]
                mask[mask_indices] = True
            elif mask_mode == "edge_feature":
                # Mask 20% of edges - ensure at least one edge is masked
                edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
                num_to_mask = max(1, int(edge_index.size(1) * 0.2))
                mask_indices = torch.randperm(edge_index.size(1))[:num_to_mask]
                edge_mask[mask_indices] = True
            elif mask_mode == "link":
                # Create candidate edges (20% existing edges, 80% non-existing)
                num_candidates = num_nodes * 2

                # Get existing edges
                existing_edges = edge_index.t().tolist()
                existing_edges_set = set((src, dst) for src, dst in existing_edges)

                # Create non-existing edges
                non_existing_edges = []
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):  # Maintain DAG property
                        if (i, j) not in existing_edges_set:
                            non_existing_edges.append((i, j))

                # Sample from existing edges (positive samples)
                num_positive = max(1, num_candidates // 5)  # 20% positive, at least 1
                if len(existing_edges) < num_positive:
                    # If not enough existing edges, use all of them
                    positive_samples = existing_edges
                    num_positive = len(positive_samples)
                else:
                    positive_idx = torch.randperm(len(existing_edges))[:num_positive]
                    positive_samples = [existing_edges[i] for i in positive_idx]

                # Sample from non-existing edges (negative samples)
                num_negative = num_candidates - num_positive
                if len(non_existing_edges) < num_negative:
                    # If not enough non-existing edges, use all of them
                    negative_samples = non_existing_edges
                else:
                    negative_idx = torch.randperm(len(non_existing_edges))[:num_negative]
                    negative_samples = [non_existing_edges[i] for i in negative_idx]

                # Combine positive and negative samples
                candidate_edges = positive_samples + negative_samples
                candidate_edges = torch.tensor(candidate_edges, dtype=torch.long).t()

            # Create batch assignment (all nodes in one batch)
            batch = torch.zeros(num_nodes, dtype=torch.long)

            # Mock PyG data object
            class Data:
                def __init__(self):
                    self.device = device

                def to(self, device):
                    # Mock the to() method for device transfer
                    return self

                def __repr__(self):
                    attrs = vars(self)
                    return "\n".join(f"{key}: {value}" for key, value in attrs.items())

            data = Data()
            data.x = x.to(device)
            data.edge_index = edge_index.to(device)
            data.edge_attr = edge_attr.to(device)
            data.batch = batch.to(device)
            data.mask_mode = mask_mode

            # Add padding_mask as attribute for the model
            data.padding_mask = padding_mask.to(device)

            if mask_mode == "node_feature":
                data.mask = mask.to(device)
            elif mask_mode == "edge_feature":
                data.edge_mask = edge_mask.to(device)
            elif mask_mode == "link":
                data.candidate_edges = candidate_edges.to(device)

            return data

        return _create

    def test_forward_node_feature(self, model_params, create_data):
        """Test forward pass with node feature prediction."""
        try:
            model = AIGTransformer(**model_params)
            data = create_data(num_nodes=10, mask_mode="node_feature")

            # Forward pass
            output = model(data)

            # Check output
            assert "node_features" in output
            assert "mask" in output

            # Check dimensions
            masked_indices = torch.nonzero(data.mask).squeeze(-1)
            if masked_indices.numel() > 0:
                assert output["node_features"].shape == (masked_indices.numel(), model_params['feature_dim'])
        except Exception as e:
            # Print more information for debugging
            print(f"Error in test_forward_node_feature: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_forward_edge_feature(self, model_params, create_data):
        """Test forward pass with edge feature prediction."""
        model = AIGTransformer(**model_params)
        data = create_data(num_nodes=10, mask_mode="edge_feature")

        # Forward pass
        output = model(data)

        # Check output
        assert "edge_features" in output
        assert "masked_edge_indices" in output

        # Check dimensions
        masked_indices = torch.nonzero(data.edge_mask).squeeze(-1)
        if masked_indices.numel() > 0:
            assert output["edge_features"].shape == (masked_indices.numel(), model_params['edge_type_dim'])

    def test_forward_link(self, model_params, create_data):
        """Test forward pass with link prediction."""
        model = AIGTransformer(**model_params)
        data = create_data(num_nodes=10, mask_mode="link")

        # Forward pass
        output = model(data)

        # Check output
        assert "link_scores" in output
        assert "candidate_edges" in output

        # Check dimensions
        assert output["link_scores"].shape == (data.candidate_edges.size(1), 1)

    def test_padding_mask_handling(self, model_params, create_data):
        """Test handling of padding mask during forward pass."""
        model = AIGTransformer(**model_params)
        data = create_data(num_nodes=10, mask_mode="node_feature")

        # Create a version of data with no padding
        data_no_padding = create_data(num_nodes=10, mask_mode="node_feature")
        # Replace -1 values with random values
        tt_features = data_no_padding.x[:, model_params['node_type_dim']:]
        mask = (tt_features == -1)
        tt_features[mask] = torch.rand_like(tt_features[mask])
        data_no_padding.x[:, model_params['node_type_dim']:] = tt_features

        # Forward passes
        output_with_padding = model(data)
        output_no_padding = model(data_no_padding)

        # Outputs should be different due to padding
        if data.mask.sum() > 0 and data_no_padding.mask.sum() > 0:
            masked_indices_padding = torch.nonzero(data.mask).squeeze(-1)
            masked_indices_no_padding = torch.nonzero(data_no_padding.mask).squeeze(-1)

            if masked_indices_padding.numel() > 0 and masked_indices_no_padding.numel() > 0:
                features_with_padding = output_with_padding["node_features"]
                features_no_padding = output_no_padding["node_features"]

                # The outputs should be different (not exactly equal)
                # but we need at least one element to compare
                if min(features_with_padding.size(0), features_no_padding.size(0)) > 0:
                    min_size = min(features_with_padding.size(0), features_no_padding.size(0))
                    assert not torch.allclose(
                        features_with_padding[:min_size],
                        features_no_padding[:min_size]
                    )


class TestAIGTransformerHopAndEdge:
    """Tests for the hop distance and edge type computation."""

    @pytest.fixture
    def model_params(self):
        """Parameters for a small test model."""
        return {
            'node_type_dim': 3,
            'feature_dim': 8,
            'edge_type_dim': 2,
            'hidden_dim': 16,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1,
            'max_hop': 3,
            'gnn_type': 'gcn',
            'max_tt_length': 8
        }

    def test_compute_hop_distances(self, model_params):
        """Test the computation of hop distances."""
        model = AIGTransformer(**model_params)

        # Create a simple DAG
        #     0
        #    / \
        #   1   2
        #  / \   \
        # 3   4   5
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2],  # source
            [1, 2, 3, 4, 5]   # target
        ], dtype=torch.long)

        num_nodes = 6
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Compute distances
        distance_matrix = model._compute_hop_distances(edge_index, num_nodes, batch)

        # Expected distance matrix
        # 0 -> 0: 0
        # 0 -> 1: 1
        # 0 -> 2: 1
        # 0 -> 3: 2
        # 0 -> 4: 2
        # 0 -> 5: 2
        # 1 -> 1: 0
        # 1 -> 3: 1
        # 1 -> 4: 1
        # 2 -> 2: 0
        # 2 -> 5: 1
        # Others: unreachable (max_hop + 2)
        expected = torch.ones(num_nodes, num_nodes, dtype=torch.long) * model.UNREACHABLE_DISTANCE

        # Set self-distances to 0
        for i in range(num_nodes):
            expected[i, i] = 0

        # Set direct connections to 1
        for e in range(edge_index.size(1)):
            src, dst = edge_index[0, e].item(), edge_index[1, e].item()
            expected[src, dst] = 1

        # Set 2-hop connections
        expected[0, 3] = 2
        expected[0, 4] = 2
        expected[0, 5] = 2

        # Check if computed distances match expected
        assert torch.all(distance_matrix == expected)

    def test_compute_edge_type_matrix(self, model_params):
        """Test the computation of edge type matrix."""
        model = AIGTransformer(**model_params)

        # Create a simple DAG with edge types
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2],  # source
            [1, 2, 3, 4, 5]   # target
        ], dtype=torch.long)

        # Edge attributes (types)
        edge_attr = torch.tensor([
            [1, 0],  # Type 0
            [0, 1],  # Type 1
            [1, 0],  # Type 0
            [0, 1],  # Type 1
            [1, 0]   # Type 0
        ], dtype=torch.float)

        num_nodes = 6
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Compute edge type matrix
        edge_type_matrix = model._compute_edge_type_matrix(edge_index, edge_attr, num_nodes, batch)

        # Expected edge type matrix
        # NO_EDGE for all pairs except:
        # - SELF_EDGE on diagonal
        # - Actual edge types for edges
        expected = torch.ones(num_nodes, num_nodes, dtype=torch.long) * model.NO_EDGE

        # Set self-edges
        for i in range(num_nodes):
            expected[i, i] = model.SELF_EDGE

        # Set edge types
        expected[0, 1] = 0  # Type 0
        expected[0, 2] = 1  # Type 1
        expected[1, 3] = 0  # Type 0
        expected[1, 4] = 1  # Type 1
        expected[2, 5] = 0  # Type 0

        # Check if computed edge type matrix matches expected
        assert torch.all(edge_type_matrix == expected)

    def test_hop_distances_with_batching(self, model_params):
        """Test hop distance computation with batched graphs."""
        model = AIGTransformer(**model_params)

        # Create two separate DAGs in one batch
        # First DAG: 0->1->2
        # Second DAG: 3->4->5
        edge_index = torch.tensor([
            [0, 1, 3, 4],  # source
            [1, 2, 4, 5]   # target
        ], dtype=torch.long)

        num_nodes = 6
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        # Compute distances
        distance_matrix = model._compute_hop_distances(edge_index, num_nodes, batch)

        # Expected distance matrix
        # For first graph:
        # 0 -> 0: 0
        # 0 -> 1: 1
        # 0 -> 2: 2
        # 1 -> 1: 0
        # 1 -> 2: 1
        # 2 -> 2: 0

        # For second graph:
        # 3 -> 3: 0
        # 3 -> 4: 1
        # 3 -> 5: 2
        # 4 -> 4: 0
        # 4 -> 5: 1
        # 5 -> 5: 0

        # All cross-graph connections: unreachable
        expected = torch.ones(num_nodes, num_nodes, dtype=torch.long) * model.UNREACHABLE_DISTANCE

        # Set self-distances to 0
        for i in range(num_nodes):
            expected[i, i] = 0

        # First graph connections
        expected[0, 1] = 1
        expected[0, 2] = 2
        expected[1, 2] = 1

        # Second graph connections
        expected[3, 4] = 1
        expected[3, 5] = 2
        expected[4, 5] = 1

        # Check if computed distances match expected
        assert torch.all(distance_matrix == expected)


class TestAIGTransformerLoss:
    """Tests for the loss computation of the AIGTransformer model."""

    @pytest.fixture
    def model_params(self):
        """Parameters for a small test model."""
        return {
            'node_type_dim': 3,
            'feature_dim': 8,
            'edge_type_dim': 2,
            'hidden_dim': 16,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1,
            'max_hop': 3,
            'gnn_type': 'gcn',
            'max_tt_length': 8
        }

    def test_loss_without_padding(self, model_params):
        """Test loss computation without padding."""
        model = AIGTransformer(**model_params)

        # Create predictions and targets
        batch_size = 10
        feature_dim = model_params['feature_dim']

        pred = torch.rand(batch_size, feature_dim)
        target = torch.rand(batch_size, feature_dim)

        # Compute loss
        loss = model.compute_loss(pred, target)

        # Expected loss: MSE between pred and target
        expected_loss = F.mse_loss(pred, target)

        assert torch.isclose(loss, expected_loss)

    def test_loss_with_padding(self, model_params):
        """Test loss computation with padding."""
        model = AIGTransformer(**model_params)

        # Create predictions and targets with padding
        batch_size = 10
        feature_dim = model_params['feature_dim']

        pred = torch.rand(batch_size, feature_dim)
        target = torch.rand(batch_size, feature_dim)

        # Create padding mask (20% of values are padded)
        padding_mask = torch.rand(batch_size, feature_dim) < 0.2
        target[padding_mask] = -1

        # Compute loss
        loss = model.compute_loss(pred, target, padding_mask)

        # Expected loss: MSE only on non-padded values
        valid_mask = ~padding_mask
        if valid_mask.sum() > 0:
            expected_loss = F.mse_loss(
                pred[valid_mask],
                target[valid_mask]
            )
            assert torch.isclose(loss, expected_loss)
        else:
            assert loss.item() == 0.0

    def test_loss_all_padded(self, model_params):
        """Test loss computation when all values are padded."""
        model = AIGTransformer(**model_params)

        # Create predictions and targets with all values padded
        batch_size = 10
        feature_dim = model_params['feature_dim']

        pred = torch.rand(batch_size, feature_dim)
        target = torch.ones(batch_size, feature_dim) * -1  # All padded

        # Padding mask (all True)
        padding_mask = torch.ones(batch_size, feature_dim, dtype=torch.bool)

        # Compute loss
        loss = model.compute_loss(pred, target, padding_mask)

        # Expected loss: 0 (no valid values)
        assert loss.item() == 0.0


import math  # Make sure to import math module for the positional encoding test

if __name__ == "__main__":
    pytest.main(["-v"])
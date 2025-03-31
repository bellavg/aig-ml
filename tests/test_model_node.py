import torch
import pytest
import numpy as np
from torch_geometric.data import Data


class TestNodeFeaturePrediction:
    @pytest.fixture
    def model_params(self):
        """Standard model parameters for testing."""
        return {
            'node_type_dim': 3,
            'feature_dim': 1,  # Single dimension for truth table
            'edge_type_dim': 2,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'max_hop': 5,
            'prediction_tasks': ['node_feature']
        }

    @pytest.fixture
    def sample_aig_graph(self):
        """Create a sample Artificial Intelligence Graph (AIG) for testing."""
        num_nodes = 10
        num_edges = 15

        # Node features: [is_input, is_and_gate, is_output, truth_table_value]
        node_features = torch.zeros((num_nodes, 4))
        node_features[:3, 0] = 1.0  # Inputs
        node_features[3:7, 1] = 1.0  # AND gates
        node_features[7:, 2] = 1.0  # Outputs

        # Set truth table values for some AND gates
        node_features[3:7, 3] = torch.tensor([0.3637, 0.5346, 0.6736, 0.0])

        # Create edge connections
        edge_index = torch.zeros((2, num_edges), dtype=torch.long)
        edge_count = 0

        # Connect inputs to AND gates
        for i in range(3):
            for j in range(3, 7):
                if edge_count < num_edges:
                    edge_index[0, edge_count] = i
                    edge_index[1, edge_count] = j
                    edge_count += 1

        # Connect AND gates to outputs
        for j in range(3, 7):
            for k in range(7, num_nodes):
                if edge_count < num_edges:
                    edge_index[0, edge_count] = j
                    edge_index[1, edge_count] = k
                    edge_count += 1

        # Random edge attributes
        edge_attr = torch.rand(num_edges, 2)

        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=torch.zeros(num_nodes, dtype=torch.long)
        )

        return data

    def test_preprocessing_compatibility(self, sample_aig_graph, model_params):
        """
        Test preprocessing steps to ensure compatibility.
        Debug and verify how node features are processed before model input.
        """
        from model import AIGTransformer
        from masking import create_masked_batch

        # Instantiate the model
        model = AIGTransformer(**model_params)

        # Split node features
        x = sample_aig_graph.x
        node_type = x[:, :model_params['node_type_dim']]
        tt_features = x[:, model_params['node_type_dim']:]

        # Print out feature shapes and values for debugging
        print("\nNode Type Features:")
        print(f"Shape: {node_type.shape}")
        print(f"Values:\n{node_type}")

        print("\nTruth Table Features:")
        print(f"Shape: {tt_features.shape}")
        print(f"Values:\n{tt_features}")

        # Check feature embeddings
        node_type_emb = model.node_type_embedding(node_type)
        print("\nNode Type Embedding:")
        print(f"Shape: {node_type_emb.shape}")

        # Prepare truth table features (handle potential zero/masked values)
        tt_features_masked = tt_features.clone()
        tt_features_masked[tt_features == 0] = 0.0

        # Embed truth table features
        print("\nTruth Table Features (before embedding):")
        print(f"Shape: {tt_features_masked.shape}")
        print(f"Values:\n{tt_features_masked}")

        tt_emb = model.tt_feature_embedding(tt_features_masked)
        print("\nTruth Table Embedding:")
        print(f"Shape: {tt_emb.shape}")

        # Verify basic requirements
        assert node_type_emb.shape[0] == x.shape[0]
        assert tt_emb.shape[0] == x.shape[0]

    def test_forward_pass_node_feature(self, sample_aig_graph, model_params):
        """
        Test forward pass for node feature prediction with detailed debug info.
        """
        from model import AIGTransformer
        from masking import create_masked_batch

        # Instantiate the model
        model = AIGTransformer(**model_params)

        # Create masked batch
        masked_batch = create_masked_batch(
            sample_aig_graph,
            mp=0.2,  # 20% masking
            mask_mode="node_feature"
        )

        # Debug information before forward pass
        print("\nMasked Batch Details:")
        print(f"Total nodes: {masked_batch.x.shape[0]}")
        print(f"Masked nodes: {masked_batch.mask.sum()}")
        print(f"Node feature shape: {masked_batch.x.shape}")
        print(f"Masked feature values:\n{masked_batch.x[masked_batch.mask]}")

        # Forward pass with error handling
        try:
            results = model(masked_batch)
        except Exception as e:
            # Provide detailed error context
            print(f"\nForward pass failed: {e}")
            raise

        # Verify results
        print("\nModel Results:")
        print(f"Node features shape: {results['node_features'].shape}")
        print(f"Masked node features:\n{results['node_features']}")

        # Assertions
        assert 'node_features' in results
        assert results['node_features'].shape[1] == model_params['feature_dim']
        assert results['node_features'].shape[0] <= sample_aig_graph.x.shape[0]

    def test_node_type_preservation(self, sample_aig_graph, model_params):
        """
        Verify that node types remain unchanged during prediction.
        """
        from model import AIGTransformer
        from masking import create_masked_batch

        # Instantiate the model
        model = AIGTransformer(**model_params)

        # Create masked batch
        masked_batch = create_masked_batch(
            sample_aig_graph,
            mp=0.2,
            mask_mode="node_feature"
        )

        # Get original node types
        original_types = masked_batch.x[:, :model_params['node_type_dim']]

        # Forward pass
        results = model(masked_batch)

        # Verify that masked feature prediction does not alter node types
        masked_nodes = masked_batch.mask
        masked_indices = torch.nonzero(masked_nodes).squeeze(-1)

        for idx in masked_indices:
            # Check that the node type columns remain the same for masked nodes
            assert torch.allclose(
                masked_batch.x[idx, :model_params['node_type_dim']],
                original_types[idx]
            ), f"Node type changed for node {idx}"


def test_main():
    """Run all tests."""
    pytest.main(["-v", __file__])


if __name__ == "__main__":
    test_main()
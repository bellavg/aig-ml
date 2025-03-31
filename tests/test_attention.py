import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the layer class from your module (adjust as needed)
from model import DAGTransformerLayer


class TestDAGTransformerLayer:
    """Tests for the DAGTransformerLayer with truth table padding support."""

    @pytest.fixture
    def transformer_params(self):
        """Basic parameters for testing DAGTransformerLayer."""
        return {
            'd_model': 64,
            'nhead': 4,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'max_hop': 5,
            'num_nodes': 20,
            'num_edge_types': 3,
            'truth_table_dim': 16  # Dimension of truth table features
        }

    @pytest.fixture
    def dag_data(self, transformer_params):
        """Create a DAG with truth tables for testing the transformer layer."""
        d_model = transformer_params['d_model']
        num_nodes = transformer_params['num_nodes']
        max_hop = transformer_params['max_hop']
        num_edge_types = transformer_params['num_edge_types']
        truth_table_dim = transformer_params['truth_table_dim']

        # Create node features with truth tables
        # Format: [node_type_features, truth_table_features]
        node_type_dim = 3  # Assuming 3-dim one-hot encoding for node types
        x = torch.randn(num_nodes, d_model - truth_table_dim)

        # Create truth tables with padding (-1)
        truth_tables = torch.rand(num_nodes, truth_table_dim)

        # Add padding to some entries (setting to -1)
        padding_mask = torch.rand(num_nodes, truth_table_dim) < 0.2
        truth_tables[padding_mask] = -1.0

        # Concatenate node features and truth tables
        x = torch.cat([x, truth_tables], dim=1)

        # Create a simple DAG structure
        edge_list = []
        for i in range(num_nodes - 1):
            # Each node connects to a few nodes ahead (ensuring DAG property)
            for j in range(i + 1, min(i + 4, num_nodes)):
                edge_list.append((i, j))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        # Create edge attributes (one-hot encoding of edge types)
        edge_attr = torch.zeros(edge_index.size(1), num_edge_types)
        for i in range(edge_index.size(1)):
            edge_type = i % num_edge_types
            edge_attr[i, edge_type] = 1

        # Create distance matrix
        distance_matrix = torch.ones(num_nodes, num_nodes) * (max_hop + 2)
        distance_matrix.fill_diagonal_(0)

        # Set direct edges to distance 1
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            distance_matrix[src, dst] = 1

        # Create edge type matrix (initialized with high value for non-edges)
        edge_type_matrix = torch.ones(num_nodes, num_nodes, dtype=torch.long) * num_edge_types
        edge_type_matrix.fill_diagonal_(num_edge_types + 1)  # Special type for self-connections

        # Set edge types
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_type_matrix[src, dst] = i % num_edge_types

        # Compute all-pairs shortest paths
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                        distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

        # Create embeddings
        query_hop_emb = nn.Embedding(max_hop + 3, d_model)  # +3 for self, unreachable, task
        key_hop_emb = nn.Embedding(max_hop + 3, d_model)
        value_hop_emb = nn.Embedding(max_hop + 3, d_model)
        query_edge_emb = nn.Embedding(num_edge_types + 4, d_model)  # +4 for special edges
        key_edge_emb = nn.Embedding(num_edge_types + 4, d_model)
        value_edge_emb = nn.Embedding(num_edge_types + 4, d_model)

        # Create padding mask for truth tables (True where we have -1 values)
        padding_mask = (truth_tables == -1)

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'distance_matrix': distance_matrix,
            'edge_type_matrix': edge_type_matrix,
            'query_hop_emb': query_hop_emb,
            'key_hop_emb': key_hop_emb,
            'value_hop_emb': value_hop_emb,
            'query_edge_emb': query_edge_emb,
            'key_edge_emb': key_edge_emb,
            'value_edge_emb': value_edge_emb,
            'padding_mask': padding_mask,
            'truth_table_dim': truth_table_dim
        }

    def test_initialization(self, transformer_params):
        """Test if the transformer layer initializes correctly."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Check dimensions of key components
        d_model = transformer_params['d_model']

        # Check self-attention components
        assert hasattr(transformer.self_attn, 'q_proj')
        assert hasattr(transformer.self_attn, 'k_proj')
        assert hasattr(transformer.self_attn, 'v_proj')
        assert hasattr(transformer.self_attn, 'out_proj')

        assert transformer.self_attn.q_proj.weight.shape == (d_model, d_model)
        assert transformer.self_attn.k_proj.weight.shape == (d_model, d_model)
        assert transformer.self_attn.v_proj.weight.shape == (d_model, d_model)
        assert transformer.self_attn.out_proj.weight.shape == (d_model, d_model)

        # Check feedforward network
        assert transformer.feed_forward[0].weight.shape[0] == transformer_params['dim_feedforward']
        assert transformer.feed_forward[3].weight.shape[0] == d_model

        # Check structure extractor
        assert hasattr(transformer, 'structure_extractor')

    def test_forward_pass(self, transformer_params, dag_data):
        """Test forward pass of the transformer layer with truth table padding."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Forward pass with padding mask
        output = transformer(
            dag_data['x'],
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            batch=None,
            padding_mask=dag_data['padding_mask']
        )

        # Check output
        assert output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(output).any()

    def test_forward_pass_without_padding(self, transformer_params, dag_data):
        """Test forward pass without passing padding mask."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Forward pass without padding mask
        output = transformer(
            dag_data['x'],
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb']
        )

        # Check output
        assert output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(output).any()

    def test_padding_mask_impact(self, transformer_params, dag_data):
        """Test that padding mask properly influences the attention weights."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Get results with padding mask
        with_mask = transformer(
            dag_data['x'],
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            padding_mask=dag_data['padding_mask']
        )

        # Create an inverted padding mask (padding in different positions)
        inverted_mask = ~dag_data['padding_mask']

        # Get results with inverted padding mask
        with_inverted_mask = transformer(
            dag_data['x'],
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            padding_mask=inverted_mask
        )

        # Outputs should be different due to different padding masks
        assert not torch.allclose(with_mask, with_inverted_mask)

    def test_dag_attention_mechanism(self, transformer_params, dag_data):
        """Test that the DAG attention mechanism properly respects node distances."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Get original output
        original_output = transformer(
            dag_data['x'],
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            padding_mask=dag_data['padding_mask']
        )

        # Modify distance matrix to make all nodes reachable directly
        modified_distance = torch.ones_like(dag_data['distance_matrix'])

        # Get output with modified distances
        modified_output = transformer(
            dag_data['x'],
            modified_distance,
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            padding_mask=dag_data['padding_mask']
        )

        # Outputs should be different due to different distance matrices
        assert not torch.allclose(original_output, modified_output)

    def test_forward_with_batch(self, transformer_params, dag_data):
        """Test forward pass with batch information."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Create batch assignment
        num_nodes = transformer_params['num_nodes']
        batch = torch.zeros(num_nodes, dtype=torch.long)
        batch[num_nodes // 2:] = 1  # Second half is second graph

        # Forward pass with batch
        output = transformer(
            dag_data['x'],
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            batch=batch,
            padding_mask=dag_data['padding_mask']
        )

        # Check output
        assert output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(output).any()

    def test_structure_extractor(self, transformer_params, dag_data):
        """Test the structure extractor component."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Run just the structure extractor
        struct_output = transformer.structure_extractor(
            dag_data['x'],
            dag_data['edge_index'],
            edge_attr=dag_data['edge_attr']
        )

        # Check output
        assert struct_output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(struct_output).any()

    def test_dag_multi_head_attention(self, transformer_params, dag_data):
        """Test the DAG multi-head attention component directly."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Apply normalization to input
        x_norm = transformer.norm1(dag_data['x'])

        # Test the multi-head attention directly
        attn_output = transformer.self_attn(
            x_norm, x_norm, x_norm,
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            None,  # batch
            dag_data['padding_mask']
        )

        # Check output
        assert attn_output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(attn_output).any()

    def test_gradient_flow(self, transformer_params, dag_data):
        """Test gradient flow through the transformer layer."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Enable gradient tracking
        x = dag_data['x'].clone().requires_grad_(True)

        # Forward pass
        output = transformer(
            x,
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            padding_mask=dag_data['padding_mask']
        )

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check if gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check gradients flow through embeddings
        assert dag_data['query_hop_emb'].weight.grad is not None
        assert dag_data['key_hop_emb'].weight.grad is not None
        assert dag_data['value_hop_emb'].weight.grad is not None
        assert dag_data['query_edge_emb'].weight.grad is not None
        assert dag_data['key_edge_emb'].weight.grad is not None
        assert dag_data['value_edge_emb'].weight.grad is not None

    def test_different_truth_table_sizes(self, transformer_params):
        """Test that the transformer can handle different truth table sizes."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Create inputs with different truth table dimensions
        d_model = transformer_params['d_model']
        num_nodes = 10

        # Test with smaller truth tables
        small_tt_dim = 8
        x_small = torch.randn(num_nodes, d_model - small_tt_dim)
        tt_small = torch.rand(num_nodes, small_tt_dim)
        padding_mask_small = torch.rand(num_nodes, small_tt_dim) < 0.2
        tt_small[padding_mask_small] = -1.0
        x_small_full = torch.cat([x_small, tt_small], dim=1)

        # Test with larger truth tables
        large_tt_dim = 24
        x_large = torch.randn(num_nodes, d_model - large_tt_dim)
        tt_large = torch.rand(num_nodes, large_tt_dim)
        padding_mask_large = torch.rand(num_nodes, large_tt_dim) < 0.2
        tt_large[padding_mask_large] = -1.0
        x_large_full = torch.cat([x_large, tt_large], dim=1)

        # Create minimal edge and distance data
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.zeros(3, transformer_params['num_edge_types'])
        distance_matrix = torch.ones(num_nodes, num_nodes) * (transformer_params['max_hop'] + 2)
        distance_matrix.fill_diagonal_(0)
        edge_type_matrix = torch.ones(num_nodes, num_nodes, dtype=torch.long) * transformer_params['num_edge_types']

        # Create embeddings
        query_hop_emb = nn.Embedding(transformer_params['max_hop'] + 3, d_model)
        key_hop_emb = nn.Embedding(transformer_params['max_hop'] + 3, d_model)
        value_hop_emb = nn.Embedding(transformer_params['max_hop'] + 3, d_model)
        query_edge_emb = nn.Embedding(transformer_params['num_edge_types'] + 4, d_model)
        key_edge_emb = nn.Embedding(transformer_params['num_edge_types'] + 4, d_model)
        value_edge_emb = nn.Embedding(transformer_params['num_edge_types'] + 4, d_model)

        # Forward pass with small truth tables
        output_small = transformer(
            x_small_full,
            distance_matrix,
            edge_type_matrix,
            edge_index,
            edge_attr,
            query_hop_emb,
            query_edge_emb,
            key_hop_emb,
            key_edge_emb,
            value_hop_emb,
            value_edge_emb,
            padding_mask=padding_mask_small
        )

        # Forward pass with large truth tables
        output_large = transformer(
            x_large_full,
            distance_matrix,
            edge_type_matrix,
            edge_index,
            edge_attr,
            query_hop_emb,
            query_edge_emb,
            key_hop_emb,
            key_edge_emb,
            value_hop_emb,
            value_edge_emb,
            padding_mask=padding_mask_large
        )

        # Check outputs
        assert output_small.shape == (num_nodes, d_model)
        assert output_large.shape == (num_nodes, d_model)
        assert not torch.isnan(output_small).any()
        assert not torch.isnan(output_large).any()

    def test_all_padding(self, transformer_params, dag_data):
        """Test the case where all truth table values are padded."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Create a padding mask where all truth table values are -1
        all_padding_mask = torch.ones_like(dag_data['padding_mask']).bool()

        # Create input with all truth table values set to -1
        x_with_padding = dag_data['x'].clone()
        truth_table_dim = dag_data['truth_table_dim']
        x_with_padding[:, -truth_table_dim:] = -1.0

        # Forward pass with all padded values
        output = transformer(
            x_with_padding,
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            padding_mask=all_padding_mask
        )

        # Check output
        assert output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(output).any()

    def test_no_padding(self, transformer_params, dag_data):
        """Test the case where no truth table values are padded."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Create a padding mask where no truth table values are -1
        no_padding_mask = torch.zeros_like(dag_data['padding_mask']).bool()

        # Create input with no truth table values set to -1
        x_no_padding = dag_data['x'].clone()
        truth_table_dim = dag_data['truth_table_dim']
        x_no_padding[:, -truth_table_dim:] = torch.rand(dag_data['x'].size(0), truth_table_dim)

        # Forward pass with no padded values
        output = transformer(
            x_no_padding,
            dag_data['distance_matrix'],
            dag_data['edge_type_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_hop_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_hop_emb'],
            dag_data['value_edge_emb'],
            padding_mask=no_padding_mask
        )

        # Check output
        assert output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(output).any()
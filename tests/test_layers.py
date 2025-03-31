"""
Test suite for the DAGTransformerLayer with truth table padding support.

This test suite verifies the correct functioning of the DAGTransformerLayer
and its components, including the attention mechanism with support for
truth table padding and DAG structure awareness.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your module
from model import DAGTransformerLayer, DAGMultiHeadAttention, StructureExtractor, GCNLayer


class TestDAGTransformerComponents:
    """Tests for the individual components of the DAGTransformerLayer."""

    @pytest.fixture
    def layer_params(self):
        """Basic parameters for testing GCNLayer."""
        return {
            'in_dim': 64,
            'out_dim': 64,
            'num_nodes': 10
        }

    @pytest.fixture
    def graph_data(self, layer_params):
        """Create a small graph for testing GCN layers."""
        num_nodes = layer_params['num_nodes']
        in_dim = layer_params['in_dim']

        # Create node features
        x = torch.randn(num_nodes, in_dim)

        # Create a simple graph where each node connects to the next
        edge_index = torch.zeros((2, num_nodes-1), dtype=torch.long)
        for i in range(num_nodes-1):
            edge_index[0, i] = i      # source
            edge_index[1, i] = i + 1  # target

        # Create edge features
        edge_features = torch.randn(edge_index.size(1), in_dim)

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_features
        }

    def test_gcn_layer(self, layer_params, graph_data):
        """Test if the GCNLayer works correctly."""
        # Create the layer
        layer = GCNLayer(layer_params['in_dim'], layer_params['out_dim'])

        # Forward pass
        output = layer(graph_data['x'], graph_data['edge_index'], graph_data['edge_attr'])

        # Check output dimensions
        assert output.shape == (layer_params['num_nodes'], layer_params['out_dim'])
        assert not torch.isnan(output).any()

        # Test with no edges
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        output_empty = layer(graph_data['x'], empty_edge_index)
        assert torch.all(output_empty == 0)

    def test_structure_extractor(self, layer_params, graph_data):
        """Test if the StructureExtractor works correctly."""
        # Create the structure extractor
        extractor = StructureExtractor(
            hidden_dim=layer_params['in_dim'],
            num_layers=2,
            batch_norm=True,
            gnn_type="gcn",
            edge_dim=graph_data['edge_attr'].size(1)  # Add this line
        )

        # Forward pass
        output = extractor(
            graph_data['x'],
            graph_data['edge_index'],
            graph_data['edge_attr']
        )

        # Check output
        assert output.shape == (layer_params['num_nodes'], layer_params['in_dim'])
        assert not torch.isnan(output).any()

        # Check gradient flow
        x = graph_data['x'].clone().requires_grad_(True)
        edge_attr = graph_data['edge_attr'].clone().requires_grad_(True)
        output = extractor(x, graph_data['edge_index'], edge_attr)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert edge_attr.grad is not None
        assert extractor.out_proj.weight.grad is not None


class TestDAGMultiHeadAttention:
    """Tests for the DAGMultiHeadAttention component."""

    @pytest.fixture
    def attention_params(self):
        """Basic parameters for testing DAGMultiHeadAttention."""
        return {
            'hidden_dim': 64,
            'num_heads': 4,
            'dropout': 0.1,
            'max_hop': 5,
            'num_nodes': 15,
            'num_hop_types': 8,
            'num_edge_types': 7,
            'truth_table_dim': 16  # Dimension of truth table features
        }

    @pytest.fixture
    def attention_data(self, attention_params):
        """Create data for testing attention mechanism."""
        hidden_dim = attention_params['hidden_dim']
        num_nodes = attention_params['num_nodes']
        max_hop = attention_params['max_hop']
        num_hop_types = attention_params['num_hop_types']
        num_edge_types = attention_params['num_edge_types']
        truth_table_dim = attention_params['truth_table_dim']

        # Create input tensors
        query = torch.randn(num_nodes, hidden_dim)
        key = query.clone()  # Same as query for self-attention
        value = query.clone()  # Same as query for self-attention

        # Create embeddings
        query_hop_emb = nn.Embedding(num_hop_types, hidden_dim)
        key_hop_emb = nn.Embedding(num_hop_types, hidden_dim)
        value_hop_emb = nn.Embedding(num_hop_types, hidden_dim)
        query_edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        key_edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        value_edge_emb = nn.Embedding(num_edge_types, hidden_dim)

        # Create distance matrix and edge type matrix
        distance_matrix = torch.randint(0, num_hop_types - 1, (num_nodes, num_nodes)).float()
        for i in range(num_nodes):
            distance_matrix[i, i] = 0  # Self-distance is 0

        edge_type_matrix = torch.randint(0, num_edge_types - 1, (num_nodes, num_nodes))

        # Create batch assignment (single batch for simplicity)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Create padding mask for truth tables (randomly masked values)
        padding_mask = torch.rand(num_nodes, truth_table_dim) < 0.2

        return {
            'query': query,
            'key': key,
            'value': value,
            'query_hop_emb': query_hop_emb,
            'key_hop_emb': key_hop_emb,
            'value_hop_emb': value_hop_emb,
            'query_edge_emb': query_edge_emb,
            'key_edge_emb': key_edge_emb,
            'value_edge_emb': value_edge_emb,
            'distance_matrix': distance_matrix,
            'edge_type_matrix': edge_type_matrix,
            'batch': batch,
            'padding_mask': padding_mask
        }

    def test_attention_initialization(self, attention_params):
        """Test if the attention layer initializes correctly."""
        attention = DAGMultiHeadAttention(
            attention_params['hidden_dim'],
            attention_params['num_heads'],
            attention_params['dropout'],
            attention_params['max_hop']
        )

        # Check dimensions
        hidden_dim = attention_params['hidden_dim']
        assert attention.q_proj.weight.shape == (hidden_dim, hidden_dim)
        assert attention.k_proj.weight.shape == (hidden_dim, hidden_dim)
        assert attention.v_proj.weight.shape == (hidden_dim, hidden_dim)
        assert attention.out_proj.weight.shape == (hidden_dim, hidden_dim)

    def test_attention_forward(self, attention_params, attention_data):
        """Test forward pass of the attention layer."""
        attention = DAGMultiHeadAttention(
            attention_params['hidden_dim'],
            attention_params['num_heads'],
            attention_params['dropout'],
            attention_params['max_hop']
        )

        # Forward pass
        output = attention(
            attention_data['query'],
            attention_data['key'],
            attention_data['value'],
            attention_data['query_hop_emb'],
            attention_data['query_edge_emb'],
            attention_data['key_hop_emb'],
            attention_data['key_edge_emb'],
            attention_data['value_hop_emb'],
            attention_data['value_edge_emb'],
            attention_data['distance_matrix'],
            attention_data['edge_type_matrix'],
            attention_data['batch'],
            attention_data['padding_mask']
        )

        # Check output
        assert output.shape == (attention_params['num_nodes'], attention_params['hidden_dim'])
        assert not torch.isnan(output).any()

    def test_attention_with_padding_mask(self, attention_params, attention_data):
        """Test if padding mask properly influences attention weights."""
        attention = DAGMultiHeadAttention(
            attention_params['hidden_dim'],
            attention_params['num_heads'],
            attention_params['dropout'],
            attention_params['max_hop']
        )

        # Forward pass with padding mask
        output_with_mask = attention(
            attention_data['query'],
            attention_data['key'],
            attention_data['value'],
            attention_data['query_hop_emb'],
            attention_data['query_edge_emb'],
            attention_data['key_hop_emb'],
            attention_data['key_edge_emb'],
            attention_data['value_hop_emb'],
            attention_data['value_edge_emb'],
            attention_data['distance_matrix'],
            attention_data['edge_type_matrix'],
            attention_data['batch'],
            attention_data['padding_mask']
        )

        # Forward pass without padding mask
        output_without_mask = attention(
            attention_data['query'],
            attention_data['key'],
            attention_data['value'],
            attention_data['query_hop_emb'],
            attention_data['query_edge_emb'],
            attention_data['key_hop_emb'],
            attention_data['key_edge_emb'],
            attention_data['value_hop_emb'],
            attention_data['value_edge_emb'],
            attention_data['distance_matrix'],
            attention_data['edge_type_matrix'],
            attention_data['batch'],
            None
        )

        # Outputs should be different due to padding mask
        assert not torch.allclose(output_with_mask, output_without_mask)

    def test_attention_with_max_hop(self, attention_params, attention_data):
        """Test if max_hop properly masks out distant nodes."""
        # Create attention with small max_hop
        attention_small_hop = DAGMultiHeadAttention(
            attention_params['hidden_dim'],
            attention_params['num_heads'],
            attention_params['dropout'],
            max_hop=1  # Very restrictive hop limit
        )

        # Create attention with large max_hop
        attention_large_hop = DAGMultiHeadAttention(
            attention_params['hidden_dim'],
            attention_params['num_heads'],
            attention_params['dropout'],
            max_hop=10  # Very permissive hop limit
        )

        # Forward pass with small max_hop
        output_small_hop = attention_small_hop(
            attention_data['query'],
            attention_data['key'],
            attention_data['value'],
            attention_data['query_hop_emb'],
            attention_data['query_edge_emb'],
            attention_data['key_hop_emb'],
            attention_data['key_edge_emb'],
            attention_data['value_hop_emb'],
            attention_data['value_edge_emb'],
            attention_data['distance_matrix'],
            attention_data['edge_type_matrix'],
            attention_data['batch'],
            attention_data['padding_mask']
        )

        # Forward pass with large max_hop
        output_large_hop = attention_large_hop(
            attention_data['query'],
            attention_data['key'],
            attention_data['value'],
            attention_data['query_hop_emb'],
            attention_data['query_edge_emb'],
            attention_data['key_hop_emb'],
            attention_data['key_edge_emb'],
            attention_data['value_hop_emb'],
            attention_data['value_edge_emb'],
            attention_data['distance_matrix'],
            attention_data['edge_type_matrix'],
            attention_data['batch'],
            attention_data['padding_mask']
        )

        # Outputs should be different due to different max_hop values
        assert not torch.allclose(output_small_hop, output_large_hop)

    def test_attention_gradient_flow(self, attention_params, attention_data):
        """Test if gradients flow correctly through the attention layer."""
        attention = DAGMultiHeadAttention(
            attention_params['hidden_dim'],
            attention_params['num_heads'],
            attention_params['dropout'],
            attention_params['max_hop']
        )

        # Enable gradient tracking
        query = attention_data['query'].clone().requires_grad_(True)

        # Forward pass
        output = attention(
            query,
            attention_data['key'],
            attention_data['value'],
            attention_data['query_hop_emb'],
            attention_data['query_edge_emb'],
            attention_data['key_hop_emb'],
            attention_data['key_edge_emb'],
            attention_data['value_hop_emb'],
            attention_data['value_edge_emb'],
            attention_data['distance_matrix'],
            attention_data['edge_type_matrix'],
            attention_data['batch'],
            attention_data['padding_mask']
        )

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check if gradients are computed
        assert query.grad is not None
        assert attention.q_proj.weight.grad is not None
        assert attention.k_proj.weight.grad is not None
        assert attention.v_proj.weight.grad is not None
        assert attention.out_proj.weight.grad is not None
        assert attention_data['query_hop_emb'].weight.grad is not None
        assert attention_data['key_hop_emb'].weight.grad is not None
        assert attention_data['value_hop_emb'].weight.grad is not None


class TestDAGTransformerLayer:
    """Tests for the complete DAGTransformerLayer."""

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
        node_type_dim = d_model - truth_table_dim
        x_features = torch.randn(num_nodes, node_type_dim)

        # Create truth tables with padding (-1)
        truth_tables = torch.rand(num_nodes, truth_table_dim)

        # Add padding to some entries (setting to -1)
        padding_mask = torch.rand(num_nodes, truth_table_dim) < 0.2
        truth_tables[padding_mask] = -1.0

        # Concatenate node features and truth tables
        x = torch.cat([x_features, truth_tables], dim=1)

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

    def test_transformer_forward_pass(self, transformer_params, dag_data):
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

    def test_transformer_with_batch(self, transformer_params, dag_data):
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

    def test_transformer_without_padding(self, transformer_params, dag_data):
        """Test forward pass without padding mask."""
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

    def test_transformer_structure_extractor(self, transformer_params, dag_data):
        """Test the structure extractor component of the transformer."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Access and test the structure extractor
        structure_output = transformer.structure_extractor(
            dag_data['x'],
            dag_data['edge_index'],
            dag_data['edge_attr']
        )

        # Check output
        assert structure_output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(structure_output).any()

    def test_transformer_gradient_flow(self, transformer_params, dag_data):
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
        assert transformer.self_attn.q_proj.weight.grad is not None
        assert transformer.self_attn.k_proj.weight.grad is not None
        assert transformer.self_attn.v_proj.weight.grad is not None
        assert transformer.self_attn.out_proj.weight.grad is not None
        assert dag_data['query_hop_emb'].weight.grad is not None


if __name__ == "__main__":
    pytest.main(["-v"])
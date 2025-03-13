"""
Test suite for neural network layer components in DAG model.

These tests verify the functionality of:
- GraphConvLayer
- StructureExtractor
- DAGTransformerLayer

Each component is tested individually and in combination to ensure proper integration.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_add

# Import your modules
# Assuming your code is in a module/package - adjust the path as needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the layer classes from your module
# You'll need to adjust these imports to match your actual module structure
from model import GraphConvLayer, StructureExtractor, DAGTransformerLayer


class TestGraphConvLayer:
    """Tests for the GraphConvLayer component."""

    @pytest.fixture
    def layer_params(self):
        """Basic parameters for testing GraphConvLayer."""
        return {
            'in_dim': 16,
            'out_dim': 32,
            'num_nodes': 10
        }

    @pytest.fixture
    def graph_data(self, layer_params):
        """Create a small graph for testing."""
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
            'edge_features': edge_features
        }

    def test_initialization(self, layer_params):
        """Test if the layer initializes correctly."""
        # Create the layer
        layer = GraphConvLayer(layer_params['in_dim'], layer_params['out_dim'])

        # Check dimensions
        assert layer.W.weight.shape == (layer_params['out_dim'], layer_params['in_dim'])
        assert layer.W_edge.weight.shape == (layer_params['out_dim'], layer_params['in_dim'])
        assert layer.W.bias.shape == (layer_params['out_dim'],)
        assert layer.W_edge.bias.shape == (layer_params['out_dim'],)

    def test_forward_without_edge_features(self, layer_params, graph_data):
        """Test forward pass without edge features."""
        # Create the layer
        layer = GraphConvLayer(layer_params['in_dim'], layer_params['out_dim'])

        # Forward pass
        output = layer(graph_data['x'], graph_data['edge_index'])

        # Check output dimensions
        assert output.shape == (layer_params['num_nodes'], layer_params['out_dim'])
        assert not torch.isnan(output).any()

    def test_forward_with_edge_features(self, layer_params, graph_data):
        """Test forward pass with edge features."""
        # Create the layer
        layer = GraphConvLayer(layer_params['in_dim'], layer_params['out_dim'])

        # Forward pass with edge features
        output = layer(
            graph_data['x'],
            graph_data['edge_index'],
            graph_data['edge_features']
        )

        # Check output dimensions
        assert output.shape == (layer_params['num_nodes'], layer_params['out_dim'])
        assert not torch.isnan(output).any()

    def test_empty_edge_index(self, layer_params, graph_data):
        """Test behavior with empty edge index."""
        # Create the layer
        layer = GraphConvLayer(layer_params['in_dim'], layer_params['out_dim'])

        # Create empty edge index
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Forward pass with empty edge index
        output = layer(graph_data['x'], empty_edge_index)

        # Output should be zeros
        assert output.shape == (layer_params['num_nodes'], layer_params['out_dim'])
        assert (output == 0).all()

    def test_gradient_flow(self, layer_params, graph_data):
        """Test if gradients flow correctly through the layer."""
        # Create the layer
        layer = GraphConvLayer(layer_params['in_dim'], layer_params['out_dim'])

        # Enable gradient tracking
        x = graph_data['x'].clone().requires_grad_(True)
        edge_features = graph_data['edge_features'].clone().requires_grad_(True)

        # Forward pass
        output = layer(x, graph_data['edge_index'], edge_features)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check if gradients are computed
        assert x.grad is not None
        assert edge_features.grad is not None
        assert layer.W.weight.grad is not None
        assert layer.W_edge.weight.grad is not None


class TestStructureExtractor:
    """Tests for the StructureExtractor component."""

    @pytest.fixture
    def extractor_params(self):
        """Basic parameters for testing StructureExtractor."""
        return {
            'embed_dim': 32,
            'num_layers': 2,
            'edge_dim': 2,
            'num_nodes': 15
        }

    @pytest.fixture
    def graph_data(self, extractor_params):
        """Create a graph for testing the structure extractor."""
        num_nodes = extractor_params['num_nodes']
        embed_dim = extractor_params['embed_dim']
        edge_dim = extractor_params['edge_dim']

        # Create node features
        x = torch.randn(num_nodes, embed_dim)

        # Create a simple graph structure
        edge_list = []
        for i in range(num_nodes-1):
            # Each node connects to the next few nodes
            for j in range(i+1, min(i+3, num_nodes)):
                edge_list.append((i, j))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        # Create edge features
        edge_attr = torch.randn(edge_index.size(1), edge_dim)

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }

    def test_initialization(self, extractor_params):
        """Test if the extractor initializes correctly."""
        # Test with default parameters
        extractor = StructureExtractor(
            embed_dim=extractor_params['embed_dim'],
            edge_dim=extractor_params['edge_dim']
        )

        # Check number of layers
        assert len(extractor.conv_layers) == 2

        # Test with custom parameters
        custom_layers = 4
        extractor = StructureExtractor(
            embed_dim=extractor_params['embed_dim'],
            num_layers=custom_layers,
            edge_dim=extractor_params['edge_dim']
        )

        # Check number of layers
        assert len(extractor.conv_layers) == custom_layers

        # Test with BatchNorm
        extractor_bn = StructureExtractor(
            embed_dim=extractor_params['embed_dim'],
            batch_norm=True,
            edge_dim=extractor_params['edge_dim']
        )

        # Check if BatchNorm is used
        assert isinstance(extractor_bn.norm_layers[0], nn.BatchNorm1d)

        # Test with LayerNorm
        extractor_ln = StructureExtractor(
            embed_dim=extractor_params['embed_dim'],
            batch_norm=False,
            edge_dim=extractor_params['edge_dim']
        )

        # Check if LayerNorm is used
        assert isinstance(extractor_ln.norm_layers[0], nn.LayerNorm)

    def test_forward_pass_with_edge_attr(self, extractor_params, graph_data):
        """Test forward pass with edge attributes."""
        # Create the extractor
        extractor = StructureExtractor(
            embed_dim=extractor_params['embed_dim'],
            edge_dim=extractor_params['edge_dim']
        )

        # Forward pass
        output = extractor(
            graph_data['x'],
            graph_data['edge_index'],
            graph_data['edge_attr']
        )

        # Check output
        assert output.shape == (extractor_params['num_nodes'], extractor_params['embed_dim'])
        assert not torch.isnan(output).any()

    def test_forward_pass_without_edge_attr(self, extractor_params, graph_data):
        """Test forward pass without edge attributes."""
        # Create the extractor
        extractor = StructureExtractor(
            embed_dim=extractor_params['embed_dim'],
            edge_dim=extractor_params['edge_dim']
        )

        # Forward pass without edge attributes
        output = extractor(
            graph_data['x'],
            graph_data['edge_index']
        )

        # Check output
        assert output.shape == (extractor_params['num_nodes'], extractor_params['embed_dim'])
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, extractor_params, graph_data):
        """Test if gradients flow correctly through the extractor."""
        # Create the extractor
        extractor = StructureExtractor(
            embed_dim=extractor_params['embed_dim'],
            edge_dim=extractor_params['edge_dim']
        )

        # Enable gradient tracking
        x = graph_data['x'].clone().requires_grad_(True)
        edge_attr = graph_data['edge_attr'].clone().requires_grad_(True)

        # Forward pass
        output = extractor(x, graph_data['edge_index'], edge_attr)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check if gradients are computed
        assert x.grad is not None
        assert edge_attr.grad is not None
        assert extractor.edge_proj.weight.grad is not None
        for layer in extractor.conv_layers:
            assert layer.W.weight.grad is not None


class TestDAGTransformerLayer:
    """Tests for the DAGTransformerLayer component."""

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
            'num_edge_types': 3
        }

    @pytest.fixture
    def dag_data(self, transformer_params):
        """Create a DAG for testing the transformer layer."""
        d_model = transformer_params['d_model']
        num_nodes = transformer_params['num_nodes']
        max_hop = transformer_params['max_hop']
        num_edge_types = transformer_params['num_edge_types']

        # Create node features
        x = torch.randn(num_nodes, d_model)

        # Create a simple DAG structure
        edge_list = []
        for i in range(num_nodes-1):
            # Each node connects to a few nodes ahead (ensuring DAG property)
            for j in range(i+1, min(i+4, num_nodes)):
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

        # Compute all-pairs shortest paths
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                        distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

        # Create embeddings
        query_hop_emb = nn.Embedding(max_hop + 1, d_model)
        key_hop_emb = nn.Embedding(max_hop + 1, d_model)
        value_hop_emb = nn.Embedding(max_hop + 1, d_model)
        query_edge_emb = nn.Embedding(num_edge_types, d_model)
        key_edge_emb = nn.Embedding(num_edge_types, d_model)
        value_edge_emb = nn.Embedding(num_edge_types, d_model)

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'distance_matrix': distance_matrix,
            'query_hop_emb': query_hop_emb,
            'key_hop_emb': key_hop_emb,
            'value_hop_emb': value_hop_emb,
            'query_edge_emb': query_edge_emb,
            'key_edge_emb': key_edge_emb,
            'value_edge_emb': value_edge_emb
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

        # Check dimensions
        d_model = transformer_params['d_model']

        assert transformer.q_proj.weight.shape == (d_model, d_model)
        assert transformer.k_proj.weight.shape == (d_model, d_model)
        assert transformer.v_proj.weight.shape == (d_model, d_model)
        assert transformer.out_proj.weight.shape == (d_model, d_model)

        # Check feedforward network
        assert transformer.ff[0].weight.shape[0] == transformer_params['dim_feedforward']
        assert transformer.ff[3].weight.shape[0] == d_model

    def test_forward_pass(self, transformer_params, dag_data):
        """Test forward pass of the transformer layer."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Forward pass
        output = transformer(
            dag_data['x'],
            dag_data['distance_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['key_hop_emb'],
            dag_data['value_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_edge_emb']
        )

        # Check output
        assert output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(output).any()

    def test_forward_pass_with_batch(self, transformer_params, dag_data):
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
        batch[num_nodes//2:] = 1  # Second half is second graph

        # Forward pass with batch
        output = transformer(
            dag_data['x'],
            dag_data['distance_matrix'],
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['key_hop_emb'],
            dag_data['value_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_edge_emb'],
            batch
        )

        # Check output
        assert output.shape == (transformer_params['num_nodes'], transformer_params['d_model'])
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, transformer_params, dag_data):
        """Test if gradients flow correctly through the transformer."""
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
            dag_data['edge_index'],
            dag_data['edge_attr'],
            dag_data['query_hop_emb'],
            dag_data['key_hop_emb'],
            dag_data['value_hop_emb'],
            dag_data['query_edge_emb'],
            dag_data['key_edge_emb'],
            dag_data['value_edge_emb']
        )

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check if gradients are computed
        assert x.grad is not None
        assert transformer.q_proj.weight.grad is not None
        assert transformer.k_proj.weight.grad is not None
        assert transformer.v_proj.weight.grad is not None
        assert transformer.out_proj.weight.grad is not None


class TestModelIntegration:
    """Tests for integrating different layers together."""

    @pytest.fixture
    def model_params(self):
        """Parameters for testing model integration."""
        return {
            'node_features': 4,
            'edge_features': 2,
            'hidden_dim': 32,
            'num_heads': 2,
            'num_nodes': 12
        }

    @pytest.fixture
    def model_data(self, model_params):
        """Create data for integration testing."""
        num_nodes = model_params['num_nodes']
        node_features = model_params['node_features']
        edge_features = model_params['edge_features']
        hidden_dim = model_params['hidden_dim']
        max_hop = 5

        # Create node features
        x = torch.randn(num_nodes, node_features)

        # Create a DAG structure
        edge_list = []
        for i in range(num_nodes-1):
            for j in range(i+1, min(i+3, num_nodes)):
                edge_list.append((i, j))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.randn(edge_index.size(1), edge_features)

        # Create distance matrix
        distance_matrix = torch.ones(num_nodes, num_nodes) * (max_hop + 2)
        distance_matrix.fill_diagonal_(0)

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            distance_matrix[src, dst] = 1

        # Compute all-pairs shortest paths
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                        distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

        # Create embeddings
        query_hop_emb = nn.Embedding(max_hop + 1, hidden_dim)
        key_hop_emb = nn.Embedding(max_hop + 1, hidden_dim)
        value_hop_emb = nn.Embedding(max_hop + 1, hidden_dim)
        query_edge_emb = nn.Embedding(2, hidden_dim)
        key_edge_emb = nn.Embedding(2, hidden_dim)
        value_edge_emb = nn.Embedding(2, hidden_dim)

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'distance_matrix': distance_matrix,
            'query_hop_emb': query_hop_emb,
            'key_hop_emb': key_hop_emb,
            'value_hop_emb': value_hop_emb,
            'query_edge_emb': query_edge_emb,
            'key_edge_emb': key_edge_emb,
            'value_edge_emb': value_edge_emb,
            'max_hop': max_hop
        }

    def test_integrated_forward_pass(self, model_params, model_data):
        """Test a complete forward pass through all layers."""
        hidden_dim = model_params['hidden_dim']

        # Create the layers
        node_proj = nn.Linear(model_params['node_features'], hidden_dim)

        structure_extractor = StructureExtractor(
            embed_dim=hidden_dim,
            edge_dim=model_params['edge_features']
        )

        transformer_layer = DAGTransformerLayer(
            d_model=hidden_dim,
            nhead=model_params['num_heads'],
            dim_feedforward=hidden_dim*2,
            dropout=0.1,
            max_hop=model_data['max_hop']
        )

        # Integrated forward pass

        # 1. Project node features
        h = node_proj(model_data['x'])
        assert h.shape == (model_params['num_nodes'], hidden_dim)

        # 2. Extract structural features
        h_struct = structure_extractor(h, model_data['edge_index'], model_data['edge_attr'])
        assert h_struct.shape == (model_params['num_nodes'], hidden_dim)

        # 3. Apply transformer layer
        output = transformer_layer(
            h_struct,
            model_data['distance_matrix'],
            model_data['edge_index'],
            model_data['edge_attr'],
            model_data['query_hop_emb'],
            model_data['key_hop_emb'],
            model_data['value_hop_emb'],
            model_data['query_edge_emb'],
            model_data['key_edge_emb'],
            model_data['value_edge_emb']
        )

        # Check final output
        assert output.shape == (model_params['num_nodes'], hidden_dim)
        assert not torch.isnan(output).any()

        # Test gradient flow through the entire model
        x = model_data['x'].clone().requires_grad_(True)

        # Repeat forward pass with gradient tracking
        h = node_proj(x)
        h_struct = structure_extractor(h, model_data['edge_index'], model_data['edge_attr'])
        output = transformer_layer(
            h_struct,
            model_data['distance_matrix'],
            model_data['edge_index'],
            model_data['edge_attr'],
            model_data['query_hop_emb'],
            model_data['key_hop_emb'],
            model_data['value_hop_emb'],
            model_data['query_edge_emb'],
            model_data['key_edge_emb'],
            model_data['value_edge_emb']
        )

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check if gradients flowed through the entire model
        assert x.grad is not None
        assert node_proj.weight.grad is not None
        assert structure_extractor.conv_layers[0].W.weight.grad is not None
        assert transformer_layer.q_proj.weight.grad is not None


if __name__ == "__main__":
    pytest.main(["-v"])
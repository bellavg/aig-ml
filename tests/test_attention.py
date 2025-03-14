import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the layer class from your module (adjust as needed)
from model import DAGTransformerLayer


class TestRefactoredDAGTransformerLayer:
    """Tests for the refactored DAGTransformerLayer with subfunctions."""

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

    def test_project_qkv(self, transformer_params, dag_data):
        """Test the _project_qkv subfunction."""
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

        # Test the project_qkv function
        q, k, v = transformer._project_qkv(x_norm)

        # Check outputs
        batch_size = dag_data['x'].size(0)
        assert q.shape == (
        batch_size, transformer_params['nhead'], transformer_params['d_model'] // transformer_params['nhead'])
        assert k.shape == (
        batch_size, transformer_params['nhead'], transformer_params['d_model'] // transformer_params['nhead'])
        assert v.shape == (
        batch_size, transformer_params['nhead'], transformer_params['d_model'] // transformer_params['nhead'])

        # Verify gradient flow
        q_sum = q.sum()
        q_sum.backward(retain_graph=True)
        assert transformer.q_proj.weight.grad is not None

        # Reset gradients for next test
        transformer.zero_grad()

    def test_handle_empty_input(self, transformer_params):
        """Test handling of empty inputs."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Call the empty input handler
        result = transformer._handle_empty_input()

        # Check that it returns a tensor
        assert isinstance(result, torch.Tensor)

        # Verify gradient flow
        result.sum().backward()

        # Check if gradients flow to the projection layers
        assert transformer.q_proj.weight.grad is not None
        assert transformer.k_proj.weight.grad is not None

        # Reset gradients for next test
        transformer.zero_grad()

    def test_ensure_gradient_flow(self, transformer_params):
        """Test the gradient flow ensuring mechanism."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Create a dummy tensor
        dummy_tensor = torch.zeros(transformer_params['num_nodes'], transformer_params['d_model'])

        # Apply the gradient flow mechanism
        result = transformer._ensure_gradient_flow(dummy_tensor)

        # Check that it returns a tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == dummy_tensor.shape

        # Verify gradient flow
        result.sum().backward()

        # Check if gradients flow to all projection layers
        assert transformer.q_proj.weight.grad is not None
        assert transformer.k_proj.weight.grad is not None
        assert transformer.v_proj.weight.grad is not None
        assert transformer.out_proj.weight.grad is not None

        # Reset gradients for next test
        transformer.zero_grad()

    def test_compute_basic_attention(self, transformer_params, dag_data):
        """Test the basic attention computation."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Apply normalization and projection
        x_norm = transformer.norm1(dag_data['x'])
        q, k, v = transformer._project_qkv(x_norm)

        # Get a subset of nodes for testing
        node_indices = torch.arange(5)
        graph_q = q[node_indices]
        graph_k = k[node_indices]
        graph_distance = dag_data['distance_matrix'][node_indices][:, node_indices]

        # Compute basic attention
        attn_scores = transformer._compute_basic_attention(graph_q, graph_k, graph_distance)

        # Check output shape
        assert attn_scores.shape == (5, 5, transformer_params['nhead'])

        # Verify that attention to unreachable nodes is masked
        mask = (graph_distance >= transformer_params['max_hop'] + 1).unsqueeze(-1).expand(-1, -1,
                                                                                          transformer_params['nhead'])
        assert torch.all(attn_scores.masked_select(mask) < -1e8)

        # Verify gradient flow
        attn_scores.sum().backward()
        assert transformer.q_proj.weight.grad is not None
        assert transformer.k_proj.weight.grad is not None

        # Reset gradients for next test
        transformer.zero_grad()

    def test_add_topology_attention(self, transformer_params, dag_data):
        """Test the topology-based attention contribution."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Apply normalization and projection
        x_norm = transformer.norm1(dag_data['x'])
        q, k, v = transformer._project_qkv(x_norm)

        # Get a subset of nodes for testing
        node_indices = torch.arange(5)
        graph_q = q[node_indices]
        graph_k = k[node_indices]
        graph_distance = dag_data['distance_matrix'][node_indices][:, node_indices]

        # Compute basic attention
        attn_scores = transformer._compute_basic_attention(graph_q, graph_k, graph_distance)

        # Add topology attention
        attn_scores_with_topology = transformer._add_topology_attention(
            attn_scores.clone(), graph_q, graph_k, graph_distance,
            dag_data['query_hop_emb'], dag_data['key_hop_emb'], len(node_indices)
        )

        # Check that topology attention changes the scores
        assert not torch.allclose(attn_scores, attn_scores_with_topology)

        # Verify gradient flow
        attn_scores_with_topology.sum().backward()
        assert transformer.q_proj.weight.grad is not None
        assert transformer.k_proj.weight.grad is not None
        assert dag_data['query_hop_emb'].weight.grad is not None
        assert dag_data['key_hop_emb'].weight.grad is not None

        # Reset gradients for next test
        transformer.zero_grad()
        dag_data['query_hop_emb'].zero_grad()
        dag_data['key_hop_emb'].zero_grad()

    def test_add_edge_attention(self, transformer_params, dag_data):
        """Test the edge-based attention contribution."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Apply normalization and projection
        x_norm = transformer.norm1(dag_data['x'])
        q, k, v = transformer._project_qkv(x_norm)

        # Get a subset of nodes for testing
        node_indices = torch.arange(5)
        graph_q = q[node_indices]
        graph_k = k[node_indices]

        # Create initial attention scores
        attn_scores = torch.zeros(5, 5, transformer_params['nhead'], device=graph_q.device)

        # Add edge attention
        attn_scores_with_edges = transformer._add_edge_attention(
            attn_scores.clone(), node_indices, graph_q, graph_k,
            dag_data['edge_index'], dag_data['edge_attr'],
            dag_data['query_edge_emb'], dag_data['key_edge_emb']
        )

        # Verify gradient flow (if edges exist between the selected nodes)
        if not torch.all(attn_scores == attn_scores_with_edges):
            attn_scores_with_edges.sum().backward()
            assert dag_data['query_edge_emb'].weight.grad is not None
            assert dag_data['key_edge_emb'].weight.grad is not None

            # Reset gradients for next test
            dag_data['query_edge_emb'].zero_grad()
            dag_data['key_edge_emb'].zero_grad()

    def test_apply_attention_to_values(self, transformer_params, dag_data):
        """Test applying attention to values."""
        # Create the transformer
        transformer = DAGTransformerLayer(
            transformer_params['d_model'],
            transformer_params['nhead'],
            transformer_params['dim_feedforward'],
            transformer_params['dropout'],
            transformer_params['max_hop']
        )

        # Apply normalization and projection
        x_norm = transformer.norm1(dag_data['x'])
        q, k, v = transformer._project_qkv(x_norm)

        # Get a subset of nodes for testing
        node_indices = torch.arange(5)
        graph_v = v[node_indices]
        graph_distance = dag_data['distance_matrix'][node_indices][:, node_indices]

        # Create attention probabilities
        attn_probs = torch.softmax(torch.randn(5, 5, transformer_params['nhead']), dim=1)

        # Apply attention to values
        output = transformer._apply_attention_to_values(
            attn_probs, graph_v, graph_distance,
            dag_data['value_hop_emb'], len(node_indices)
        )

        # Check output shape
        assert output.shape == (5, transformer_params['d_model'])

        # Verify gradient flow
        output.sum().backward()
        assert dag_data['value_hop_emb'].weight.grad is not None

        # Reset gradients for next test
        dag_data['value_hop_emb'].zero_grad()

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

    def test_gradient_flow_through_all_parameters(self, transformer_params, dag_data):
        """Comprehensive test for gradient flow through all parameters."""
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

        # List all parameters to check for gradients
        parameter_names = []
        missing_gradients = []

        for name, param in transformer.named_parameters():
            parameter_names.append(name)
            if param.grad is None or torch.all(param.grad == 0):
                missing_gradients.append(name)

        # Check if ANY parameter is missing gradients
        assert len(missing_gradients) == 0, f"Parameters missing gradients: {missing_gradients}"

        # Specifically check key parameters
        assert transformer.q_proj.weight.grad is not None
        assert transformer.k_proj.weight.grad is not None
        assert transformer.v_proj.weight.grad is not None
        assert transformer.out_proj.weight.grad is not None
        assert transformer.ff[0].weight.grad is not None
        assert transformer.ff[3].weight.grad is not None

        # Check for non-zero gradients
        total_grad_norm = sum(p.grad.detach().abs().sum() for p in transformer.parameters() if p.grad is not None)
        assert total_grad_norm > 0, "Gradient norm is zero, no effective learning would happen"
import unittest
import torch
from torch_geometric.data import Data, Batch
import sys
from typing import Dict, Any, Tuple, List, Optional

# Import the module containing the function to test
# Assuming the original code is in a file called masking.py
# If not, please adjust the import statement
sys.path.append('../')  # Adjust if needed
from masking import create_masked_batch, _identify_and_gates


class TestNodeFeatureMasking(unittest.TestCase):
    """Test suite for the node feature masking functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple test AIG graph with the following structure:
        # - 2 input nodes (type [1,0,0])
        # - 2 AND gates (type [0,1,0])
        # - 1 output node (type [0,0,1])
        # Input1 and Input2 feed into both AND gates
        # Both AND gates feed into the output

        # Create node features
        # Format: [node_type (3), truth_table_value (1)]
        x = torch.tensor([
            [1, 0, 0, 0.5],  # Input 1
            [1, 0, 0, 0.7],  # Input 2
            [0, 1, 0, 0.9],  # AND gate 1 with truth table value 0.9
            [0, 1, 0, 0.8],  # AND gate 2 with truth table value 0.8
            [0, 0, 1, 0.0],  # Output
        ], dtype=torch.float)

        # Create edges (source, destination)
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 3],  # From
            [2, 3, 2, 3, 4, 4],  # To
        ], dtype=torch.long)

        # Edge attributes (optional)
        edge_attr = torch.tensor([
            [1.0],  # Input1 -> AND1
            [1.0],  # Input1 -> AND2
            [1.0],  # Input2 -> AND1
            [1.0],  # Input2 -> AND2
            [1.0],  # AND1 -> Output
            [1.0],  # AND2 -> Output
        ], dtype=torch.float)

        # Create PyG Data objects for individual graphs
        self.single_graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # Create a second graph with slightly different values for batch testing
        x2 = x.clone()
        x2[:, 3] += 0.1  # Add 0.1 to all truth table values

        self.second_graph = Data(
            x=x2,
            edge_index=edge_index.clone(),
            edge_attr=edge_attr.clone()
        )

        # Create a batch with these graphs
        self.batch_size = 2
        self.test_batch = Batch.from_data_list([self.single_graph, self.second_graph])

        # Create a single graph with batch attribute manually added
        # This is important for testing since your implementation expects batch
        self.test_graph = self.single_graph.clone()
        self.test_graph.batch = torch.zeros(self.test_graph.x.size(0), dtype=torch.long,
                                            device=self.test_graph.x.device)

    def test_identify_and_gates(self):
        """Test that AND gates are correctly identified."""
        is_and_gate = _identify_and_gates(self.test_graph)

        expected_mask = torch.tensor([False, False, True, True, False])
        self.assertTrue(torch.all(is_and_gate == expected_mask),
                        f"Expected AND gate mask: {expected_mask}, got: {is_and_gate}")

        # Test on batch
        is_and_gate_batch = _identify_and_gates(self.test_batch)
        expected_batch_mask = torch.cat([expected_mask, expected_mask])
        self.assertTrue(torch.all(is_and_gate_batch == expected_batch_mask),
                        "AND gate identification failed on batch")

    def test_node_feature_masking_single_graph(self):
        """Test that node feature masking works correctly on a single graph."""
        # Create masked batch with 100% masking probability to ensure all AND gates are masked
        masked_graph = create_masked_batch(self.test_graph, mp=1.0, mask_mode="node_feature", mask_value=-1.0)

        # Check that the mask was created
        self.assertTrue(hasattr(masked_graph, 'node_mask'), "node_mask attribute missing")

        # Verify AND gates are masked
        is_and_gate = _identify_and_gates(self.test_graph)
        and_gates = torch.nonzero(is_and_gate).squeeze(-1)

        # Check all AND gates are masked
        for gate_idx in and_gates:
            self.assertTrue(masked_graph.node_mask[gate_idx],
                            f"AND gate at index {gate_idx} should be masked")

            # Check truth table value is masked with the mask_value
            self.assertEqual(masked_graph.x[gate_idx, 3].item(), -1.0,
                             f"Truth table value for AND gate at index {gate_idx} should be -1.0")

        # Check non-AND gates are not masked
        non_and_gates = torch.nonzero(~is_and_gate).squeeze(-1)
        for non_gate_idx in non_and_gates:
            self.assertFalse(masked_graph.node_mask[non_gate_idx],
                             f"Non-AND gate at index {non_gate_idx} should not be masked")

            # Check original value is preserved
            self.assertEqual(masked_graph.x[non_gate_idx, 3].item(),
                             self.test_graph.x[non_gate_idx, 3].item(),
                             f"Truth table value for non-AND gate should not change")

    def test_node_feature_masking_batch(self):
        """Test that node feature masking works correctly on a batch of graphs."""
        # Create masked batch with 100% masking probability
        masked_batch = create_masked_batch(self.test_batch, mp=1.0, mask_mode="node_feature", mask_value=-1.0)

        # Check that the mask was created
        self.assertTrue(hasattr(masked_batch, 'node_mask'), "node_mask attribute missing")

        # Verify AND gates are masked in each graph of the batch
        is_and_gate = _identify_and_gates(self.test_batch)
        for b in range(self.batch_size):
            # Get the nodes for this graph
            graph_nodes = torch.nonzero(self.test_batch.batch == b).squeeze(-1)

            # Get AND gates for this graph
            graph_and_gates = torch.nonzero(is_and_gate & (self.test_batch.batch == b)).squeeze(-1)

            # Check all AND gates in this graph are masked
            for gate_idx in graph_and_gates:
                self.assertTrue(masked_batch.node_mask[gate_idx],
                                f"AND gate at index {gate_idx} in graph {b} should be masked")

                # Check truth table value is masked
                self.assertEqual(masked_batch.x[gate_idx, 3].item(), -1.0,
                                 f"Truth table value for AND gate at index {gate_idx} should be -1.0")

    def test_masking_probability(self):
        """Test that the masking probability affects the number of masked nodes."""
        # Test with various masking probabilities
        masking_probs = [0.0, 0.5, 1.0]

        for mp in masking_probs:
            masked_graph = create_masked_batch(self.test_graph, mp=mp, mask_mode="node_feature")

            # Count the number of AND gates
            is_and_gate = _identify_and_gates(self.test_graph)
            total_and_gates = is_and_gate.sum().item()

            # Count how many are masked
            masked_and_gates = masked_graph.node_mask.sum().item()

            if mp == 0.0:
                # With mp=0.0, we should still mask at least 1 AND gate per graph
                self.assertGreaterEqual(masked_and_gates, 1,
                                        "Should mask at least 1 AND gate even with mp=0.0")
            elif mp == 1.0:
                # With mp=1.0, all AND gates should be masked
                self.assertEqual(masked_and_gates, total_and_gates,
                                 "All AND gates should be masked with mp=1.0")

    def test_original_values_stored(self):
        """Test that original values are stored correctly for loss computation."""
        masked_graph = create_masked_batch(self.test_graph, mp=1.0, mask_mode="node_feature")

        # Check that original values are stored
        self.assertTrue(hasattr(masked_graph, 'original_truth_table_values'),
                        "original_truth_table_values attribute missing")

        # Check original values match
        is_and_gate = _identify_and_gates(self.test_graph)
        and_gates = torch.nonzero(is_and_gate).squeeze(-1)

        # Get the masked AND gates (all of them with mp=1.0)
        masked_and_gates = torch.nonzero(masked_graph.node_mask).squeeze(-1)

        # Check that they match the expected AND gates
        self.assertEqual(len(masked_and_gates), len(and_gates),
                         "Number of masked gates doesn't match expected count")

        # Check that original values were stored correctly
        # Note: original_truth_table_values stores values for masked nodes only
        for i, gate_idx in enumerate(masked_and_gates):
            original_value = self.test_graph.x[gate_idx, 3].item()
            # The index in original_truth_table_values depends on the order of masking
            # We need to find the right index
            found = False
            for j in range(len(masked_graph.original_truth_table_values)):
                if abs(masked_graph.original_truth_table_values[j].item() - original_value) < 1e-5:
                    found = True
                    break
            self.assertTrue(found, f"Original value {original_value} not found in stored values")

    def test_mask_metadata(self):
        """Test that mask metadata is correctly stored."""
        mask_value = -99.0  # Unusual value to ensure it's stored correctly
        masked_graph = create_masked_batch(self.test_graph, mp=0.5,
                                           mask_mode="node_feature",
                                           mask_value=mask_value)

        # Check mask settings are stored
        self.assertEqual(masked_graph.mask_mode, "node_feature", "mask_mode not stored correctly")
        self.assertEqual(masked_graph.mask_prob, 0.5, "mask_prob not stored correctly")
        self.assertEqual(masked_graph.node_mask_value, mask_value, "mask_value not stored correctly")
        self.assertEqual(masked_graph.truth_table_idx, 3, "truth_table_idx not stored correctly")
        self.assertEqual(masked_graph.node_type_dim, 3, "node_type_dim not stored correctly")

    def test_target_preservation(self):
        """Test that x_target preserves the original node features."""
        masked_graph = create_masked_batch(self.test_graph, mp=1.0, mask_mode="node_feature")

        # Check target is created and matches original
        self.assertTrue(hasattr(masked_graph, 'x_target'), "x_target attribute missing")
        self.assertTrue(torch.all(masked_graph.x_target == self.test_graph.x),
                        "x_target should match original x")

    def test_only_truth_table_masked(self):
        """Test that only the truth table value is masked, not the node type."""
        masked_graph = create_masked_batch(self.test_graph, mp=1.0, mask_mode="node_feature")

        # Check that node types are preserved for masked nodes
        is_and_gate = _identify_and_gates(self.test_graph)
        and_gates = torch.nonzero(is_and_gate).squeeze(-1)

        for gate_idx in and_gates:
            # Check node type is preserved (first 3 dimensions)
            for dim in range(3):
                self.assertEqual(masked_graph.x[gate_idx, dim].item(),
                                 self.test_graph.x[gate_idx, dim].item(),
                                 f"Node type dimension {dim} should not be masked")

    def test_single_and_gate_handling(self):
        """Test that a graph with only one AND gate is handled correctly."""
        # Create a simple graph with just one AND gate
        x = torch.tensor([
            [1, 0, 0, 0.5],  # Input 1
            [1, 0, 0, 0.7],  # Input 2
            [0, 1, 0, 0.9],  # The only AND gate
            [0, 0, 1, 0.0],  # Output
        ], dtype=torch.float)

        edge_index = torch.tensor([
            [0, 1, 2],  # From
            [2, 2, 3],  # To
        ], dtype=torch.long)

        edge_attr = torch.tensor([
            [1.0],  # Input1 -> AND
            [1.0],  # Input2 -> AND
            [1.0],  # AND -> Output
        ], dtype=torch.float)

        single_and_graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=torch.zeros(x.size(0), dtype=torch.long)  # Add batch info
        )

        # Create masked batch
        masked_graph = create_masked_batch(single_and_graph, mp=1.0, mask_mode="node_feature", mask_value=-1.0)

        # Check that the AND gate is masked
        is_and_gate = _identify_and_gates(single_and_graph)
        and_gate_idx = torch.nonzero(is_and_gate).squeeze(-1).item()  # Should be just one value (2)

        self.assertEqual(and_gate_idx, 2, "AND gate should be at index 2")
        self.assertTrue(masked_graph.node_mask[and_gate_idx], "The only AND gate should be masked")
        self.assertEqual(masked_graph.x[and_gate_idx, 3].item(), -1.0, "AND gate truth table should be masked")

    def test_batch_with_different_and_gate_counts(self):
        """Test that batches with different numbers of AND gates in each graph are handled correctly."""
        # First graph has 2 AND gates
        x1 = torch.tensor([
            [1, 0, 0, 0.5],  # Input 1
            [1, 0, 0, 0.7],  # Input 2
            [0, 1, 0, 0.9],  # AND gate 1
            [0, 1, 0, 0.8],  # AND gate 2
            [0, 0, 1, 0.0],  # Output
        ], dtype=torch.float)

        edge_index1 = torch.tensor([
            [0, 0, 1, 1, 2, 3],  # From
            [2, 3, 2, 3, 4, 4],  # To
        ], dtype=torch.long)

        edge_attr1 = torch.ones((edge_index1.size(1), 1), dtype=torch.float)

        graph1 = Data(
            x=x1,
            edge_index=edge_index1,
            edge_attr=edge_attr1
        )

        # Second graph has 3 AND gates
        x2 = torch.tensor([
            [1, 0, 0, 0.1],  # Input 1
            [1, 0, 0, 0.2],  # Input 2
            [0, 1, 0, 0.3],  # AND gate 1
            [0, 1, 0, 0.4],  # AND gate 2
            [0, 1, 0, 0.5],  # AND gate 3
            [0, 0, 1, 0.0],  # Output
        ], dtype=torch.float)

        edge_index2 = torch.tensor([
            [0, 0, 1, 1, 2, 3, 4],  # From
            [2, 3, 3, 4, 5, 5, 5],  # To
        ], dtype=torch.long)

        edge_attr2 = torch.ones((edge_index2.size(1), 1), dtype=torch.float)

        graph2 = Data(
            x=x2,
            edge_index=edge_index2,
            edge_attr=edge_attr2
        )

        # Create a batch with these two graphs
        mixed_batch = Batch.from_data_list([graph1, graph2])

        # Apply masking with 100% probability to ensure all AND gates are masked
        masked_batch = create_masked_batch(mixed_batch, mp=1.0, mask_mode="node_feature", mask_value=-1.0)

        # Verify all AND gates are masked in both graphs
        is_and_gate = _identify_and_gates(mixed_batch)

        # First graph (indices 0-4)
        graph1_and_gates = torch.nonzero(is_and_gate & (mixed_batch.batch == 0)).squeeze(-1)
        self.assertEqual(len(graph1_and_gates), 2, "Graph 1 should have 2 AND gates")

        # Second graph (indices 5+)
        graph2_and_gates = torch.nonzero(is_and_gate & (mixed_batch.batch == 1)).squeeze(-1)
        self.assertEqual(len(graph2_and_gates), 3, "Graph 2 should have 3 AND gates")

        # Check all AND gates in both graphs are masked
        for gate_idx in torch.cat([graph1_and_gates, graph2_and_gates]):
            self.assertTrue(masked_batch.node_mask[gate_idx], f"AND gate at index {gate_idx} should be masked")
            self.assertEqual(masked_batch.x[gate_idx, 3].item(), -1.0,
                             f"Truth table for AND gate at index {gate_idx} should be masked")


if __name__ == "__main__":
    unittest.main()
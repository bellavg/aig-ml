import unittest
import torch
from torch_geometric.data import Data, Batch
import sys
import os
import random
import numpy as np

# Check if CUDA is available
HAS_CUDA = torch.cuda.is_available()

# Import your masking function - adjust the path as needed
sys.path.append(os.path.abspath('..'))
from masking import create_masked_batch


class TestTruthTableMasking(unittest.TestCase):

    def setUp(self):
        """Create synthetic AIG graphs for testing."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # Create a small synthetic graph resembling an AIG
        # Node types: [1,0,0] = PI, [0,1,0] = AND, [0,0,1] = PO
        # Fourth value (index 3) is the truth table value
        self.single_graph = Data(
            x=torch.tensor([
                [1, 0, 0, 0.5],  # PI
                [1, 0, 0, 0.2],  # PI
                [0, 1, 0, 1.0],  # AND
                [0, 1, 0, 12.0],  # AND
                [0, 0, 1, 3.6],  # PO
            ], dtype=torch.float),
            edge_index=torch.tensor([
                [0, 1, 2, 3],  # From
                [2, 2, 3, 4],  # To
            ], dtype=torch.long),
            edge_attr=torch.tensor([
                [1, 0],  # Normal
                [1, 0],  # Normal
                [1, 0],  # Normal
                [1, 0],  # Normal
            ], dtype=torch.float)
        )

        # Create a larger graph for testing
        self.larger_graph = Data(
            x=torch.tensor([
                [1, 0, 0, 2.0],  # PI
                [1, 0, 0, 4.0],  # PI
                [1, 0, 0, 6.0],  # PI
                [0, 1, 0, 8.0],  # AND
                [0, 1, 0, 1.0],  # AND
                [0, 1, 0, 4.0],  # AND
                [0, 1, 0, 4.0],  # AND
                [0, 0, 1, 5.0],  # PO
                [0, 0, 1, 2.2],  # PO
            ], dtype=torch.float),
            edge_index=torch.tensor([
                [0, 1, 2, 0, 1, 3, 4, 5, 6],  # From
                [3, 4, 5, 6, 6, 7, 7, 8, 8],  # To
            ], dtype=torch.long),
            edge_attr=torch.tensor([
                [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]
            ], dtype=torch.float)
        )

        # Create batch of multiple graphs
        self.batch = Batch.from_data_list([
            self.single_graph,
            self.single_graph  # Just duplicate for simplicity
        ])

        # Create a batch with larger graphs
        self.larger_batch = Batch.from_data_list([
            self.larger_graph,
            self.larger_graph
        ])

    def test_truth_table_masking_basics(self):
        """Test that only the truth table value (index 3) is masked for AND gates."""
        mask_prob = 1.0  # Mask all eligible AND gates

        masked_batch = create_masked_batch(
            self.batch,
            mp=mask_prob,
            mask_mode="node_feature"
        )

        # Check that attributes are preserved and added
        self.assertTrue(hasattr(masked_batch, 'x_target'))
        self.assertTrue(hasattr(masked_batch, 'node_mask'))
        self.assertTrue(hasattr(masked_batch, 'truth_table_idx'))
        self.assertTrue(hasattr(masked_batch, 'node_type_dim'))

        # Check that the truth table index is correctly set to 3
        self.assertEqual(masked_batch.truth_table_idx, 3)

        # Check that masking mode is stored
        self.assertEqual(masked_batch.mask_mode, "node_feature")

        # Identify AND gates
        is_and_gate = (masked_batch.x_target[:, 0] == 0) & (masked_batch.x_target[:, 1] == 1) & (
                    masked_batch.x_target[:, 2] == 0)
        and_gate_indices = torch.nonzero(is_and_gate).squeeze()

        # For each node in the batch
        for i in range(masked_batch.x.size(0)):
            if masked_batch.node_mask[i]:
                # Should be an AND gate
                self.assertTrue(is_and_gate[i].item())

                # Only the truth table value should be masked
                # Node type (first 3 dimensions) should be preserved
                self.assertTrue(torch.equal(
                    masked_batch.x[i, :3],
                    masked_batch.x_target[i, :3]
                ))

                # Truth table value at index 3 should be zero
                self.assertEqual(masked_batch.x[i, 3].item(), 0.0)
            elif is_and_gate[i]:
                # If it's an AND gate but not masked, it should be unchanged
                self.assertTrue(torch.equal(
                    masked_batch.x[i],
                    masked_batch.x_target[i]
                ))
            else:
                # Non-AND gates should never be masked
                self.assertFalse(masked_batch.node_mask[i].item())

                # Features should remain unchanged
                self.assertTrue(torch.equal(
                    masked_batch.x[i],
                    masked_batch.x_target[i]
                ))

    def test_masking_only_and_gates(self):
        """Test that only AND gates are masked, not inputs or outputs."""
        mask_prob = 1.0  # Mask all eligible nodes

        masked_batch = create_masked_batch(
            self.batch,
            mp=mask_prob,
            mask_mode="node_feature"
        )

        # Identify different node types
        is_and_gate = (masked_batch.x_target[:, 0] == 0) & (masked_batch.x_target[:, 1] == 1) & (
                    masked_batch.x_target[:, 2] == 0)
        is_input = (masked_batch.x_target[:, 0] == 1) & (masked_batch.x_target[:, 1] == 0) & (
                    masked_batch.x_target[:, 2] == 0)
        is_output = (masked_batch.x_target[:, 0] == 0) & (masked_batch.x_target[:, 1] == 0) & (
                    masked_batch.x_target[:, 2] == 1)

        # Check that only AND gates are masked
        for i in range(masked_batch.x.size(0)):
            if is_input[i] or is_output[i]:
                # Inputs and outputs should never be masked
                self.assertFalse(masked_batch.node_mask[i].item())

                # Features should remain unchanged
                self.assertTrue(torch.equal(
                    masked_batch.x[i],
                    masked_batch.x_target[i]
                ))

            # All AND gates should be masked (when mask_prob=1.0)
            if is_and_gate[i]:
                self.assertTrue(masked_batch.node_mask[i].item())

                # Truth table value should be zero
                self.assertEqual(masked_batch.x[i, 3].item(), 0.0)

    def test_varying_mask_probability(self):
        """Test masking with different probabilities."""
        # Test with different masking probabilities
        for mask_prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
            masked_batch = create_masked_batch(
                self.larger_batch,
                mp=mask_prob,
                mask_mode="node_feature"
            )

            # Identify AND gates
            is_and_gate = (masked_batch.x_target[:, 0] == 0) & (masked_batch.x_target[:, 1] == 1) & (
                        masked_batch.x_target[:, 2] == 0)
            and_gate_count = is_and_gate.sum().item()

            # Count masked AND gates
            masked_and_count = (masked_batch.node_mask & is_and_gate).sum().item()

            # Check if the number of masked AND gates is approximately as expected
            # Allow for small variations due to randomization and rounding
            expected_masked = int(and_gate_count * mask_prob)

            # Special cases
            if mask_prob == 0.0:
                # Should mask at least 1 gate per graph in batch
                self.assertGreaterEqual(masked_and_count, len(torch.unique(masked_batch.batch)))
            elif mask_prob == 1.0:
                # Should mask all AND gates
                self.assertEqual(masked_and_count, and_gate_count)
            else:
                # Should be close to expected percentage
                self.assertLessEqual(abs(masked_and_count - expected_masked), 2)

    def test_truth_table_preservation(self):
        """Test that non-masked values are preserved correctly."""
        mask_prob = 0.5

        masked_batch = create_masked_batch(
            self.batch,
            mp=mask_prob,
            mask_mode="node_feature"
        )

        # Check that unmasked nodes have their features preserved exactly
        for i in range(masked_batch.x.size(0)):
            if not masked_batch.node_mask[i]:
                self.assertTrue(torch.equal(
                    masked_batch.x[i],
                    masked_batch.x_target[i]
                ))
            else:
                # Only the truth table value should be zeroed
                self.assertEqual(masked_batch.x[i, 3].item(), 0.0)

                # The node type should be preserved
                self.assertTrue(torch.equal(
                    masked_batch.x[i, :3],
                    masked_batch.x_target[i, :3]
                ))

    def test_masking_consistency_across_batches(self):
        """Test that masking behavior is consistent across different batch sizes."""
        # Create different sized batches
        single_graph_batch = Batch.from_data_list([self.single_graph])
        double_graph_batch = Batch.from_data_list([self.single_graph, self.single_graph])
        quad_graph_batch = Batch.from_data_list([self.single_graph, self.single_graph,
                                                 self.single_graph, self.single_graph])

        # Apply masking with same parameters
        mask_prob = 1.0

        single_masked = create_masked_batch(single_graph_batch, mp=mask_prob, mask_mode="node_feature")
        double_masked = create_masked_batch(double_graph_batch, mp=mask_prob, mask_mode="node_feature")
        quad_masked = create_masked_batch(quad_graph_batch, mp=mask_prob, mask_mode="node_feature")

        # Identify AND gates in original graph
        is_and_gate = (self.single_graph.x[:, 0] == 0) & (self.single_graph.x[:, 1] == 1) & (
                    self.single_graph.x[:, 2] == 0)
        and_gate_count = is_and_gate.sum().item()

        # Check that all batches mask all AND gates
        self.assertEqual((single_masked.node_mask).sum().item(), and_gate_count)
        self.assertEqual((double_masked.node_mask).sum().item(), and_gate_count * 2)
        self.assertEqual((quad_masked.node_mask).sum().item(), and_gate_count * 4)

        # Verify that each graph in the batch has its AND gates masked
        if hasattr(double_masked, 'batch'):
            for b in range(2):
                graph_mask = double_masked.batch == b

                # Identify which nodes are AND gates in this specific graph of the batch
                is_and_gate_in_batch = (double_masked.x_target[graph_mask, 0] == 0) & \
                                       (double_masked.x_target[graph_mask, 1] == 1) & \
                                       (double_masked.x_target[graph_mask, 2] == 0)

                and_gate_indices = torch.nonzero(graph_mask).squeeze()[is_and_gate_in_batch]
                masked_indices = torch.nonzero(double_masked.node_mask).squeeze()

                # Check that all AND gates in this graph are in the masked indices
                for idx in and_gate_indices:
                    self.assertIn(idx.item(), masked_indices.tolist())

    def test_masked_batch_integrity(self):
        """Test that the masked batch maintains overall integrity."""
        mask_prob = 0.5

        masked_batch = create_masked_batch(
            self.larger_batch,
            mp=mask_prob,
            mask_mode="node_feature"
        )

        # Check that edge structure is unchanged
        self.assertTrue(torch.equal(masked_batch.edge_index, self.larger_batch.edge_index))
        self.assertTrue(torch.equal(masked_batch.edge_attr, self.larger_batch.edge_attr))

        # Check that number of nodes is unchanged
        self.assertEqual(masked_batch.x.size(0), self.larger_batch.x.size(0))

        # Check that node features have the same shape
        self.assertEqual(masked_batch.x.size(1), self.larger_batch.x.size(1))

        # Check that batch assignment is preserved (if applicable)
        if hasattr(self.larger_batch, 'batch'):
            self.assertTrue(torch.equal(masked_batch.batch, self.larger_batch.batch))

    def test_masking_with_zero_probability(self):
        """Test behavior when masking probability is 0."""
        mask_prob = 0.0

        masked_batch = create_masked_batch(
            self.batch,
            mp=mask_prob,
            mask_mode="node_feature"
        )

        # Should still mask at least one AND gate per graph in batch
        # Identify AND gates
        is_and_gate = (masked_batch.x_target[:, 0] == 0) & (masked_batch.x_target[:, 1] == 1) & (
                    masked_batch.x_target[:, 2] == 0)

        # Count number of unique graphs in batch
        num_graphs = len(torch.unique(masked_batch.batch)) if hasattr(masked_batch, 'batch') else 1

        # Check minimum number of masked nodes
        self.assertGreaterEqual((masked_batch.node_mask).sum().item(), num_graphs)

        # All masked nodes should be AND gates
        self.assertTrue(torch.all(is_and_gate[masked_batch.node_mask]).item())

    def test_masking_with_full_probability(self):
        """Test behavior when masking probability is 1."""
        mask_prob = 1.0

        masked_batch = create_masked_batch(
            self.batch,
            mp=mask_prob,
            mask_mode="node_feature"
        )

        # Should mask all AND gates
        is_and_gate = (masked_batch.x_target[:, 0] == 0) & (masked_batch.x_target[:, 1] == 1) & (
                    masked_batch.x_target[:, 2] == 0)

        # Check that all AND gates are masked
        self.assertEqual((masked_batch.node_mask).sum().item(), is_and_gate.sum().item())

        # Check that all masked nodes are AND gates
        self.assertTrue(torch.all(is_and_gate[masked_batch.node_mask]).item())

        # Check that all AND gates' truth table values are masked
        for i in range(masked_batch.x.size(0)):
            if is_and_gate[i]:
                self.assertEqual(masked_batch.x[i, 3].item(), 0.0)

    def test_edge_structure_preservation(self):
        """Test that edge structure is preserved during node feature masking."""
        mask_prob = 1.0

        masked_batch = create_masked_batch(
            self.batch,
            mp=mask_prob,
            mask_mode="node_feature"
        )

        # Edge structure should be unchanged
        self.assertTrue(torch.equal(masked_batch.edge_index, self.batch.edge_index))
        self.assertTrue(torch.equal(masked_batch.edge_attr, self.batch.edge_attr))

    def test_compatibility_with_edge_feature_masking(self):
        """Test that truth table masking is compatible with edge feature masking."""
        # First apply node feature masking
        node_mask_prob = 0.5
        node_masked_batch = create_masked_batch(
            self.batch,
            mp=node_mask_prob,
            mask_mode="node_feature"
        )

        # Then apply edge feature masking
        edge_mask_prob = 0.5
        edge_masked_batch = create_masked_batch(
            node_masked_batch,
            mp=edge_mask_prob,
            mask_mode="edge_feature"
        )

        # Check that edge masking was applied
        self.assertTrue(hasattr(edge_masked_batch, 'edge_mask'))
        self.assertTrue((edge_masked_batch.edge_mask).sum() > 0)

        # Original node masking info should be preserved
        self.assertTrue(hasattr(edge_masked_batch, 'x_target'))

        # Node features should not be further modified by edge masking
        self.assertTrue(torch.equal(edge_masked_batch.x, node_masked_batch.x))

    def test_compatibility_with_connectivity_masking(self):
        """Test that truth table masking is compatible with connectivity masking."""
        # Create a fresh batch for this test
        test_batch = Batch.from_data_list([self.single_graph])

        # Apply node feature masking first
        node_mask_prob = 1.0  # Mask all AND gates for clarity
        node_masked_batch = create_masked_batch(
            test_batch,
            mp=node_mask_prob,
            mask_mode="node_feature"
        )

        # Store the node-masked x values for later comparison
        node_masked_x = node_masked_batch.x.clone()

        # Then apply connectivity masking to a copy of the batch
        # (this creates a new masking rather than stacking the masking)
        connectivity_mask_prob = 0.5
        connectivity_masked_batch = create_masked_batch(
            test_batch,  # Use original batch, not the node-masked one
            mp=connectivity_mask_prob,
            mask_mode="connectivity"
        )

        # Check that connectivity masking was applied
        self.assertTrue(hasattr(connectivity_masked_batch, 'masked_edge_indices'))
        self.assertTrue((connectivity_masked_batch.masked_edge_indices).numel() > 0)

        # Verify that connectivity masking doesn't affect node features
        # (They should match the original features since we didn't apply node masking)
        self.assertTrue(torch.equal(connectivity_masked_batch.x, test_batch.x))

        # Identify AND gates
        is_and_gate = (test_batch.x[:, 0] == 0) & \
                      (test_batch.x[:, 1] == 1) & \
                      (test_batch.x[:, 2] == 0)

        # Verify our node masking worked as expected (separate test)
        for i in range(test_batch.x.size(0)):
            if is_and_gate[i]:
                # AND gates should have their truth table value masked
                self.assertEqual(node_masked_x[i, 3].item(), 0.0)

                # But node type should be preserved
                self.assertTrue(torch.equal(
                    node_masked_x[i, :3],
                    test_batch.x[i, :3]
                ))


if __name__ == '__main__':
    unittest.main()
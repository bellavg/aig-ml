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


class TestNewMasking(unittest.TestCase):

    def setUp(self):
        """Create synthetic AIG graphs for testing."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # Create a small synthetic graph resembling an AIG
        # Node types: [1,0,0] = PI, [0,1,0] = AND, [0,0,1] = PO
        self.single_graph = Data(
            x=torch.tensor([
                [1, 0, 0],  # PI
                [1, 0, 0],  # PI
                [0, 1, 0],  # AND
                [0, 1, 0],  # AND
                [0, 0, 1],  # PO
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

        # Create a larger graph for testing connectivity masking
        self.larger_graph = Data(
            x=torch.tensor([
                [1, 0, 0],  # PI
                [1, 0, 0],  # PI
                [1, 0, 0],  # PI
                [0, 1, 0],  # AND
                [0, 1, 0],  # AND
                [0, 1, 0],  # AND
                [0, 1, 0],  # AND
                [0, 0, 1],  # PO
                [0, 0, 1],  # PO
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

        # Create a batch with larger graphs for testing connectivity
        self.larger_batch = Batch.from_data_list([
            self.larger_graph,
            self.larger_graph
        ])

    def test_node_feature_masking_basic(self):
        """Basic test for node feature masking mode."""
        mask_prob = 1.0  # Ensure all eligible nodes are masked
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="node_feature")

        # Check that attributes are preserved
        self.assertTrue(hasattr(masked_batch, 'x_target'))
        self.assertTrue(hasattr(masked_batch, 'edge_index_target'))
        self.assertTrue(hasattr(masked_batch, 'edge_attr_target'))
        self.assertTrue(hasattr(masked_batch, 'node_mask'))

        # Check that masking mode is stored
        self.assertEqual(masked_batch.mask_mode, "node_feature")

        # Check that node features are masked (zeroed out) where node_mask is True
        # and only for AND gates (nodes 2, 3, 7, 8 in our batch)
        for i in range(masked_batch.x.size(0)):
            if masked_batch.node_mask[i]:
                self.assertTrue(torch.all(masked_batch.x[i] == 0).item())
                # Check it's an AND gate in the original
                self.assertTrue(torch.all(masked_batch.x_target[i, 1:2] == 1).item())

        # Verify no edges were modified
        self.assertTrue(torch.equal(masked_batch.edge_index, masked_batch.edge_index_target))
        self.assertTrue(torch.equal(masked_batch.edge_attr, masked_batch.edge_attr_target))

    def test_edge_feature_masking_basic(self):
        """Basic test for edge feature masking mode."""
        mask_prob = 1.0  # Ensure all eligible edges are masked
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="edge_feature")

        # Check that attributes are preserved
        self.assertTrue(hasattr(masked_batch, 'edge_mask'))

        # Check that edge features are masked (zeroed out) where edge_mask is True
        for i in range(masked_batch.edge_attr.size(0)):
            if masked_batch.edge_mask[i]:
                self.assertTrue(torch.all(masked_batch.edge_attr[i] == 0).item())

        # Verify edge structure is preserved
        self.assertTrue(torch.equal(masked_batch.edge_index, masked_batch.edge_index_target))
        self.assertEqual(masked_batch.edge_index.size(1), masked_batch.edge_index_target.size(1))

    def test_masking_functions_exist(self):
        """Test that all masking functions are available."""
        modes = ["node_feature", "edge_feature", "connectivity"]

        for mode in modes:
            # Should not raise exception
            masked_batch = create_masked_batch(self.batch, mp=0.5, mask_mode=mode)
            self.assertEqual(masked_batch.mask_mode, mode)

    def test_connectivity_masking_basic(self):
        """Test basic connectivity masking functionality."""
        mask_prob = 0.5  # Mask half of edges
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="connectivity")

        # Check that needed attributes exist
        self.assertTrue(hasattr(masked_batch, 'edge_mask'))
        self.assertTrue(hasattr(masked_batch, 'masked_edge_indices'))
        self.assertTrue(hasattr(masked_batch, 'masked_edge_node_pairs'))
        self.assertTrue(hasattr(masked_batch, 'connectivity_target'))

        # Check that masked edges were removed
        num_original_edges = self.batch.edge_index.size(1)
        num_masked_edges = masked_batch.masked_edge_indices.size(0)
        num_remaining_edges = masked_batch.edge_index.size(1)

        self.assertEqual(num_original_edges, num_remaining_edges + num_masked_edges)

        # Check that connectivity targets are all ones (since they were real edges)
        self.assertTrue(torch.all(masked_batch.connectivity_target == 1).item())

        # Check masked_edge_node_pairs contains source and destination nodes
        self.assertEqual(masked_batch.masked_edge_node_pairs.size(0), 2)  # [source, dest] format
        self.assertEqual(masked_batch.masked_edge_node_pairs.size(1), num_masked_edges)

    def test_connectivity_masking_negative_examples(self):
        """Test that connectivity masking generates negative examples."""
        # Use larger batch to have enough nodes for negative sampling
        mask_prob = 0.3  # Mask 30% of edges
        masked_batch = create_masked_batch(self.larger_batch, mp=mask_prob, mask_mode="connectivity")

        # Check that negative examples were generated
        self.assertTrue(hasattr(masked_batch, 'negative_edge_pairs'))
        self.assertTrue(hasattr(masked_batch, 'negative_edge_targets'))
        self.assertTrue(hasattr(masked_batch, 'all_candidate_pairs'))
        self.assertTrue(hasattr(masked_batch, 'all_candidate_targets'))

        # Check that negative targets are all zeros
        self.assertTrue(torch.all(masked_batch.negative_edge_targets == 0).item())

        # Check combined examples have correct structure
        num_pos = masked_batch.connectivity_target.size(0)
        num_neg = masked_batch.negative_edge_targets.size(0)

        self.assertEqual(masked_batch.all_candidate_pairs.size(1), num_pos + num_neg)
        self.assertEqual(masked_batch.all_candidate_targets.size(0), num_pos + num_neg)

        # Check that negative edge pairs don't exist in the original graph
        edge_set = set()
        for i in range(self.larger_batch.edge_index.size(1)):
            src = self.larger_batch.edge_index[0, i].item()
            dst = self.larger_batch.edge_index[1, i].item()
            edge_set.add((src, dst))

        for i in range(masked_batch.negative_edge_pairs.size(1)):
            src = masked_batch.negative_edge_pairs[0, i].item()
            dst = masked_batch.negative_edge_pairs[1, i].item()
            self.assertFalse((src, dst) in edge_set)

    def test_connectivity_masking_edge_removal(self):
        """Test that connectivity masking actually removes edges from the graph."""
        mask_prob = 0.5  # Mask half of edges
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="connectivity")

        # Get original edges
        original_edges = set()
        for i in range(self.batch.edge_index.size(1)):
            src = self.batch.edge_index[0, i].item()
            dst = self.batch.edge_index[1, i].item()
            original_edges.add((src, dst))

        # Get remaining edges
        remaining_edges = set()
        for i in range(masked_batch.edge_index.size(1)):
            src = masked_batch.edge_index[0, i].item()
            dst = masked_batch.edge_index[1, i].item()
            remaining_edges.add((src, dst))

        # Get masked edges
        masked_edges = set()
        for i in range(masked_batch.masked_edge_node_pairs.size(1)):
            src = masked_batch.masked_edge_node_pairs[0, i].item()
            dst = masked_batch.masked_edge_node_pairs[1, i].item()
            masked_edges.add((src, dst))

        # Verify that:
        # 1. Masked edges are not in remaining edges
        self.assertEqual(len(masked_edges.intersection(remaining_edges)), 0)

        # 2. Union of masked and remaining edges equals original edges
        self.assertEqual(masked_edges.union(remaining_edges), original_edges)

    def test_connectivity_masking_edge_attribute_preservation(self):
        """Test that connectivity masking preserves edge attributes for masked edges."""
        mask_prob = 0.5
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="connectivity")

        # Check that edge attributes are preserved
        self.assertTrue(hasattr(masked_batch, 'masked_edge_attr_target'))

        # Check dimensions match
        num_masked_edges = masked_batch.masked_edge_indices.size(0)
        self.assertEqual(masked_batch.masked_edge_attr_target.size(0), num_masked_edges)

        # Check that attributes match original values
        for i in range(num_masked_edges):
            edge_idx = masked_batch.masked_edge_indices[i].item()
            self.assertTrue(torch.equal(
                masked_batch.masked_edge_attr_target[i],
                masked_batch.edge_attr_target[edge_idx]
            ))

    def test_connectivity_masking_zero_prob(self):
        """Test connectivity masking with zero probability."""
        masked_batch = create_masked_batch(self.batch, mp=0.0, mask_mode="connectivity")

        # Should have masked zero or one edge (implementation uses max(1, int(edges * mp)))
        num_masked = masked_batch.masked_edge_indices.size(0)
        self.assertLessEqual(num_masked, 1)

        # Check that the graph structure is mostly preserved
        self.assertGreaterEqual(masked_batch.edge_index.size(1), self.batch.edge_index.size(1) - 1)

    def test_connectivity_masking_full_prob(self):
        """Test connectivity masking with probability 1.0."""
        masked_batch = create_masked_batch(self.batch, mp=1.0, mask_mode="connectivity")

        # All edges should be masked
        self.assertEqual(masked_batch.masked_edge_indices.size(0), self.batch.edge_index.size(1))

        # No edges should remain
        self.assertEqual(masked_batch.edge_index.size(1), 0)

    def test_connectivity_masked_edge_indices(self):
        """Test that masked_edge_indices correctly identifies masked edges."""
        mask_prob = 0.5
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="connectivity")

        # Check that indices are valid
        self.assertTrue(torch.all(masked_batch.masked_edge_indices < self.batch.edge_index.size(1)))

        # Check that indices correspond to masked edges
        for idx in masked_batch.masked_edge_indices:
            edge_idx = idx.item()
            self.assertTrue(masked_batch.edge_mask[edge_idx].item())

    def test_connectivity_on_complex_graph(self):
        """Test connectivity masking on a more complex graph structure."""
        # Create a more complex graph
        complex_graph = Data(
            x=torch.tensor([
                [1, 0, 0],  # PI
                [1, 0, 0],  # PI
                [1, 0, 0],  # PI
                [0, 1, 0],  # AND
                [0, 1, 0],  # AND
                [0, 0, 1],  # PO
                [0, 0, 1],  # PO
            ], dtype=torch.float),
            edge_index=torch.tensor([
                [0, 1, 2, 3, 3, 4],  # From
                [3, 3, 4, 5, 6, 6],  # To
            ], dtype=torch.long),
            edge_attr=torch.tensor([
                [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]
            ], dtype=torch.float)
        )

        complex_batch = Batch.from_data_list([complex_graph])

        # Apply connectivity masking
        mask_prob = 0.5
        masked_batch = create_masked_batch(complex_batch, mp=mask_prob, mask_mode="connectivity")

        # Check that masking worked
        self.assertGreater(masked_batch.masked_edge_indices.size(0), 0)

        # Check that the graph is still valid (no dangling edges)
        if masked_batch.edge_index.size(1) > 0:
            max_node_idx = complex_graph.x.size(0) - 1
            self.assertTrue(torch.all(masked_batch.edge_index[0] <= max_node_idx))
            self.assertTrue(torch.all(masked_batch.edge_index[1] <= max_node_idx))

    def test_combined_positive_negative_examples(self):
        """Test that combined positive and negative examples are correctly formatted."""
        masked_batch = create_masked_batch(self.larger_batch, mp=0.3, mask_mode="connectivity")

        if hasattr(masked_batch, 'all_candidate_pairs'):
            # Check that combined examples have correct dimensions
            self.assertEqual(masked_batch.all_candidate_pairs.size(0), 2)  # [src, dst] pairs

            # Check that combined targets have matching length
            self.assertEqual(
                masked_batch.all_candidate_pairs.size(1),
                masked_batch.all_candidate_targets.size(0)
            )

            # Check that targets are binary (0 or 1)
            unique_targets = torch.unique(masked_batch.all_candidate_targets)
            self.assertTrue(torch.all(torch.isin(unique_targets, torch.tensor([0., 1.]))).item())

            # First part should be positive examples, second part negative
            pos_count = masked_batch.connectivity_target.size(0)
            total_count = masked_batch.all_candidate_targets.size(0)

            if pos_count < total_count:  # If we have negative examples
                # Check positive targets
                self.assertTrue(torch.all(masked_batch.all_candidate_targets[:pos_count] == 1).item())

                # Check negative targets
                self.assertTrue(torch.all(masked_batch.all_candidate_targets[pos_count:] == 0).item())

    def test_batch_consistency_after_connectivity_masking(self):
        """Test that batch assignment is properly preserved in connectivity masking."""
        # Only test if batch information is available
        if hasattr(self.larger_batch, 'batch'):
            masked_batch = create_masked_batch(self.larger_batch, mp=0.3, mask_mode="connectivity")

            # Batch attribute should still exist
            self.assertTrue(hasattr(masked_batch, 'batch'))

            # Batch assignments should be valid
            num_graphs = len(torch.unique(self.larger_batch.batch))
            self.assertTrue(torch.all(masked_batch.batch < num_graphs))

            # The number of nodes should remain the same
            self.assertEqual(masked_batch.x.size(0), self.larger_batch.x.size(0))

            # Batch assignments should match the originals
            self.assertTrue(torch.equal(masked_batch.batch, self.larger_batch.batch))


if __name__ == '__main__':
    unittest.main()
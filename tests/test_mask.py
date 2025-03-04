import unittest
import torch
from torch_geometric.data import Data, Batch
import sys
import os
import random
import numpy as np

# Import your masking function - adjust the path as needed
sys.path.append(os.path.abspath('..'))
from masking import create_masked_batch


class TestMasking(unittest.TestCase):

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

        # Create batch of multiple graphs
        self.batch = Batch.from_data_list([
            self.single_graph,
            self.single_graph  # Just duplicate for simplicity
        ])

    def test_node_feature_masking(self):
        """Test node feature masking mode."""
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

    def test_edge_feature_masking(self):
        """Test edge feature masking mode."""
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

    def test_node_existence_masking(self):
        """Test node existence masking mode."""
        mask_prob = 1.0  # Ensure all eligible nodes are masked
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="node_existence")

        # Check that attributes are preserved
        self.assertTrue(hasattr(masked_batch, 'node_existence_mask'))
        self.assertTrue(hasattr(masked_batch, 'node_existence_target'))

        # Check that node features are masked where node_mask is True
        for i in range(masked_batch.x.size(0)):
            if masked_batch.node_mask[i]:
                self.assertTrue(torch.all(masked_batch.x[i] == 0).item())
                # Check that the node is also marked for existence prediction
                self.assertTrue(masked_batch.node_existence_mask[i].item())

        # Verify default existence target is all ones
        self.assertTrue(torch.all(masked_batch.node_existence_target == 1).item())

    def test_edge_existence_masking(self):
        """Test edge existence masking mode."""
        mask_prob = 1.0  # Ensure all eligible edges are masked
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="edge_existence")

        # Check that attributes are preserved
        self.assertTrue(hasattr(masked_batch, 'edge_existence_mask'))
        self.assertTrue(hasattr(masked_batch, 'edge_existence_target'))

        # Check that edge features are masked where edge_mask is True
        for i in range(masked_batch.edge_attr.size(0)):
            if masked_batch.edge_mask[i]:
                self.assertTrue(torch.all(masked_batch.edge_attr[i] == 0).item())
                # Check that the edge is also marked for existence prediction
                self.assertTrue(masked_batch.edge_existence_mask[i].item())

        # Verify default existence target is all ones
        self.assertTrue(torch.all(masked_batch.edge_existence_target == 1).item())

        # Verify edge structure is preserved
        self.assertTrue(torch.equal(masked_batch.edge_index, masked_batch.edge_index_target))
        self.assertEqual(masked_batch.edge_index.size(1), masked_batch.edge_index_target.size(1))

    def test_removal_masking(self):
        """Test node removal masking mode."""
        mask_prob = 1.0  # Ensure all eligible nodes are masked
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="removal")

        # Check that attributes are preserved
        self.assertTrue(hasattr(masked_batch, 'node_removal_mask'))
        self.assertTrue(hasattr(masked_batch, 'original_to_new_indices'))
        self.assertTrue(hasattr(masked_batch, 'old_to_new_mapping'))
        self.assertTrue(hasattr(masked_batch, 'num_original_nodes'))

        # Verify node reduction
        self.assertLess(masked_batch.x.size(0), masked_batch.num_original_nodes)

        # Verify structure consistency
        if masked_batch.edge_index.size(1) > 0:
            # All edge indices should be valid for the reduced node set
            self.assertTrue(torch.all(masked_batch.edge_index[0] < masked_batch.x.size(0)).item())
            self.assertTrue(torch.all(masked_batch.edge_index[1] < masked_batch.x.size(0)).item())

        # Check node mask has correct size for the reduced graph
        self.assertEqual(masked_batch.node_mask.size(0), masked_batch.x.size(0))

        # Verify existence targets
        self.assertTrue(hasattr(masked_batch, 'node_existence_target'))
        self.assertTrue(hasattr(masked_batch, 'edge_existence_target'))

        # Check that removed nodes have existence target = 0
        for i in range(masked_batch.num_original_nodes):
            if masked_batch.node_removal_mask[i]:
                self.assertEqual(masked_batch.node_existence_target[i].item(), 0.0)

    def test_edge_consistency_removal(self):
        """Test edge consistency in removal mode."""
        mask_prob = 0.5  # Partial masking
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="removal")

        # Verify that all edges connect to valid nodes
        if masked_batch.edge_index.size(1) > 0:
            max_node_idx = masked_batch.x.size(0) - 1
            self.assertTrue(torch.all(masked_batch.edge_index[0] <= max_node_idx).item())
            self.assertTrue(torch.all(masked_batch.edge_index[1] <= max_node_idx).item())

            # Check that no edges connect to removed nodes
            old_to_new = masked_batch.old_to_new_mapping
            for i in range(masked_batch.edge_index_target.size(1)):
                if masked_batch.edge_mask[i]:
                    # This edge should not be in the reduced graph
                    src, dst = masked_batch.edge_index_target[:, i]
                    # At least one endpoint should be a removed node
                    self.assertTrue(old_to_new[src] == -1 or old_to_new[dst] == -1)

    def test_fixed_mask_size(self):
        """Test that node_mask has the correct size in removal mode."""
        mask_prob = 1.0  # Ensure all eligible nodes are masked
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="removal")

        # The node_mask should match the reduced graph size, not the original
        self.assertEqual(masked_batch.node_mask.size(0), masked_batch.x.size(0))

        # Check node_removal_mask has original size
        self.assertEqual(masked_batch.node_removal_mask.size(0), masked_batch.num_original_nodes)


if __name__ == '__main__':
    unittest.main()
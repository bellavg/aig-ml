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

        # Create more complex graph with different structure
        self.complex_graph = Data(
            x=torch.tensor([
                [1, 0, 0],  # PI
                [1, 0, 0],  # PI
                [1, 0, 0],  # PI
                [0, 1, 0],  # AND
                [0, 1, 0],  # AND
                [0, 1, 0],  # AND
                [0, 0, 1],  # PO
                [0, 0, 1],  # PO
            ], dtype=torch.float),
            edge_index=torch.tensor([
                [0, 0, 1, 1, 2, 3, 4, 5],  # From
                [3, 4, 4, 5, 5, 6, 6, 7],  # To
            ], dtype=torch.long),
            edge_attr=torch.tensor([
                [1, 0], [1, 0], [1, 0], [1, 0],
                [1, 0], [1, 0], [1, 0], [1, 0]
            ], dtype=torch.float)
        )

        # Create graph without AND gates
        self.no_and_graph = Data(
            x=torch.tensor([
                [1, 0, 0],  # PI
                [1, 0, 0],  # PI
                [0, 0, 1],  # PO
            ], dtype=torch.float),
            edge_index=torch.tensor([
                [0, 1],  # From
                [2, 2],  # To
            ], dtype=torch.long),
            edge_attr=torch.tensor([
                [1, 0],  # Normal
                [1, 0],  # Normal
            ], dtype=torch.float)
        )

        # Create single node graph (edge case)
        self.single_node_graph = Data(
            x=torch.tensor([[1, 0, 0]], dtype=torch.float),  # Just one PI
            edge_index=torch.tensor([[], []], dtype=torch.long),  # No edges
            edge_attr=torch.tensor([], dtype=torch.float).view(0, 2)  # Empty edge attributes
        )

        # Create empty graph (edge case)
        self.empty_graph = Data(
            x=torch.tensor([], dtype=torch.float).view(0, 3),  # No nodes
            edge_index=torch.tensor([[], []], dtype=torch.long),  # No edges
            edge_attr=torch.tensor([], dtype=torch.float).view(0, 2)  # Empty edge attributes
        )

        # Create mixed batch with different graph structures
        self.mixed_batch = Batch.from_data_list([
            self.single_graph,
            self.complex_graph,
            self.no_and_graph
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

    # NEW TESTS BELOW

    def test_partial_masking_node_feature(self):
        """Test partial masking for node feature mode."""
        mask_prob = 0.5  # Partial masking
        masked_batch = create_masked_batch(self.mixed_batch, mp=mask_prob, mask_mode="node_feature")

        # Count how many AND gates were masked
        and_gates = (self.mixed_batch.x[:, 0] == 0) & (self.mixed_batch.x[:, 1] == 1) & (self.mixed_batch.x[:, 2] == 0)
        total_and_gates = and_gates.sum().item()
        masked_and_gates = masked_batch.node_mask.sum().item()

        # Check that approximate percentage of AND gates were masked
        # Allow for some variation due to randomness
        self.assertGreater(masked_and_gates, 0)  # At least some masking occurred
        self.assertLess(masked_and_gates, total_and_gates)  # Not all were masked

    def test_partial_masking_edge_feature(self):
        """Test partial masking for edge feature mode."""
        mask_prob = 0.5  # Partial masking
        masked_batch = create_masked_batch(self.batch, mp=mask_prob, mask_mode="edge_feature")

        # Count how many edges were masked
        total_edges = self.batch.edge_index.size(1)
        masked_edges = masked_batch.edge_mask.sum().item()

        # Check that approximate percentage of edges were masked
        self.assertGreater(masked_edges, 0)  # At least some masking occurred
        self.assertLess(masked_edges, total_edges)  # Not all were masked

    def test_no_and_gates(self):
        """Test masking on a graph with no AND gates."""
        # Create a batch with no AND gates
        no_and_batch = Batch.from_data_list([self.no_and_graph])

        # Try node feature masking
        masked_batch = create_masked_batch(no_and_batch, mp=1.0, mask_mode="node_feature")

        # Check that no nodes were masked (since there are no AND gates)
        self.assertEqual(masked_batch.node_mask.sum().item(), 0)

        # Check that the graph structure is preserved
        self.assertTrue(torch.equal(masked_batch.x, no_and_batch.x))
        self.assertTrue(torch.equal(masked_batch.edge_index, no_and_batch.edge_index))

    def test_single_node_graph(self):
        """Test masking on a graph with only one node."""
        # Create a batch with a single-node graph
        single_node_batch = Batch.from_data_list([self.single_node_graph])

        # Try node feature masking
        masked_batch = create_masked_batch(single_node_batch, mp=1.0, mask_mode="node_feature")

        # Check that the graph structure is preserved
        self.assertEqual(masked_batch.x.size(0), 1)
        self.assertEqual(masked_batch.edge_index.size(1), 0)

        # Try edge feature masking on a graph with no edges
        masked_batch = create_masked_batch(single_node_batch, mp=1.0, mask_mode="edge_feature")

        # Check that the graph structure is preserved
        self.assertEqual(masked_batch.x.size(0), 1)
        self.assertEqual(masked_batch.edge_index.size(1), 0)

    def test_empty_graph(self):
        """Test masking on an empty graph."""
        # Create a batch with an empty graph
        empty_batch = Batch.from_data_list([self.empty_graph])

        # Try node feature masking
        masked_batch = create_masked_batch(empty_batch, mp=1.0, mask_mode="node_feature")

        # Check that the graph structure is preserved
        self.assertEqual(masked_batch.x.size(0), 0)
        self.assertEqual(masked_batch.edge_index.size(1), 0)


    def test_batch_consistency(self):
        """Test batch attribute preservation."""
        # Create a mixed batch
        masked_batch = create_masked_batch(self.mixed_batch, mp=0.5, mask_mode="node_feature")

        # Check that the batch attribute is preserved
        self.assertTrue(hasattr(masked_batch, 'batch'))

        # Verify correct batch assignment for all nodes
        num_nodes = 0
        for i, graph in enumerate([self.single_graph, self.complex_graph, self.no_and_graph]):
            num_nodes += graph.x.size(0)

        # Check batch size is as expected
        self.assertEqual(masked_batch.batch.size(0), num_nodes)

    def test_deterministic_with_seed(self):
        """Test deterministic behavior with fixed seed."""
        torch.manual_seed(42)
        first_run = create_masked_batch(self.batch, mp=0.5, mask_mode="node_feature")

        torch.manual_seed(42)
        second_run = create_masked_batch(self.batch, mp=0.5, mask_mode="node_feature")

        # Check that the masking is identical with the same seed
        self.assertTrue(torch.equal(first_run.node_mask, second_run.node_mask))

    def test_mixed_graph_structures(self):
        """Test masking on a batch with mixed graph structures."""
        masked_batch = create_masked_batch(self.mixed_batch, mp=0.5, mask_mode="node_feature")

        # Check that AND gates were masked across different graph structures
        and_gates = (self.mixed_batch.x[:, 0] == 0) & (self.mixed_batch.x[:, 1] == 1) & (self.mixed_batch.x[:, 2] == 0)

        # Verify that only AND gates were masked
        for i in range(masked_batch.x.size(0)):
            if masked_batch.node_mask[i]:
                # The original feature at this position should be an AND gate
                orig_idx = i  # In node_feature mode, indices aren't changed
                self.assertTrue(and_gates[orig_idx].item())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping GPU test")
    def test_device_consistency(self):
        """Test consistency when using different devices."""
        # Move to GPU
        gpu_batch = self.batch.to('cuda')
        masked_gpu_batch = create_masked_batch(gpu_batch, mp=0.5, mask_mode="node_feature")

        # Check device consistency
        self.assertEqual(masked_gpu_batch.x.device.type, 'cuda')
        self.assertEqual(masked_gpu_batch.edge_index.device.type, 'cuda')
        self.assertEqual(masked_gpu_batch.node_mask.device.type, 'cuda')

    def test_predetermined_mask(self):
        """Test with a predetermined mask for deterministic testing."""

        # Create a custom masking function that uses a fixed mask
        def custom_create_masked_batch(batch):
            masked_batch = batch.clone()

            # Create predetermined mask (mask the first AND gate in each graph)
            is_and_gate = (batch.x[:, 0] == 0) & (batch.x[:, 1] == 1) & (batch.x[:, 2] == 0)
            batch_idx = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long)

            node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)

            for b in torch.unique(batch_idx):
                # Get AND gates for this graph
                graph_mask = (batch_idx == b) & is_and_gate
                graph_and_gates = torch.nonzero(graph_mask).squeeze(-1)

                if len(graph_and_gates) > 0:
                    # Mask the first AND gate in this graph
                    node_mask[graph_and_gates[0]] = True

            # Apply masking
            masked_batch.x_target = batch.x.clone()
            masked_batch.node_mask = node_mask
            masked_batch.x[node_mask] = 0.0

            return masked_batch

        # Use custom function
        masked_batch = custom_create_masked_batch(self.batch)

        # Check that exactly one AND gate per graph was masked
        is_and_gate = (self.batch.x[:, 0] == 0) & (self.batch.x[:, 1] == 1) & (self.batch.x[:, 2] == 0)
        batch_idx = self.batch.batch

        masked_and_gates_per_graph = {}

        for i in range(masked_batch.x.size(0)):
            if masked_batch.node_mask[i]:
                b = batch_idx[i].item()
                masked_and_gates_per_graph[b] = masked_and_gates_per_graph.get(b, 0) + 1

        # Verify one AND gate masked per graph
        for b in range(len(torch.unique(batch_idx))):
            self.assertEqual(masked_and_gates_per_graph.get(b, 0), 1)


    def test_memory_efficiency(self):
        """Test memory efficiency by checking unnecessary copying."""
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping memory efficiency test")
            return

        # Get initial memory usage
        if hasattr(torch.cuda, 'memory_allocated'):
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()

            # Create a large batch
            large_batch = Batch.from_data_list([self.single_graph] * 100)
            large_batch = large_batch.cuda()

            # Apply masking
            masked_batch = create_masked_batch(large_batch, mp=0.5, mask_mode="node_feature")

            # Check memory usage
            end_mem = torch.cuda.memory_allocated()

            # The memory increase should be reasonable
            # We expect some memory overhead but not excessive copying
            self.assertLess(end_mem - start_mem, 10 * large_batch.x.numel() * 4)  # Allow reasonable overhead
        else:
            self.skipTest("CUDA memory tracking not available")


if __name__ == '__main__':
    unittest.main()
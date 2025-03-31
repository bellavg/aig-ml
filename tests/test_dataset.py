import unittest
import torch
import networkx as nx
import pickle
import os
import random

# Import the AIGDataset class
from aig_dataset import AIGDataset  # Replace with actual import path


class TestAIGDataset(unittest.TestCase):
    def setUp(self):
        """
        Create a sample networkx graph for testing.
        This method runs before each test.
        """
        # Create a sample networkx graph
        self.sample_graph = nx.DiGraph()

        # Add nodes with type and feature information
        self.sample_graph.add_node(0, type=[1, 0, 0], feature=[0.5, 0.3])  # PI
        self.sample_graph.add_node(1, type=[0, 1, 0], feature=[0.7, 0.2])  # AND
        self.sample_graph.add_node(2, type=[0, 0, 1], feature=[0.1, 0.9])  # PO

        # Add edges
        self.sample_graph.add_edge(0, 1, type=1.0)  # Normal edge
        self.sample_graph.add_edge(1, 2, type=1.0)  # Normal edge

        # Add graph-level metadata
        self.sample_graph.graph['inputs'] = 1
        self.sample_graph.graph['outputs'] = 1

    def create_test_pickle(self, filename='test_graphs.pkl'):
        """
        Create a temporary pickle file with test graphs for dataset processing.

        Returns:
            str: Path to the created pickle file
        """
        test_graphs = [self.sample_graph] * 10  # Create multiple copies

        # Ensure the directory exists
        os.makedirs('test_data', exist_ok=True)
        filepath = os.path.join('test_data', filename)

        with open(filepath, 'wb') as f:
            pickle.dump(test_graphs, f)

        return filepath

    def test_dataset_initialization(self):
        """Test basic dataset initialization."""
        # Create test pickle file
        file_path = self.create_test_pickle()

        try:
            # Initialize dataset
            dataset = AIGDataset(
                file_path=file_path,
                mask_ratio=0.15,
                mask_mode="and_gates",
                num_graphs=5
            )

            # Check basic properties
            self.assertIsNotNone(dataset)
            self.assertEqual(len(dataset), 5)
        finally:
            # Clean up the test file
            os.remove(file_path)

    def test_masking_modes(self):
        """Test different masking modes."""
        # Create test pickle file
        file_path = self.create_test_pickle()

        try:
            # Test different masking modes
            mask_modes = ["random", "and_gates", "inputs"]

            for mode in mask_modes:
                dataset = AIGDataset(
                    file_path=file_path,
                    mask_ratio=0.15,
                    mask_mode=mode
                )

                # Check a few samples
                for data in dataset[:3]:
                    # Verify mask exists
                    self.assertTrue(hasattr(data, 'mask'), f"Mask not found for {mode} mode")

                    # Verify original features stored in y
                    self.assertTrue(hasattr(data, 'y'), f"Original features not stored for {mode} mode")

                    # Verify node types remain intact
                    original_types = data.y[:, :3]
                    current_types = data.x[:, :3]
                    self.assertTrue(torch.allclose(original_types, current_types),
                                    f"Node types changed in {mode} mode")
        finally:
            # Clean up the test file
            os.remove(file_path)

    def test_node_type_preservation(self):
        """Verify that node types are preserved during masking."""
        # Create test pickle file
        file_path = self.create_test_pickle()

        try:
            dataset = AIGDataset(
                file_path=file_path,
                mask_ratio=0.15,
                mask_mode="and_gates"
            )

            # Check a sample
            data = dataset[0]

            # Verify node types (first 3 columns) remain the same
            original_types = data.y[:, :3]
            current_types = data.x[:, :3]
            self.assertTrue(torch.allclose(original_types, current_types),
                            "Node types were modified during masking")
        finally:
            # Clean up the test file
            os.remove(file_path)

    def test_mask_ratio(self):
        """Test that the masking ratio is approximately correct."""
        # Create test pickle file
        file_path = self.create_test_pickle()

        try:
            # Test different mask ratios
            for ratio in [0.1, 0.15, 0.3]:
                dataset = AIGDataset(
                    file_path=file_path,
                    mask_ratio=ratio,
                    mask_mode="and_gates"
                )

                # Check a sample
                data = dataset[0]

                # Count masked nodes
                node_types = torch.argmax(data.x[:, :3], dim=1)
                and_gates = torch.where(node_types == 1)[0]

                # Count masked AND gates
                masked_and_gates = and_gates[data.mask[and_gates]]

                # Check if the number of masked nodes is close to the expected ratio
                expected_masked = max(1, int(len(and_gates) * ratio))
                self.assertLessEqual(len(masked_and_gates), expected_masked,
                                     f"Too many nodes masked for ratio {ratio}")
        finally:
            # Clean up the test file
            os.remove(file_path)

    def test_data_attributes(self):
        """Test that the generated data has expected attributes."""
        # Create test pickle file
        file_path = self.create_test_pickle()

        try:
            dataset = AIGDataset(
                file_path=file_path,
                mask_ratio=0.15,
                mask_mode="and_gates"
            )

            # Check a sample
            data = dataset[0]

            # Verify required attributes
            required_attrs = ['x', 'edge_index', 'edge_attr', 'y', 'mask']
            for attr in required_attrs:
                self.assertTrue(hasattr(data, attr), f"Missing {attr} attribute")

            # Verify attribute types and shapes
            self.assertIsInstance(data.x, torch.Tensor)
            self.assertIsInstance(data.edge_index, torch.Tensor)
            self.assertIsInstance(data.edge_attr, torch.Tensor)
            self.assertIsInstance(data.y, torch.Tensor)
            self.assertIsInstance(data.mask, torch.Tensor)
        finally:
            # Clean up the test file
            os.remove(file_path)

    def test_invalid_mask_mode(self):
        """Test that an invalid mask mode raises an error."""
        # Create test pickle file
        file_path = self.create_test_pickle()

        try:
            with self.assertRaises(ValueError):
                AIGDataset(
                    file_path=file_path,
                    mask_ratio=0.15,
                    mask_mode="invalid_mode"
                )
        finally:
            # Clean up the test file
            os.remove(file_path)

    def test_num_graphs_limit(self):
        """Test that num_graphs parameter limits the dataset size."""
        # Create test pickle file with multiple graphs
        file_path = self.create_test_pickle()

        try:
            # Test different graph limits
            for limit in [1, 3, 5]:
                dataset = AIGDataset(
                    file_path=file_path,
                    mask_ratio=0.15,
                    mask_mode="and_gates",
                    num_graphs=limit
                )

                # Check dataset length
                self.assertEqual(len(dataset), limit)
        finally:
            # Clean up the test file
            os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
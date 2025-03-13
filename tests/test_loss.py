import unittest
import torch
import numpy as np
import sys
import os
from torch_geometric.data import Data, Batch

# Import your loss computation functions - adjust the path as needed
sys.path.append(os.path.abspath('..'))
from loss import (
    compute_loss,
    compute_node_feature_loss,
    compute_edge_feature_loss,
    compute_connectivity_loss,
    finalize_loss
)
from masking import create_masked_batch


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test data for all test cases."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Create a simple graph with node features
        # Node types: [1,0,0] = PI, [0,1,0] = AND, [0,0,1] = PO
        # Fourth index (idx 3) is the truth table value
        self.graph = Data(
            x=torch.tensor([
                [1, 0, 0, 0.5],  # PI
                [1, 0, 0, 0.2],  # PI
                [0, 1, 0, 0.7],  # AND
                [0, 1, 0, 0.9],  # AND
                [0, 0, 1, 0.3],  # PO
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

        # Create a batch for testing
        self.batch = Batch.from_data_list([self.graph, self.graph])

        # Create masked batches for different masking modes
        self.node_masked_batch = create_masked_batch(
            self.batch, mp=1.0, mask_mode="node_feature"
        )

        self.edge_masked_batch = create_masked_batch(
            self.batch, mp=0.5, mask_mode="edge_feature"
        )

        self.connectivity_masked_batch = create_masked_batch(
            self.batch, mp=0.5, mask_mode="connectivity"
        )

    def test_finalize_loss(self):
        """Test that finalize_loss correctly handles zero and non-zero losses."""
        # Test with non-zero loss
        loss = torch.tensor(2.5)
        loss_dict = {"some_loss": loss}
        predictions = {"dummy": torch.tensor([0.0])}

        final_loss, final_losses = finalize_loss(loss, loss_dict, predictions)

        self.assertEqual(final_loss, loss)
        self.assertEqual(final_losses["total_loss"], loss)
        self.assertEqual(final_losses["some_loss"], loss)

        # Test with zero loss (should add a small dummy loss)
        zero_loss = torch.tensor(0.0)
        zero_loss_dict = {}

        final_loss, final_losses = finalize_loss(zero_loss, zero_loss_dict, predictions)

        self.assertGreater(final_loss, 0)
        self.assertTrue("dummy_loss" in final_losses)
        self.assertEqual(final_losses["total_loss"], final_loss)

    def test_compute_loss_dispatching(self):
        """Test that compute_loss correctly dispatches to the appropriate loss function."""
        # Create sample predictions (we're just testing dispatching, not values)
        node_preds = {
            "node_features": self.node_masked_batch.x_target.clone()
        }

        edge_preds = {
            "edge_preds": {
                "edge_features": self.edge_masked_batch.edge_attr_target[self.edge_masked_batch.edge_mask].clone()
            }
        }

        connectivity_preds = {
            "edge_preds": {
                "edge_existence": torch.ones(self.connectivity_masked_batch.all_candidate_targets.size()),
                "edge_features": torch.zeros(
                    self.connectivity_masked_batch.all_candidate_targets.size(0),
                    self.connectivity_masked_batch.edge_attr_target.size(1)
                )
            }
        }

        # Test node feature masking mode
        loss, losses = compute_loss(node_preds, self.node_masked_batch)
        self.assertIn("truth_table_loss", losses)

        # Test edge feature masking mode
        loss, losses = compute_loss(edge_preds, self.edge_masked_batch)
        self.assertIn("edge_feature_loss", losses)

        # Test connectivity masking mode
        loss, losses = compute_loss(connectivity_preds, self.connectivity_masked_batch)
        self.assertIn("edge_existence_loss", losses)

        # Test invalid masking mode
        invalid_batch = self.node_masked_batch.clone()
        invalid_batch.mask_mode = "invalid_mode"
        with self.assertRaises(ValueError):
            compute_loss(node_preds, invalid_batch)

    def test_node_feature_loss_relative_values(self):
        """Test that node feature loss values are relatively correct."""
        # Make predictions with different levels of error
        perfect_preds = {
            "node_features": self.node_masked_batch.x_target.clone()
        }

        large_error_preds = {
            "node_features": self.node_masked_batch.x_target.clone()
        }

        # Create large errors only for truth table values
        for i in range(self.node_masked_batch.x.size(0)):
            if self.node_masked_batch.node_mask[i]:
                tt_idx = getattr(self.node_masked_batch, "truth_table_idx", 3)
                target_val = self.node_masked_batch.x_target[i, tt_idx]
                # Use larger shifts that will definitely increase loss
                large_error_preds["node_features"][i, tt_idx] = 10.0 if target_val < 0.5 else -10.0

        # Compute losses
        _, perfect_losses = compute_node_feature_loss(perfect_preds, self.node_masked_batch)
        _, large_losses = compute_node_feature_loss(large_error_preds, self.node_masked_batch)

        # The losses should increase with large error magnitude
        perfect_loss = perfect_losses["truth_table_loss"].item()
        large_loss = large_losses["truth_table_loss"].item()

        self.assertLess(perfect_loss, large_loss)
        self.assertGreater(large_loss, perfect_loss + 1.0)  # Large error should give substantial increase

    def test_node_feature_loss_truth_table_focus(self):
        """Test that node feature loss only focuses on the truth table value."""
        # Create a batch where we know which nodes are masked
        masked_batch = self.node_masked_batch.clone()

        # Verify that only AND gates are masked
        is_and_gate = (masked_batch.x_target[:, 0] == 0) & (masked_batch.x_target[:, 1] == 1) & (
                    masked_batch.x_target[:, 2] == 0)
        masked_and_gates = masked_batch.node_mask & is_and_gate
        self.assertTrue(torch.all(masked_and_gates == masked_batch.node_mask))

        # Create predictions that are perfect
        perfect_preds = {
            "node_features": masked_batch.x_target.clone()
        }

        # Compute baseline loss
        _, perfect_losses = compute_node_feature_loss(perfect_preds, masked_batch)
        perfect_loss = perfect_losses["truth_table_loss"].item()

        # Create predictions with errors only in node type
        node_type_error_preds = {
            "node_features": masked_batch.x_target.clone()
        }

        # For masked nodes, change node type (indices 0,1,2) but keep truth table value (index 3)
        for i in range(masked_batch.x.size(0)):
            if masked_batch.node_mask[i]:
                # Set node type to nonsensical values
                node_type_error_preds["node_features"][i, :3] = torch.tensor([0.9, 0.9, 0.9])

        _, node_type_error_losses = compute_node_feature_loss(node_type_error_preds, masked_batch)
        node_type_error_loss = node_type_error_losses["truth_table_loss"].item()

        # Loss should be the same as perfect since node type doesn't matter
        self.assertAlmostEqual(perfect_loss, node_type_error_loss, places=5)

        # Now create predictions with errors only in truth table
        tt_error_preds = {
            "node_features": masked_batch.x_target.clone()
        }

        # For masked nodes, keep node type correct but make truth table value wrong
        for i in range(masked_batch.x.size(0)):
            if masked_batch.node_mask[i]:
                tt_idx = getattr(masked_batch, "truth_table_idx", 3)
                tt_error_preds["node_features"][i, tt_idx] = -5.0  # Very different from target

        _, tt_error_losses = compute_node_feature_loss(tt_error_preds, masked_batch)
        tt_error_loss = tt_error_losses["truth_table_loss"].item()

        # Loss should be much higher with truth table errors
        self.assertGreater(tt_error_loss, perfect_loss + 0.5)

    def test_node_feature_loss_masked_nodes_only(self):
        """Test that node feature loss is computed only for masked nodes."""
        # Create a batch with only some AND gates masked
        partial_masked_batch = create_masked_batch(
            self.batch, mp=0.5, mask_mode="node_feature"
        )

        # Skip test if no nodes were masked
        if partial_masked_batch.node_mask.sum() == 0:
            self.skipTest("No nodes were masked, cannot test masked vs unmasked behavior")

        # Find unmasked AND gates
        is_and_gate = (partial_masked_batch.x_target[:, 0] == 0) & (partial_masked_batch.x_target[:, 1] == 1) & (
                    partial_masked_batch.x_target[:, 2] == 0)
        unmasked_and_gates = is_and_gate & ~partial_masked_batch.node_mask

        # Skip test if all AND gates were masked
        if unmasked_and_gates.sum() == 0:
            self.skipTest("All AND gates were masked, cannot test masked vs unmasked behavior")

        # Create predictions that are perfect
        perfect_preds = {
            "node_features": partial_masked_batch.x_target.clone()
        }

        # Compute baseline loss
        _, perfect_losses = compute_node_feature_loss(perfect_preds, partial_masked_batch)
        perfect_loss = perfect_losses["truth_table_loss"].item()

        # Create predictions with errors only in unmasked AND gates
        unmasked_error_preds = {
            "node_features": partial_masked_batch.x_target.clone()
        }

        for i in range(partial_masked_batch.x.size(0)):
            if unmasked_and_gates[i]:
                tt_idx = getattr(partial_masked_batch, "truth_table_idx", 3)
                unmasked_error_preds["node_features"][i, tt_idx] = -5.0  # Very different

        _, unmasked_error_losses = compute_node_feature_loss(unmasked_error_preds, partial_masked_batch)
        unmasked_error_loss = unmasked_error_losses["truth_table_loss"].item()

        # Loss should be the same as perfect since unmasked nodes don't matter
        self.assertAlmostEqual(perfect_loss, unmasked_error_loss, places=5)

        # Now create predictions with errors in masked nodes
        masked_error_preds = {
            "node_features": partial_masked_batch.x_target.clone()
        }

        for i in range(partial_masked_batch.x.size(0)):
            if partial_masked_batch.node_mask[i]:
                tt_idx = getattr(partial_masked_batch, "truth_table_idx", 3)
                masked_error_preds["node_features"][i, tt_idx] = -5.0  # Very different

        _, masked_error_losses = compute_node_feature_loss(masked_error_preds, partial_masked_batch)
        masked_error_loss = masked_error_losses["truth_table_loss"].item()

        # Loss should be higher with masked node errors
        self.assertGreater(masked_error_loss, perfect_loss + 0.5)

    def test_edge_feature_loss_relative_values(self):
        """Test that edge feature loss values are relatively correct."""
        # Skip test if no edges are masked
        if self.edge_masked_batch.edge_mask.sum() == 0:
            self.skipTest("No edges were masked, cannot test edge feature loss")

        # Make predictions with different levels of error
        perfect_preds = {
            "edge_preds": {
                "edge_features": self.edge_masked_batch.edge_attr_target[self.edge_masked_batch.edge_mask].clone()
            }
        }

        slight_error_preds = {
            "edge_preds": {
                "edge_features": self.edge_masked_batch.edge_attr_target[self.edge_masked_batch.edge_mask].clone() + 0.2
            }
        }

        large_error_preds = {
            "edge_preds": {
                "edge_features": torch.ones_like(
                    self.edge_masked_batch.edge_attr_target[self.edge_masked_batch.edge_mask]) * 10.0
            }
        }

        # Compute losses
        _, perfect_losses = compute_edge_feature_loss(perfect_preds, self.edge_masked_batch)
        _, slight_losses = compute_edge_feature_loss(slight_error_preds, self.edge_masked_batch)
        _, large_losses = compute_edge_feature_loss(large_error_preds, self.edge_masked_batch)

        # The losses should increase with error magnitude
        perfect_loss = perfect_losses["edge_feature_loss"].item()
        slight_loss = slight_losses["edge_feature_loss"].item()
        large_loss = large_losses["edge_feature_loss"].item()

        self.assertLess(perfect_loss, slight_loss)
        self.assertLess(slight_loss, large_loss)

    def test_edge_feature_loss_masked_edges_only(self):
        """Test that edge feature loss is computed only for masked edges."""
        # Create a batch with some edges masked
        edge_masked_batch = self.edge_masked_batch

        # Skip test if no edges are masked
        if edge_masked_batch.edge_mask.sum() == 0:
            self.skipTest("No edges were masked, cannot test masked vs unmasked behavior")

        # Create predictions that match targets for masked edges
        perfect_preds = {
            "edge_preds": {
                "edge_features": edge_masked_batch.edge_attr_target[edge_masked_batch.edge_mask].clone()
            }
        }

        # Create predictions with large errors
        error_preds = {
            "edge_preds": {
                "edge_features": torch.ones_like(edge_masked_batch.edge_attr_target[edge_masked_batch.edge_mask]) * 10.0
            }
        }

        # Compute losses
        _, perfect_losses = compute_edge_feature_loss(perfect_preds, edge_masked_batch)
        _, error_losses = compute_edge_feature_loss(error_preds, edge_masked_batch)

        perfect_loss = perfect_losses["edge_feature_loss"].item()
        error_loss = error_losses["edge_feature_loss"].item()

        # Loss should be higher with errors
        self.assertLess(perfect_loss, error_loss)

    def test_connectivity_loss_existence(self):
        """Test that connectivity loss correctly handles edge existence prediction."""
        # Skip test if needed attributes don't exist
        if not hasattr(self.connectivity_masked_batch, 'all_candidate_targets'):
            self.skipTest("Missing all_candidate_targets attribute")

        conn_masked_batch = self.connectivity_masked_batch

        # Create predictions where edge existence is perfect
        perfect_preds = {
            "edge_preds": {
                "edge_existence": conn_masked_batch.all_candidate_targets.clone().unsqueeze(1),
                "edge_features": torch.zeros(
                    conn_masked_batch.all_candidate_targets.size(0),
                    conn_masked_batch.edge_attr_target.size(1)
                )
            }
        }

        # Create predictions where edge existence is flipped
        flipped_preds = {
            "edge_preds": {
                "edge_existence": (1.0 - conn_masked_batch.all_candidate_targets).clone().unsqueeze(1) * 10.0,
                "edge_features": torch.zeros(
                    conn_masked_batch.all_candidate_targets.size(0),
                    conn_masked_batch.edge_attr_target.size(1)
                )
            }
        }

        # Compute losses
        _, perfect_losses = compute_connectivity_loss(perfect_preds, conn_masked_batch)
        _, flipped_losses = compute_connectivity_loss(flipped_preds, conn_masked_batch)

        perfect_loss = perfect_losses["edge_existence_loss"].item()
        flipped_loss = flipped_losses["edge_existence_loss"].item()

        # Loss should be higher for flipped predictions
        self.assertLess(perfect_loss, flipped_loss)

    def test_connectivity_loss_with_edge_features(self):
        """Test connectivity loss with edge features."""
        conn_masked_batch = self.connectivity_masked_batch

        # Skip test if no masked edge attributes
        if not hasattr(conn_masked_batch, 'masked_edge_attr_target') or conn_masked_batch.masked_edge_attr_target.size(
                0) == 0:
            self.skipTest("No masked edge attributes available")

        # Verify we have positive examples (edges that should exist)
        positive_mask = conn_masked_batch.all_candidate_targets > 0.5
        if positive_mask.sum() == 0:
            self.skipTest("No positive edge examples")

        # Create predictions with good edge existence but bad features
        mixed_preds = {
            "edge_preds": {
                "edge_existence": conn_masked_batch.all_candidate_targets.clone().unsqueeze(1),
                "edge_features": torch.ones(
                    conn_masked_batch.all_candidate_targets.size(0),
                    conn_masked_batch.edge_attr_target.size(1)
                ) * 10.0  # Very different from targets
            }
        }

        # Compute losses
        _, mixed_losses = compute_connectivity_loss(mixed_preds, conn_masked_batch)

        # Both existence and feature losses should be present
        self.assertIn("edge_existence_loss", mixed_losses)
        self.assertIn("edge_feature_loss", mixed_losses)

        # Feature loss should be significant
        self.assertGreater(mixed_losses["edge_feature_loss"].item(), 0.1)


if __name__ == "__main__":
    unittest.main()
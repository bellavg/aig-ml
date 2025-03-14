import unittest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from collections import namedtuple
import sys
import os
import numpy as np

# Import your modules - adjust paths as needed
from masking import (
    create_masked_batch,
    _apply_edge_feature_masking,
    _create_edge_masks,
    _identify_and_gates
)
from loss import compute_loss, compute_edge_feature_loss
from model import AIGTransformer  # Adjust import for your model class
from prediction import reconstruct_predictions


class TestEdgeFeatureMasking(unittest.TestCase):
    """Test suite for edge feature masking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple synthetic graph for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create node features: [input_nodes, and_gates, output_nodes]
        # Format: [one-hot node type (3), truth table value (1)]
        x = torch.tensor([
            [1, 0, 0, 0],  # Input 1
            [1, 0, 0, 0],  # Input 2
            [0, 1, 0, 1],  # AND gate 1 (truth table = 1)
            [0, 1, 0, 0],  # AND gate 2 (truth table = 0)
            [0, 0, 1, 0],  # Output
        ], dtype=torch.float)

        # Edge structure: [source_nodes, target_nodes]
        edge_index = torch.tensor([
            [0, 1, 0, 1, 2, 3],  # Source nodes
            [2, 2, 3, 3, 4, 4],  # Target nodes
        ], dtype=torch.long)

        # Edge attributes: [INV, REG] one-hot encoded
        edge_attr = torch.tensor([
            [0, 1],  # Regular edge (Input 1 -> AND gate 1)
            [0, 1],  # Regular edge (Input 2 -> AND gate 1)
            [1, 0],  # Inverted edge (Input 1 -> AND gate 2)
            [0, 1],  # Regular edge (Input 2 -> AND gate 2)
            [0, 1],  # Regular edge (AND gate 1 -> Output)
            [1, 0],  # Inverted edge (AND gate 2 -> Output)
        ], dtype=torch.float)

        # Create a simple AIG data object
        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.data = self.data.to(self.device)

        # Create a batch with a single graph
        self.batch = Batch.from_data_list([self.data])

        # Create a larger batch with multiple graphs
        self.multi_batch = Batch.from_data_list([self.data, self.data])

        # Define model parameters
        self.model_params = {
            'node_features': 4,
            'edge_features': 2,
            'hidden_dim': 64,  # Smaller for testing
            'num_layers': 2,  # Fewer layers for speed
            'num_heads': 2,  # Fewer heads for testing
            'dropout': 0.1,
            'max_nodes': 10,
            'max_hop': 3
        }

        # Define mock args for training/validation
        Args = namedtuple('Args', ['mask_mode', 'mask_prob'])
        self.args = Args(mask_mode="edge_feature", mask_prob=0.5)

    def test_masking_creation(self):
        """Test if edge feature masking correctly masks edge attributes."""
        masked_batch = create_masked_batch(
            self.batch,
            mp=0.5,
            mask_mode="edge_feature",
            mask_value=-1.0
        )

        # Verify batch properties
        self.assertEqual(masked_batch.mask_mode, "edge_feature")
        self.assertTrue(hasattr(masked_batch, 'edge_mask'))
        self.assertTrue(hasattr(masked_batch, 'original_edge_attr'))
        self.assertTrue(hasattr(masked_batch, 'edge_mask_value'))
        self.assertEqual(masked_batch.edge_mask_value, -1.0)

        # Check if edges are preserved (edge_index shouldn't change)
        self.assertTrue(torch.equal(masked_batch.edge_index, self.batch.edge_index))

        # Check that some edges have been masked
        masked_count = masked_batch.edge_mask.sum().item()
        self.assertGreater(masked_count, 0)
        self.assertLessEqual(masked_count, len(masked_batch.edge_attr))

        # Check if masked edge attributes have the mask value
        for i, is_masked in enumerate(masked_batch.edge_mask):
            if is_masked:
                self.assertTrue(torch.all(masked_batch.edge_attr[i] == -1.0))

        # Original values should be stored correctly
        self.assertEqual(masked_batch.original_edge_attr.shape[0], masked_count)

        print(f"✅ Edge feature masking applied to {masked_count} out of {len(masked_batch.edge_attr)} edges")

    def test_multi_graph_masking(self):
        """Test edge feature masking on multiple graphs in a batch."""
        masked_multi_batch = create_masked_batch(
            self.multi_batch,
            mp=0.5,
            mask_mode="edge_feature",
            mask_value=-1.0
        )

        # Check masking was applied
        masked_count = masked_multi_batch.edge_mask.sum().item()
        self.assertGreater(masked_count, 0)

        # Check that original values are preserved for masked edges
        self.assertEqual(masked_multi_batch.original_edge_attr.shape[0], masked_count)

        print(f"✅ Edge feature masking applied correctly to a batch with multiple graphs")

    def test_edge_mask_creation(self):
        """Test the edge mask creation function specifically."""
        edge_mask = _create_edge_masks(self.batch, mp=0.5)

        # Check basic properties
        self.assertEqual(edge_mask.shape, torch.Size([self.batch.edge_index.shape[1]]))
        self.assertGreater(edge_mask.sum().item(), 0)

        print(f"✅ Edge mask creation works correctly")

    def test_model_forward_pass(self):
        """Test model forward pass with edge feature masking."""
        # Initialize model
        model = AIGTransformer(**self.model_params).to(self.device)

        # Create masked batch
        masked_batch = create_masked_batch(
            self.batch,
            mp=0.5,
            mask_mode="edge_feature",
            mask_value=-1.0
        )

        # Forward pass
        with torch.no_grad():
            results = model(masked_batch)

        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertTrue(
            'edge_attr_pred' in results or ('edge_preds' in results and 'edge_features' in results['edge_preds']))

        # If direct output format
        if 'edge_attr_pred' in results:
            preds = results['edge_attr_pred']
            self.assertEqual(preds.shape[1], 2)  # [INV, REG] predictions
            self.assertGreater(preds.shape[0], 0)  # At least one prediction

        # If nested dictionary format
        elif 'edge_preds' in results and 'edge_features' in results['edge_preds']:
            preds = results['edge_preds']['edge_features']
            self.assertEqual(preds.shape[1], 2)  # [INV, REG] predictions
            self.assertGreater(preds.shape[0], 0)  # At least one prediction

        print(f"✅ Model forward pass works with edge feature masking")

    def test_loss_computation(self):
        """Test loss computation for edge feature masking."""
        # Create masked batch
        masked_batch = create_masked_batch(
            self.batch,
            mp=0.5,
            mask_mode="edge_feature",
            mask_value=-1.0
        )

        # Create mock predictions
        masked_indices = torch.nonzero(masked_batch.edge_mask).squeeze(-1)
        num_masked = masked_indices.size(0)

        # Create realistic predictions (logits that when sigmoid'ed give values between 0 and 1)
        mock_logits = torch.randn(num_masked, 2, device=self.device)

        # Create mock prediction dictionary
        predictions = {
            'edge_attr_pred': mock_logits,
            'masked_edge_indices': masked_indices
        }

        # Create targets dictionary
        targets = {
            'mask_mode': 'edge_feature',
            'edge_mask': masked_batch.edge_mask,
            'original_edge_attr': masked_batch.original_edge_attr,
            'edge_attr_target': masked_batch.edge_attr_target
        }

        # Compute loss
        loss, loss_dict = compute_loss(predictions, targets)

        # Check loss values
        self.assertGreater(loss.item(), 0)
        self.assertIn('edge_feature_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)

        # Should also have accuracy metrics
        self.assertIn('edge_feat_accuracy', loss_dict)
        self.assertIn('inv_edge_accuracy', loss_dict)
        self.assertIn('reg_edge_accuracy', loss_dict)

        print(f"✅ Loss computation works correctly for edge feature masking")

    def test_direct_loss_function(self):
        """Test the direct edge feature loss function."""
        # Create mock data
        num_samples = 4
        mock_preds = torch.tensor([
            [2.0, -2.0],  # Strong prediction for INV edge
            [-2.0, 2.0],  # Strong prediction for REG edge
            [0.0, 0.0],  # Uncertain
            [1.0, 1.0],  # Confused prediction
        ], device=self.device)

        mock_targets = torch.tensor([
            [1.0, 0.0],  # INV edge
            [0.0, 1.0],  # REG edge
            [1.0, 0.0],  # INV edge
            [0.0, 1.0],  # REG edge
        ], device=self.device)

        predictions = {'edge_attr_pred': mock_preds}

        # The issue is that the dimensions must match properly
        # We're creating a "full" edge_attr_target and a matching edge_mask
        full_edge_attr = torch.zeros((num_samples, 2),
                                     device=self.device)  # Create a tensor of same size as predictions
        edge_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        # Only use original_edge_attr and don't rely on indexing edge_attr_target
        targets = {
            'mask_mode': 'edge_feature',
            'original_edge_attr': mock_targets,
            'edge_mask': edge_mask,
            'edge_attr_target': full_edge_attr
        }

        # Compute loss
        loss, loss_dict = compute_edge_feature_loss(predictions, targets)

        # Check loss and accuracy values
        self.assertGreater(loss.item(), 0)
        # Expected values based on our mock data
        self.assertGreater(loss_dict['edge_feat_accuracy'].item(), 0)
        # 2 correct predictions out of 4: index 0 predicts INV (1,0), index 1 predicts REG (0,1)
        self.assertAlmostEqual(loss_dict['edge_feat_accuracy'].item(), 0.5, delta=0.1)

        print(f"✅ Direct edge feature loss function works correctly")

    def test_prediction_reconstruction(self):
        """Test reconstruction of predictions for evaluation."""
        # Create masked batch
        masked_batch = create_masked_batch(
            self.batch,
            mp=0.5,
            mask_mode="edge_feature",
            mask_value=-1.0
        )

        # Create mock predictions
        masked_indices = torch.nonzero(masked_batch.edge_mask).squeeze(-1)
        num_masked = masked_indices.size(0)

        # Mock predictions in both possible formats
        # 1. Direct output format
        direct_preds = {
            'edge_attr_pred': torch.randn(num_masked, 2, device=self.device),
            'masked_edge_indices': masked_indices
        }

        # 2. Nested dictionary format
        nested_preds = {
            'edge_preds': {
                'edge_features': torch.randn(num_masked, 2, device=self.device),
                'edge_indices': masked_indices,
                'masked_edges': masked_batch.edge_index[:, masked_indices]
            }
        }

        # Create targets
        targets = {
            'mask_mode': 'edge_feature',
            'edge_mask': masked_batch.edge_mask,
            'edge_attr_target': masked_batch.edge_attr_target,
            'original_edge_attr': masked_batch.original_edge_attr
        }

        # Test both prediction formats
        for preds, name in [(direct_preds, "direct"), (nested_preds, "nested")]:
            full_pred = reconstruct_predictions(preds, targets)

            # Check if reconstruction was successful
            if name == "direct":
                self.assertIn('edge_attr_pred', full_pred)
            else:
                self.assertIn('edge_attr_pred', full_pred)

            # Should also reconstruct full edge features
            self.assertIn('full_edge_features', full_pred)
            self.assertEqual(full_pred['full_edge_features'].shape, masked_batch.edge_attr_target.shape)

            print(f"✅ Prediction reconstruction works for {name} format")

    def test_end_to_end_training(self):
        """Test one training step end-to-end."""
        # Initialize model
        model = AIGTransformer(**self.model_params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create masked batch
        masked_batch = create_masked_batch(
            self.batch,
            mp=0.5,
            mask_mode="edge_feature",
            mask_value=-1.0
        )

        # Prepare target dictionary
        targets = {
            'x_target': masked_batch.x_target,
            'edge_index_target': masked_batch.edge_index_target,
            'edge_attr_target': masked_batch.edge_attr_target,
            'mask_mode': 'edge_feature',
            'edge_mask': masked_batch.edge_mask
        }

        if hasattr(masked_batch, 'original_edge_attr'):
            targets['original_edge_attr'] = masked_batch.original_edge_attr

        if hasattr(masked_batch, 'edge_mask_value'):
            targets['edge_mask_value'] = masked_batch.edge_mask_value

        # Forward pass
        model.train()
        predictions = model(masked_batch)

        # Compute loss
        loss, loss_dict = compute_loss(predictions, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Check if gradients are flowing
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                has_grad = True
                break

        self.assertTrue(has_grad, "No gradients are flowing in the model")

        # Parameter update
        optimizer.step()

        print(f"✅ End-to-end training step works correctly")
        print(f"  Loss: {loss.item():.4f}")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.4f}")
            else:
                print(f"  {k}: {v:.4f}")


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions related to edge masking."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create node features
        x = torch.tensor([
            [1, 0, 0, 0],  # Input 1
            [0, 1, 0, 1],  # AND gate (truth table = 1)
            [0, 0, 1, 0],  # Output
        ], dtype=torch.float)

        edge_index = torch.tensor([
            [0, 1],  # Source nodes
            [1, 2],  # Target nodes
        ], dtype=torch.long)

        edge_attr = torch.tensor([
            [0, 1],  # Regular edge
            [1, 0],  # Inverted edge
        ], dtype=torch.float)

        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.data = self.data.to(self.device)

    def test_identify_and_gates(self):
        """Test AND gate identification."""
        is_and_gate = _identify_and_gates(self.data)
        expected = torch.tensor([False, True, False], device=self.device)
        self.assertTrue(torch.equal(is_and_gate, expected))

        print(f"✅ AND gate identification works correctly")


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestEdgeFeatureMasking))
    suite.addTest(unittest.makeSuite(TestUtilityFunctions))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()

    # Summarize results
    print("\n=== TEST SUMMARY ===")
    print(f"Total tests: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    # Exit with appropriate code
    sys.exit(len(result.failures) + len(result.errors))
import torch
import torch.nn as nn
import unittest
from loss import *

class TestTruthTableFeatureLoss(unittest.TestCase):
    def setUp(self):
        # Initialize the loss function
        self.loss_fn = TruthTableFeatureLoss()

        # Set random seed for reproducibility
        torch.manual_seed(42)

    def test_basic_loss_computation(self):
        """Test basic loss computation with different scenarios."""
        # Perfect prediction
        pred_features = torch.tensor([[0.5, 1.0, 0.0]])
        true_features = torch.tensor([[0.5, 1.0, 0.0]])
        loss = self.loss_fn(pred_features, true_features)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

        # Moderate difference
        pred_features = torch.tensor([[0.6, 0.9, 0.1]])
        true_features = torch.tensor([[0.5, 1.0, 0.0]])
        loss = self.loss_fn(pred_features, true_features)
        self.assertGreater(loss.item(), 0.0)

    def test_padding_handling(self):
        """Test handling of padding values."""
        # Scenario with padding values
        pred_features = torch.tensor([[0.5, 1.0, 0.0], [0.6, -1.0, 0.1]])
        true_features = torch.tensor([[0.5, 1.0, 0.0], [0.6, -1.0, 0.1]])

        # Default behavior should ignore -1 values
        loss = self.loss_fn(pred_features, true_features)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_binary_truth_table_values(self):
        """Test loss computation for binary truth table values."""
        # Predictions very close to 0 and 1
        pred_features = torch.tensor([[0.1, 0.9, 0.02]])
        true_features = torch.tensor([[0.0, 1.0, 0.0]])
        loss = self.loss_fn(pred_features, true_features)
        self.assertGreater(loss.item(), 0.0)

    def test_feature_normalization(self):
        """Test feature normalization option."""
        # Create a loss function with normalization
        norm_loss_fn = TruthTableFeatureLoss(feature_normalization=True)

        pred_features = torch.tensor([[10.0, 20.0, 30.0]])
        true_features = torch.tensor([[11.0, 21.0, 31.0]])

        loss = norm_loss_fn(pred_features, true_features)
        self.assertGreater(loss.item(), 0.0)

    def test_metrics(self):
        """Test feature metrics computation."""
        # Predictions and true values
        pred_features = torch.tensor([[0.1, 0.9, 0.02]])
        true_features = torch.tensor([[0.0, 1.0, 0.0]])

        # Compute metrics
        metrics = FeatureMetrics.compute_metrics(pred_features, true_features)

        # Check key metrics exist
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rel_l2', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('binary_accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

    def test_truth_table_accuracy(self):
        """Test truth table accuracy computation."""
        # Predictions and true values
        pred_features = torch.tensor([[0.1, 0.9, 0.02]])
        true_features = torch.tensor([[0.0, 1.0, 0.0]])

        # Compute accuracy
        accuracy_metrics = FeatureMetrics.compute_truth_table_accuracy(pred_features, true_features)

        # Check key metrics exist
        self.assertIn('truth_table_accuracy', accuracy_metrics)
        self.assertIn('correct_bits', accuracy_metrics)
        self.assertIn('total_bits', accuracy_metrics)


if __name__ == '__main__':
    unittest.main()
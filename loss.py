import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union

class TruthTableFeatureLoss(nn.Module):
    """
    Loss function for predicting truth table features of nodes in AIGs.
    Handles padding values (-1) in the truth tables and supports different weighting schemes.
    """

    def __init__(
            self,
            l1_weight: float = 0.1,
            feature_normalization: bool = False,
            ignore_padding: bool = True,
            binary_loss_weight: float = 2.0
    ):
        """
        Initialize the truth table feature loss.

        Args:
            l1_weight: Weight for L1 regularization (default: 0.1)
            feature_normalization: Whether to normalize features before computing loss (default: False)
            ignore_padding: Whether to ignore padded values (-1) in loss computation (default: True)
            binary_loss_weight: Extra weight for binary classification loss for 0/1 values (default: 2.0)
        """
        super(TruthTableFeatureLoss, self).__init__()
        self.l1_weight = l1_weight
        self.feature_normalization = feature_normalization
        self.ignore_padding = ignore_padding
        self.binary_loss_weight = binary_loss_weight

    def forward(
            self,
            pred_features: torch.Tensor,
            true_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss between predicted and true node features.

        Args:
            pred_features: Predicted node features [num_masked_nodes, feature_dim]
            true_features: Target node features [num_masked_nodes, feature_dim]

        Returns:
            Loss value
        """
        # Check if pred_features and true_features are exactly the same
        if torch.allclose(pred_features, true_features, atol=1e-8):
            return torch.tensor(0.0, device=pred_features.device)

        # Create mask for padding values and non-padded valid values
        if self.ignore_padding:
            # For handling both Truth table values and padding
            valid_mask = ~((true_features == -1) | torch.isnan(true_features))
        else:
            valid_mask = torch.ones_like(true_features, dtype=torch.bool)

        # If no valid values, return zero loss
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_features.device)

        # Extract valid values
        pred_valid = pred_features[valid_mask]
        true_valid = true_features[valid_mask]

        # Normalize features if requested
        if self.feature_normalization:
            pred_valid = self._normalize_features(pred_valid)
            true_valid = self._normalize_features(true_valid)

        # MSE loss (primary component)
        mse_loss = F.mse_loss(pred_valid, true_valid)

        # L1 loss (sparsity component)
        l1_loss = F.l1_loss(pred_valid, true_valid)

        # Binary classification loss for 0/1 values (truth table bits)
        # We want to encourage exact 0/1 predictions
        binary_values_mask = (true_valid == 0) | (true_valid == 1)
        if binary_values_mask.sum() > 0:
            binary_pred = pred_valid[binary_values_mask]
            binary_true = true_valid[binary_values_mask]

            # Binary cross entropy loss
            binary_pred_sigmoid = torch.sigmoid(5.0 * binary_pred)  # Sharpen the sigmoid
            binary_true_float = binary_true.float()
            bce_loss = F.binary_cross_entropy(binary_pred_sigmoid, binary_true_float)

            # Combined loss with binary classification component
            loss = mse_loss + self.l1_weight * l1_loss + self.binary_loss_weight * bce_loss
        else:
            # Combined loss without binary component
            loss = mse_loss + self.l1_weight * l1_loss

        return loss

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features to improve loss calculation stability."""
        # Standard normalization per feature
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
        return (features - mean) / std

class FeatureMetrics:
    """Utility class to compute metrics for evaluating truth table feature prediction."""

    @staticmethod
    def compute_metrics(
            pred_features: torch.Tensor,
            true_features: torch.Tensor,
            ignore_padding: bool = True
    ) -> Dict[str, float]:
        """
        Compute metrics for evaluating prediction quality.

        Args:
            pred_features: Predicted node features [num_masked_nodes, feature_dim]
            true_features: Target node features [num_masked_nodes, feature_dim]
            ignore_padding: Whether to ignore padded values (-1) in metrics computation

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Create mask for padding and valid values
        if ignore_padding:
            valid_mask = ~((true_features == -1) | torch.isnan(true_features))
            pred_valid = pred_features[valid_mask]
            true_valid = true_features[valid_mask]
        else:
            pred_valid = pred_features
            true_valid = true_features

        # Ensure 2D tensor for metrics
        pred_valid = pred_valid.view(-1)
        true_valid = true_valid.view(-1)

        # Mean Squared Error
        metrics['mse'] = F.mse_loss(pred_valid, true_valid).item()

        # Mean Absolute Error
        metrics['mae'] = F.l1_loss(pred_valid, true_valid).item()

        # Relative L2 error (using vector norm)
        true_norm = torch.linalg.vector_norm(true_valid, ord=2)
        error_norm = torch.linalg.vector_norm(pred_valid - true_valid, ord=2)
        rel_l2 = (error_norm / (true_norm + 1e-8)).item()
        metrics['rel_l2'] = rel_l2

        # R-squared (coefficient of determination)
        true_mean = true_valid.mean()
        ss_tot = torch.sum((true_valid - true_mean) ** 2)
        ss_res = torch.sum((true_valid - pred_valid) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics['r2'] = r2.item()

        # Add binary classification metrics for truth table values (0/1)
        binary_values_mask = (true_valid == 0) | (true_valid == 1)
        if binary_values_mask.sum() > 0:
            binary_pred = pred_valid[binary_values_mask]
            binary_true = true_valid[binary_values_mask]

            # Convert to binary predictions
            binary_pred_thresholded = (binary_pred > 0.5).float()

            # Accuracy
            accuracy = (binary_pred_thresholded == binary_true).float().mean().item()
            metrics['binary_accuracy'] = accuracy

            # Calculate precision, recall, and F1 for binary values
            tp = ((binary_pred_thresholded == 1) & (binary_true == 1)).float().sum().item()
            fp = ((binary_pred_thresholded == 1) & (binary_true == 0)).float().sum().item()
            fn = ((binary_pred_thresholded == 0) & (binary_true == 1)).float().sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1

        return metrics

    @staticmethod
    def compute_truth_table_accuracy(
            pred_features: torch.Tensor,
            true_features: torch.Tensor,
            threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute accuracy for truth table predictions (treating them as binary values).

        Args:
            pred_features: Predicted truth table features [num_masked_nodes, feature_dim]
            true_features: Target truth table features [num_masked_nodes, feature_dim]
            threshold: Threshold for converting predictions to binary values

        Returns:
            Dictionary with accuracy metrics
        """
        # Create mask for valid truth table values (0 or 1)
        valid_mask = (true_features == 0) | (true_features == 1)

        # Count number of valid truth table entries
        num_valid = valid_mask.sum().item()

        if num_valid == 0:
            return {
                'truth_table_accuracy': 0.0,
                'correct_bits': 0,
                'total_bits': 0
            }

        # Convert predictions to binary using threshold
        binary_pred = (pred_features > threshold).float()

        # Calculate accuracy only on valid entries
        correct_predictions = (binary_pred[valid_mask] == true_features[valid_mask]).float().sum().item()
        accuracy = correct_predictions / num_valid

        return {
            'truth_table_accuracy': accuracy,
            'correct_bits': correct_predictions,
            'total_bits': num_valid
        }
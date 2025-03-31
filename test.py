import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import logging
import os
import json
import numpy as np
from typing import Dict, Any, Optional

from loss import FeatureMetrics


def test(
        model: nn.Module,
        test_dataset,
        feature_dim: int = 256,  # Updated default for truth table size
        node_type_dim: int = 3,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: Optional[str] = None,
        output_dir: str = "./test_results"
) -> Dict[str, Any]:
    """
    Test the AIG transformer model on a test dataset.

    Args:
        model: AIG Transformer model
        test_dataset: Test dataset
        feature_dim: Dimension of node features
        node_type_dim: Dimension of node type encoding
        batch_size: Batch size for testing
        device: Device to test on
        model_path: Path to the model checkpoint to load (if None, use model as is)
        output_dir: Directory to save test results

    Returns:
        Dictionary with test results
    """
    # Set up logging
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "test.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Add console handler to see logs in stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info("Starting testing")

    # Move model to device
    device = torch.device(device)
    model = model.to(device)

    # Load model checkpoint if provided
    if model_path is not None:
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Initialize dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    all_metrics = {"mse": 0.0, "mae": 0.0, "rel_l2": 0.0, "r2": 0.0}
    total_tt_metrics = {"correct_bits": 0, "total_bits": 0}

    # Additional statistics
    per_graph_metrics = []
    prediction_errors = []

    # Testing loop
    num_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)

            # Extract predictions and ground truth
            pred_features = outputs['node_features']
            node_mask = batch.mask

            # Get features of masked nodes (only truth table part)
            true_features = batch.y[node_mask, node_type_dim:]

            # Create padding mask for -1 values
            padding_mask = (true_features == -1)
            valid_mask = ~padding_mask

            # Compute metrics (ignoring padding)
            metrics = FeatureMetrics.compute_metrics(pred_features, true_features, ignore_padding=True)

            # Calculate truth table accuracy
            tt_metrics = FeatureMetrics.compute_truth_table_accuracy(pred_features, true_features)

            # Update totals
            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k] += v

            total_tt_metrics["correct_bits"] += tt_metrics["correct_bits"]
            total_tt_metrics["total_bits"] += tt_metrics["total_bits"]

            # Collect per-graph metrics if batch tracking is available
            if hasattr(batch, 'batch'):
                for graph_idx in torch.unique(batch.batch):
                    graph_mask = batch.batch[node_mask] == graph_idx
                    if graph_mask.sum() > 0:
                        graph_pred = pred_features[graph_mask]
                        graph_true = true_features[graph_mask]

                        # Compute per-graph metrics
                        graph_metrics = FeatureMetrics.compute_metrics(
                            graph_pred, graph_true, ignore_padding=True
                        )

                        # Add truth table accuracy for this graph
                        graph_tt_metrics = FeatureMetrics.compute_truth_table_accuracy(
                            graph_pred, graph_true
                        )
                        graph_metrics["truth_table_accuracy"] = graph_tt_metrics["truth_table_accuracy"]

                        per_graph_metrics.append(graph_metrics)

            # Collect absolute errors for distribution analysis (ignoring padding)
            abs_errors = torch.abs(pred_features - true_features)
            abs_errors = abs_errors[valid_mask].cpu().numpy()
            prediction_errors.extend(abs_errors.flatten().tolist())

            num_batches += 1

    # Calculate averages
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

    # Calculate truth table accuracy
    if total_tt_metrics["total_bits"] > 0:
        avg_metrics["truth_table_accuracy"] = total_tt_metrics["correct_bits"] / total_tt_metrics["total_bits"]
    else:
        avg_metrics["truth_table_accuracy"] = 0.0

    logger.info(f"Test Results - MSE: {avg_metrics['mse']:.6f}, MAE: {avg_metrics['mae']:.6f}, "
                f"Relative L2: {avg_metrics['rel_l2']:.6f}, R²: {avg_metrics['r2']:.6f}, "
                f"TT Accuracy: {avg_metrics['truth_table_accuracy']:.6f}")

    # Compute error statistics
    error_stats = {
        "min": float(np.min(prediction_errors)) if prediction_errors else 0.0,
        "max": float(np.max(prediction_errors)) if prediction_errors else 0.0,
        "mean": float(np.mean(prediction_errors)) if prediction_errors else 0.0,
        "median": float(np.median(prediction_errors)) if prediction_errors else 0.0,
        "std": float(np.std(prediction_errors)) if prediction_errors else 0.0,
        "percentiles": {
            "25": float(np.percentile(prediction_errors, 25)) if prediction_errors else 0.0,
            "50": float(np.percentile(prediction_errors, 50)) if prediction_errors else 0.0,
            "75": float(np.percentile(prediction_errors, 75)) if prediction_errors else 0.0,
            "90": float(np.percentile(prediction_errors, 90)) if prediction_errors else 0.0,
            "95": float(np.percentile(prediction_errors, 95)) if prediction_errors else 0.0,
            "99": float(np.percentile(prediction_errors, 99)) if prediction_errors else 0.0
        }
    }

    # Compute per-graph statistics if available
    graph_stats = {}
    if per_graph_metrics:
        for metric in ['mse', 'mae', 'rel_l2', 'r2', 'truth_table_accuracy']:
            if metric in per_graph_metrics[0]:
                values = [m[metric] for m in per_graph_metrics]
                graph_stats[metric] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values))
                }

    # Truth table specific analysis
    if total_tt_metrics["total_bits"] > 0:
        tt_stats = {
            "total_bits_evaluated": int(total_tt_metrics["total_bits"]),
            "correct_bits": int(total_tt_metrics["correct_bits"]),
            "accuracy": float(total_tt_metrics["correct_bits"] / total_tt_metrics["total_bits"])
        }
    else:
        tt_stats = {
            "total_bits_evaluated": 0,
            "correct_bits": 0,
            "accuracy": 0.0
        }

    # Save results to file
    test_results = {
        "overall_metrics": avg_metrics,
        "error_statistics": error_stats,
        "per_graph_statistics": graph_stats,
        "truth_table_statistics": tt_stats
    }

    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    logger.info(f"Test results saved to {os.path.join(output_dir, 'test_results.json')}")

    return test_results
import argparse
import os
import torch
import logging
import numpy as np
import random
from datetime import datetime
import json
from pathlib import Path

# Import your custom modules
from aig_dataset import AIGDataset  # Your dataset class
from loss import TruthTableFeatureLoss, FeatureMetrics  # Updated import
from train import train
from test import test


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train and test AIG Transformer for node feature prediction")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train", "test", "train_test"], default="train_test",
                        help="Mode: train, test, or train_test (default)")

    # Dataset parameters
    parser.add_argument("--data_path", type=str, default="complete_tt_graphs.pkl",
                        help="Path to AIG dataset pickle file")
    parser.add_argument("--processed_dir", type=str, default="data", help="Directory for processed data")
    parser.add_argument("--mask_ratio", type=float, default=0.20, help="Ratio of nodes to mask")
    parser.add_argument("--num_graphs", type=int, default=100, help="Number of graphs to use (default: all)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data for validation")
    parser.add_argument("--mask_mode", type=str, default="and_gates",
                        choices=["random", "and_gates", "inputs"],
                        help="Strategy for masking nodes")

    # Model parameters
    parser.add_argument("--node_type_dim", type=int, default=3, help="Dimension of node type encoding")
    parser.add_argument("--feature_dim", type=int, default=256,
                        help="Dimension of truth table features")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--max_hop", type=int, default=5, help="Maximum hop distance for DAG attention")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--edge_dim", type=int, default=2, help="Edge feature dimension")
    parser.add_argument("--max_tt_length", type=int, default=256, help="Maximum truth table length")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--l1_weight", type=float, default=0.1, help="Weight for L1 loss")
    parser.add_argument("--binary_loss_weight", type=float, default=2.0,
                        help="Weight for binary classification loss")
    parser.add_argument("--feature_normalization", action="store_true",
                        help="Normalize features in loss calculation")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval (epochs)")
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Factor for learning rate scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Patience for learning rate scheduler")

    # Testing parameters
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint to load (required for test mode)")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./runs/DATETIME)")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d%H")
        args.output_dir = os.path.join("runs", f"{timestamp}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "run.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Add console handler to see logs in stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger.info(f"Arguments saved to {os.path.join(args.output_dir, 'args.json')}")
    logger.info(f"Mode: {args.mode}")

    # Load dataset
    logger.info("Loading dataset...")

    # Import model class (this is done here to avoid circular imports)
    from model import AIGTransformer

    if args.mode in ["train", "train_test"]:
        # Load datasets for training, validation, and testing
        dataset = AIGDataset(
            file_path=args.data_path,
            processed_dir=args.processed_dir,
            mask_ratio=args.mask_ratio,
            mask_mode=args.mask_mode,
            node_type_dim=args.node_type_dim,
            num_graphs=args.num_graphs
        )

        # Split dataset
        num_graphs = len(dataset)
        indices = list(range(num_graphs))
        np.random.shuffle(indices)

        train_size = int(args.train_ratio * num_graphs)
        val_size = int(args.val_ratio * num_graphs)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        logger.info(f"Dataset split: {len(train_dataset)} training, "
                    f"{len(val_dataset)} validation, {len(test_dataset)} testing")

        # Initialize model
        logger.info("Initializing model...")
        model = AIGTransformer(
            node_type_dim=args.node_type_dim,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_hop=args.max_hop,
            edge_type_dim=args.edge_dim,
            max_tt_length=args.max_tt_length
        )

        logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

        # Create directories for model checkpoints
        model_dir = os.path.join(args.output_dir, "model_checkpoints")
        os.makedirs(model_dir, exist_ok=True)

        # Create directories for logs
        log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Train model
        logger.info("Starting training...")
        train_results = train(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            feature_dim=args.feature_dim,
            node_type_dim=args.node_type_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            l1_weight=args.l1_weight,
            binary_loss_weight=args.binary_loss_weight,
            feature_normalization=args.feature_normalization,
            val_interval=args.val_interval,
            patience=args.patience,
            clip_grad_norm=args.clip_grad_norm,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            device=args.device,
            save_dir=model_dir,
            log_dir=log_dir
        )

        logger.info(f"Training completed. Best model saved at {train_results['best_model_path']}")

        # Set best model path for testing if we're doing train_test
        model_path = train_results['best_model_path']

    if args.mode in ["test", "train_test"]:
        if args.mode == "test" and args.model_path is None:
            logger.error("Model path must be specified in test mode")
            return

        # Use model path from command line if in test mode
        if args.mode == "test":
            model_path = args.model_path

            # Load test dataset
            test_dataset = AIGDataset(
                file_path=args.data_path,
                processed_dir=args.processed_dir,
                mask_ratio=args.mask_ratio,
                mask_mode=args.mask_mode,
                node_type_dim=args.node_type_dim,
                num_graphs=args.num_graphs
            )

            # Initialize model for testing
            model = AIGTransformer(
                node_type_dim=args.node_type_dim,
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_hop=args.max_hop,
                edge_type_dim=args.edge_dim,
                max_tt_length=args.max_tt_length
            )

        # Create test results directory
        test_dir = os.path.join(args.output_dir, "test_results")
        os.makedirs(test_dir, exist_ok=True)

        # Test model
        logger.info(f"Testing model from {model_path}...")
        test_results = test(
            model=model,
            test_dataset=test_dataset,
            feature_dim=args.feature_dim,
            node_type_dim=args.node_type_dim,
            batch_size=args.batch_size,
            device=args.device,
            model_path=model_path,
            output_dir=test_dir
        )

        # Log test results
        logger.info(f"Test MSE: {test_results['overall_metrics']['mse']:.6f}")
        logger.info(f"Test MAE: {test_results['overall_metrics']['mae']:.6f}")
        logger.info(f"Test R²: {test_results['overall_metrics']['r2']:.6f}")
        logger.info(f"Test Truth Table Accuracy: {test_results['overall_metrics']['truth_table_accuracy']:.6f}")
        logger.info(f"Test results saved to {test_dir}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
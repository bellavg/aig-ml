import os
import argparse
import logging
import json
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

# Import your custom modules
from aig_dataset import AIGDataset
from model import AIGTransformerLightning


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate AIG Transformer using PyTorch Lightning")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train", "test", "train_test"], default="train_test",
                        help="Mode: train, test, or train_test (default)")

    # Dataset parameters
    parser.add_argument("--data_path", type=str, default="complete_tt_graphs.pkl",
                        help="Path to AIG dataset pickle file")
    parser.add_argument("--processed_dir", type=str, default="data", help="Directory for processed data")
    parser.add_argument("--mask_ratio", type=float, default=0.20, help="Ratio of nodes to mask")
    parser.add_argument("--num_graphs", type=int, default=100, help="Number of graphs to use (default: 100)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data for validation")
    parser.add_argument("--mask_mode", type=str, default="and_gates",
                        choices=["random", "and_gates", "inputs"],
                        help="Strategy for masking nodes")

    # Model parameters
    parser.add_argument("--node_type_dim", type=int, default=3, help="Dimension of node type encoding")
    parser.add_argument("--feature_dim", type=int, default=256, help="Dimension of truth table features")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--max_hop", type=int, default=5, help="Maximum hop distance for DAG attention")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--edge_dim", type=int, default=2, help="Edge feature dimension")
    parser.add_argument("--max_tt_length", type=int, default=256, help="Maximum truth table length")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--l1_weight", type=float, default=0.1, help="Weight for L1 loss")
    parser.add_argument("--binary_loss_weight", type=float, default=2.0, help="Weight for binary classification loss")
    parser.add_argument("--feature_normalization", action="store_true", help="Normalize features in loss calculation")
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Factor for learning rate scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Patience for learning rate scheduler")

    # Lightning specific parameters
    parser.add_argument("--precision", type=str, default="32", choices=["32", "16", "bf16"],
                        help="Precision for training (32, 16, or bf16)")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator to use (auto, gpu, cpu)")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of compute nodes to use")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Training strategy (None, ddp, deepspeed, etc.)")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Number of batches to accumulate gradients")
    parser.add_argument("--val_check_interval", type=float, default=1.0,
                        help="How often to check validation (1.0 = once per epoch)")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="Limit validation batches (1.0 = use all)")
    parser.add_argument("--log_every_n_steps", type=int, default=50,
                        help="How often to log metrics within an epoch")

    # Testing parameters
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint to load (required for test mode)")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./lightning_logs/TIMESTAMP)")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("lightning_logs", f"{timestamp}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "training.log"),
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

    # Initialize the dataset
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
    random.shuffle(indices)

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

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Initialize Lightning loggers
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="tensorboard_logs"
    )
    csv_logger = CSVLogger(
        save_dir=args.output_dir,
        name="csv_logs"
    )

    # Initialize callbacks
    callbacks = [
        # Save the model with the best validation loss
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        # Early stopping based on validation loss
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min",
            verbose=True
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="epoch")
    ]

    # Initialize training strategy
    if args.strategy == "ddp" and args.devices > 1:
        # Use DDP strategy for multi-GPU training
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = args.strategy

    # Initialize model
    if args.mode in ["train", "train_test"]:
        logger.info("Initializing model...")

        model = AIGTransformerLightning(
            node_type_dim=args.node_type_dim,
            feature_dim=args.feature_dim,
            edge_type_dim=args.edge_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_hop=args.max_hop,
            gnn_type="gcn",
            max_tt_length=args.max_tt_length,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            l1_weight=args.l1_weight,
            feature_normalization=args.feature_normalization,
            binary_loss_weight=args.binary_loss_weight,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            clip_grad_norm=args.clip_grad_norm
        )

        logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            num_nodes=args.num_nodes,
            strategy=strategy,
            precision=args.precision,
            callbacks=callbacks,
            logger=[tb_logger, csv_logger],
            log_every_n_steps=args.log_every_n_steps,
            accumulate_grad_batches=args.accumulate_grad_batches,
            val_check_interval=args.val_check_interval,
            limit_val_batches=args.limit_val_batches,
            deterministic=True
        )

        # Train model
        logger.info("Starting training...")
        trainer.fit(model, train_loader, val_loader)

        # Get path to best checkpoint
        best_model_path = trainer.checkpoint_callback.best_model_path
        logger.info(f"Training completed. Best model saved at {best_model_path}")

        # Use best checkpoint for testing
        checkpoint_path = best_model_path

    # Testing
    if args.mode in ["test", "train_test"]:
        if args.mode == "test":
            # Load the specified checkpoint
            if args.checkpoint_path is None:
                logger.error("Model checkpoint path must be specified in test mode")
                return
            checkpoint_path = args.checkpoint_path

            # Initialize model for testing (parameters will be loaded from checkpoint)
            model = AIGTransformerLightning(
                node_type_dim=args.node_type_dim,
                feature_dim=args.feature_dim,
                edge_type_dim=args.edge_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_hop=args.max_hop,
                max_tt_length=args.max_tt_length
            )

        # Initialize testing trainer
        test_trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=min(1, args.devices),  # Use only one device for testing
            logger=[tb_logger, csv_logger],
            precision=args.precision
        )

        # Test model
        logger.info(f"Testing model from {checkpoint_path}...")
        test_results = test_trainer.test(model, test_loader, ckpt_path=checkpoint_path)

        # Save detailed test results
        test_output_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_output_path, "w") as f:
            json.dump(test_results[0], f, indent=4)

        logger.info(f"Test results saved to {test_output_path}")

        # Log test results
        logger.info(f"Test MSE: {test_results[0]['test_mse_final']:.6f}")
        logger.info(f"Test MAE: {test_results[0]['test_mae_final']:.6f}")
        logger.info(f"Test R²: {test_results[0]['test_r2_final']:.6f}")
        logger.info(f"Test Truth Table Accuracy: {test_results[0]['test_tt_accuracy_final']:.6f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
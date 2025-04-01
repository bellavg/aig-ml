"""
Script for running AIG Transformer training on a GPU cluster with SLURM.

Example usage:
    # Submit a job to SLURM
    sbatch run_training.sh

Contents of run_training.sh:
    #!/bin/bash
    #SBATCH --job-name=aig-transformer
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=4
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64G
    #SBATCH --time=24:00:00
    #SBATCH --output=slurm_logs/%j.out
    #SBATCH --error=slurm_logs/%j.err

    # Activate your environment
    source /path/to/your/environment/bin/activate

    # Run the script
    python cluster_run.py \
        --num_nodes=$SLURM_JOB_NUM_NODES \
        --devices=4 \
        --strategy=ddp \
        --precision=16 \
        --batch_size=64 \
        --accumulate_grad_batches=4 \
        --num_epochs=100
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
import argparse
from datetime import datetime
import logging
import random
import numpy as np
import json

# Import your modules
from aig_dataset import AIGDataset
from model import AIGTransformerLightning
from datamodule import AIGDataModule


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
    parser = argparse.ArgumentParser(description="Run AIG Transformer training on a GPU cluster")

    # Dataset parameters
    parser.add_argument("--data_path", type=str, default="complete_tt_graphs.pkl",
                        help="Path to AIG dataset pickle file")
    parser.add_argument("--processed_dir", type=str, default="data", help="Directory for processed data")
    parser.add_argument("--mask_ratio", type=float, default=0.20, help="Ratio of nodes to mask")
    parser.add_argument("--num_graphs", type=int, default=None, help="Number of graphs to use (default: all)")
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
    parser.add_argument("--precision", type=str, default="16", choices=["32", "16", "bf16"],
                        help="Precision for training (32, 16, or bf16)")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator to use")
    parser.add_argument("--strategy", type=str, default="ddp",
                        choices=["ddp", "deepspeed", "fsdp", None],
                        help="Training strategy (ddp, deepspeed, fsdp, etc.)")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Number of batches to accumulate gradients")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker processes for data loading")
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for data loading")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./lightning_logs/TIMESTAMP)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

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
        filename=os.path.join(args.output_dir, "cluster_training.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Add console handler to see logs in stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Save arguments
    with open(os.path.join(args.output_dir, "cluster_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger.info(f"Arguments saved to {os.path.join(args.output_dir, 'cluster_args.json')}")

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

    # Initialize DDP strategy
    if args.strategy == "ddp":
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,  # Might improve performance for fixed graph structure
            gradient_as_bucket_view=True,  # More memory efficient
        )
    else:
        strategy = args.strategy

    # Initialize data module
    datamodule = AIGDataModule(
        data_path=args.data_path,
        processed_dir=args.processed_dir,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        node_type_dim=args.node_type_dim,
        num_graphs=args.num_graphs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # Initialize model
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

    # Initialize trainer with plugins for SLURM environment
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        strategy=strategy,
        callbacks=callbacks,
        logger=csv_logger,
        deterministic=True,
        resume_from_checkpoint=args.resume_from_checkpoint,
        # Additional optimization options
        gradient_clip_val=args.clip_grad_norm if args.clip_grad_norm > 0 else None,
        enable_progress_bar=True,
        enable_model_summary=True,
        profiler="simple",  # Use 'advanced' for more detailed profiling
    )

    # Train model
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Get path to best checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Training completed. Best model saved at {best_model_path}")

    # Test model
    logger.info(f"Testing model from {best_model_path}...")
    test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)

    # Save test results
    test_output_path = os.path.join(args.output_dir, "test_results.json")
    with open(test_output_path, "w") as f:
        json.dump(test_results[0], f, indent=4)

    logger.info(f"Test results saved to {test_output_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
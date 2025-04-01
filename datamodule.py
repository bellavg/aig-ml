import os
import random
import numpy as np
import torch
from typing import Optional, List, Union, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

# Import your dataset
from aig_dataset import AIGDataset


class AIGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for AIG datasets.
    Handles loading, splitting, and preparing data for training.
    """

    def __init__(
            self,
            data_path: str = "complete_tt_graphs.pkl",
            processed_dir: str = "data",
            mask_ratio: float = 0.15,
            mask_mode: str = "and_gates",
            node_type_dim: int = 3,
            num_graphs: Optional[int] = None,
            train_val_test_split: List[float] = [0.7, 0.1, 0.2],
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
            seed: int = 42
    ):
        """
        Initialize the AIG data module.

        Args:
            data_path: Path to the original pickle file
            processed_dir: Directory where processed data will be saved
            mask_ratio: Percentage of node features to mask
            mask_mode: Strategy for masking ("random", "and_gates", "inputs")
            node_type_dim: Dimension of the node type one-hot encoding
            num_graphs: Number of graphs to use (None = all)
            train_val_test_split: Ratios for train/val/test splits
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to use pinned memory for data loading
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.data_path = data_path
        self.processed_dir = processed_dir
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.node_type_dim = node_type_dim
        self.num_graphs = num_graphs
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        # Save split ratios and validate they sum to 1
        assert sum(train_val_test_split) == 1.0, "Split ratios must sum to 1.0"
        self.train_ratio, self.val_ratio, self.test_ratio = train_val_test_split

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize dataset variables
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download or prepare the dataset on the main process.
        This method is called only once on the main process.
        """
        # Load dataset to make sure it's prepared, data won't be loaded into memory
        AIGDataset(
            file_path=self.data_path,
            processed_dir=self.processed_dir,
            mask_ratio=self.mask_ratio,
            mask_mode=self.mask_mode,
            node_type_dim=self.node_type_dim,
            num_graphs=self.num_graphs
        )

    def setup(self, stage: Optional[str] = None):
        """
        Prepare splits and create datasets for each stage.
        This method is called on every process when using DDP.

        Args:
            stage: Either 'fit', 'validate', 'test' or None
        """
        # Load dataset if not already loaded
        if self.dataset is None:
            self.dataset = AIGDataset(
                file_path=self.data_path,
                processed_dir=self.processed_dir,
                mask_ratio=self.mask_ratio,
                mask_mode=self.mask_mode,
                node_type_dim=self.node_type_dim,
                num_graphs=self.num_graphs
            )

        # Create train/val/test splits if not already created
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            # Get dataset size
            num_graphs = len(self.dataset)
            indices = list(range(num_graphs))
            random.shuffle(indices)

            # Calculate split sizes
            train_size = int(self.train_ratio * num_graphs)
            val_size = int(self.val_ratio * num_graphs)
            test_size = num_graphs - train_size - val_size

            # Create splits
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # Create subsets
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
            self.test_dataset = Subset(self.dataset, test_indices)

            print(f"Dataset split: {len(self.train_dataset)} training, "
                  f"{len(self.val_dataset)} validation, {len(self.test_dataset)} testing")

    def train_dataloader(self):
        """Returns the training dataloader."""
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        """Returns the test dataloader."""
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def get_train_val_test_sizes(self):
        """Returns the sizes of the train, validation, and test sets."""
        return {
            'train_size': len(self.train_dataset) if self.train_dataset else 0,
            'val_size': len(self.val_dataset) if self.val_dataset else 0,
            'test_size': len(self.test_dataset) if self.test_dataset else 0,
            'total_size': len(self.dataset) if self.dataset else 0
        }

    def get_node_type_dim(self):
        """Returns the node type dimension."""
        return self.node_type_dim
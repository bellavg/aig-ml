import torch
import numpy as np
import random
from torch_geometric.data import Data, Dataset
import os
import pickle
import networkx as nx


class AIGDataset(Dataset):
    """
    Dataset for masked node feature prediction on AIG graphs.

    This dataset masks a portion of node features (keeping node type intact)
    and sets up the target task as predicting the original features.
    """

    def __init__(
            self,
            file_path: str = "complete_tt_graphs.pkl",
            processed_dir: str = "node_mask_processed",
            processed_file: str = "node_masked_data.pt",
            mask_ratio: float = 0.15,
            mask_mode: str = "and_gates",
            node_type_dim: int = 3,
            transform=None,
            pre_transform=None,
            num_graphs: int = 10000
    ):
        """
        Initialize the masked AIG dataset.

        Args:
            file_path: Path to the original pickle file (if processing from scratch)
            processed_dir: Directory where processed data will be saved
            processed_file: Filename for the processed data
            mask_ratio: Percentage of node features to mask (default: 0.15)
            mask_mode: Strategy for masking ("random", "and_gates", "inputs")
            node_type_dim: Dimension of the node type one-hot encoding (default: 3)
            transform: PyG transform to apply on each sample
            pre_transform: PyG transform to apply before saving processed data
            num_graphs: Number of graphs to load (default: 10000)
        """
        self.file_path = file_path
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.node_type_dim = node_type_dim
        self.num_graphs = num_graphs
        self._processed_file = processed_file
        self._data_list = None

        # Node type mapping (from one-hot encoding to label)
        self.node_types = {
            "ZERO": [0, 0, 0],
            "PI": [1, 0, 0],
            "AND": [0, 1, 0],
            "PO": [0, 0, 1]
        }

        # Initialize PyG Dataset with the root directory
        super().__init__(root=processed_dir, transform=transform, pre_transform=pre_transform)

        # Check if processed file exists and load it
        processed_path = os.path.join(self.processed_dir, self._processed_file)
        if os.path.exists(processed_path):
            self._load_processed_data()
            print(f"Loaded pre-processed masked dataset from {processed_path}")

    def _load_processed_data(self):
        """Load processed data directly."""
        processed_path = os.path.join(self.processed_dir, self._processed_file)
        with torch.serialization.safe_globals(["torch_geometric.data.Data"]):
            self._data_list = torch.load(processed_path, weights_only=False)

        # Apply num_graphs limit if specified
        if self.num_graphs is not None and self.num_graphs < len(self._data_list):
            self._data_list = self._data_list[:self.num_graphs]
            print(f"Limited dataset to {self.num_graphs} graphs")

    @property
    def processed_file_names(self):
        """Required for PyG to avoid NotImplementedError."""
        return [self._processed_file]

    def process(self):
        """
        Process the AIG graphs: load from pickle, convert to PyG, and apply masking.
        """
        # If data is already loaded, no need to process
        if self._data_list is not None:
            return

        # Check if processed file exists
        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path):
            self._load_processed_data()
            return

        # If processing is needed but no file_path is provided, raise an error
        if self.file_path is None:
            raise ValueError("file_path must be provided when processed data doesn't exist")

        # Process data from pickle file
        print(f"Processing data from {self.file_path} with masking...")

        with open(self.file_path, 'rb') as f:
            nx_graphs = pickle.load(f)

        # If num_graphs is specified, take only the first `num_graphs` graphs
        if self.num_graphs is not None:
            nx_graphs = nx_graphs[:self.num_graphs]
            print(f"Limited dataset to {self.num_graphs} graphs")

        data_list = []
        for nx_graph in nx_graphs:
            # Convert to PyG Data
            data = self.convert_to_pyg_data(nx_graph)

            # Apply masking
            data = self.apply_masking(data)

            data_list.append(data)

        # Save processed data
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(data_list, processed_path)
        self._data_list = data_list
        print(f"Saved processed masked dataset to {processed_path}")

    def convert_to_pyg_data(self, nx_graph: nx.DiGraph) -> Data:
        """
        Convert networkx graph to PyG Data object.

        Args:
            nx_graph: NetworkX directed graph representing an AIG

        Returns:
            PyG Data object with node features, edge indices and edge attributes
        """
        # Extract edge information
        edges = list(nx_graph.edges(data=True))
        edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t().contiguous()

        # Extract edge attributes (types)
        edge_attr = torch.tensor([d['type'] for _, _, d in edges], dtype=torch.float)

        # Extract node features (type + truth table)
        node_features = []
        for n in sorted(nx_graph.nodes()):
            node_data = nx_graph.nodes[n]
            node_type = torch.tensor(node_data['type'], dtype=torch.float)
            node_feature = torch.tensor(node_data['feature'], dtype=torch.float)

            # Concatenate node type and feature
            node_features.append(torch.cat([node_type, node_feature]))

        # Stack all node features
        x = torch.stack(node_features)

        # Store graph metadata
        graph_data = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }

        # Store graph-level attributes if available
        if 'inputs' in nx_graph.graph:
            graph_data['num_inputs'] = nx_graph.graph['inputs']
        if 'outputs' in nx_graph.graph:
            graph_data['num_outputs'] = nx_graph.graph['outputs']

        return Data(**graph_data)

    def apply_masking(self, data: Data) -> Data:
        """
        Apply masking to node features based on the specified mask mode.

        This keeps the original features as the target (y) and creates a mask
        indicating which nodes have been masked.

        Args:
            data: PyG Data object containing the graph

        Returns:
            PyG Data object with masked features and mask indicator
        """
        x = data.x
        num_nodes = x.size(0)

        # Store original features as target
        data.y = x.clone()

        # Create a mask: initially all False (no masking)
        mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Determine which nodes to mask based on mask_mode
        if self.mask_mode == "random":
            # Mask random nodes
            num_to_mask = max(1, int(num_nodes * self.mask_ratio))
            nodes_to_mask = random.sample(range(num_nodes), num_to_mask)
            mask[nodes_to_mask] = True

        elif self.mask_mode == "and_gates":
            # Identify node types based on one-hot encoding
            node_types = torch.argmax(x[:, :self.node_type_dim], dim=1)

            # Find AND gates (node type index 1)
            maskable_indices = torch.where(node_types == 1)[0].tolist()

            # Calculate the number of nodes to mask
            num_to_mask = max(1, int(len(maskable_indices) * self.mask_ratio))

            # Randomly select nodes to mask
            if len(maskable_indices) > 0:
                nodes_to_mask = random.sample(maskable_indices, min(num_to_mask, len(maskable_indices)))
                mask[nodes_to_mask] = True

        elif self.mask_mode == "inputs":
            # Identify input nodes (PI, node type index 0)
            node_types = torch.argmax(x[:, :self.node_type_dim], dim=1)
            maskable_indices = torch.where(node_types == 0)[0].tolist()

            # Calculate the number of nodes to mask
            num_to_mask = max(1, int(len(maskable_indices) * self.mask_ratio))

            # Randomly select nodes to mask
            if len(maskable_indices) > 0:
                nodes_to_mask = random.sample(maskable_indices, min(num_to_mask, len(maskable_indices)))
                mask[nodes_to_mask] = True
        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

        # Apply masking - only mask the feature part, keep node type intact
        masked_x = x.clone()
        masked_x[mask, self.node_type_dim:] = 0.0  # Mask with zeros
        data.x = masked_x

        # Store the mask
        data.mask = mask

        return data

    def len(self):
        """Returns the number of masked AIGs in the dataset."""
        # Load data if not already loaded
        if self._data_list is None:
            self._load_processed_data()
        return len(self._data_list)

    def get(self, idx):
        """Gets the masked AIG at the specified index."""
        # Load data if not already loaded
        if self._data_list is None:
            self._load_processed_data()
        return self._data_list[idx]
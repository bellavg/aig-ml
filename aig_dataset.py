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
            processed_file: str = "node_data.pt",  # Changed processed file name as it won't contain masking
            mask_ratio: float = 0.15,  # Keep mask_ratio as an instance variable
            mask_mode: str = "and_gates",
            node_type_dim: int = 3,
            transform=None,
            pre_transform=None,
            num_graphs: int = 10000,
            process_chunk_size: int = None
    ):
        self.file_path = file_path
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.node_type_dim = node_type_dim
        self.num_graphs = num_graphs
        self._processed_file = processed_file
        self._data_list = None
        self.process_chunk_size = process_chunk_size

        # Node type mapping
        self.node_types = {
            "ZERO": [0, 0, 0],
            "PI": [1, 0, 0],
            "AND": [0, 1, 0],
            "PO": [0, 0, 1]
        }

        super().__init__(root=processed_dir, transform=transform, pre_transform=pre_transform)

        # Check if processed file exists and load it
        processed_path = os.path.join(self.processed_dir, self._processed_file)
        if os.path.exists(processed_path):
            self._load_processed_data()
            print(f"Loaded pre-processed dataset from {processed_path}")

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
        Process the AIG graphs in chunks: load from pickle and convert to PyG.
        """
        # Check if processed file exists
        processed_path = self.processed_paths[0]
        existing_data_list = []
        if os.path.exists(processed_path):
            try:
                with torch.serialization.safe_globals(["torch_geometric.data.Data"]):
                    existing_data_list = torch.load(processed_path, weights_only=False)
                print(f"Loaded {len(existing_data_list)} existing processed graphs.")
            except Exception as e:
                print(f"Warning: Could not load existing processed data. Starting from scratch. Error: {e}")

        # Process data from pickle file
        print(f"Processing data from {self.file_path}...")

        with open(self.file_path, 'rb') as f:
            nx_graphs = pickle.load(f)
        total_graphs = len(nx_graphs)

        chunk_size = self.process_chunk_size if hasattr(self, 'process_chunk_size') and self.process_chunk_size is not None else total_graphs
        start_index = 0
        while start_index < total_graphs:
            end_index = min(start_index + chunk_size, total_graphs)
            nx_graphs_to_process = nx_graphs[start_index:end_index]
            print(f"Processing graphs from index {start_index} to {end_index - 1} (total {len(nx_graphs_to_process)} in this chunk).")

            data_list_chunk = []
            for i, nx_graph in enumerate(nx_graphs_to_process):
                global_index = start_index + i
                print(f"Processing graph {global_index + 1} of {total_graphs}...", end='\r')
                try:
                    # Convert to PyG Data
                    data = self.convert_to_pyg_data(nx_graph)
                    data_list_chunk.append(data)
                except Exception as e:
                    print(f"Error processing graph at index {global_index}: {e}")

            combined_data_list = existing_data_list + data_list_chunk
            existing_data_list = combined_data_list # Update the list for the next chunk

            # Save processed data
            os.makedirs(self.processed_dir, exist_ok=True)
            torch.save(combined_data_list, processed_path)
            print(f"\nSaved {len(data_list_chunk)} processed graphs in this chunk. Total processed: {len(combined_data_list)}")

            start_index = end_index

        self._data_list = combined_data_list
        print(f"\nFinished processing and saved a total of {len(self._data_list)} processed graphs to {processed_path}")

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

        # Store original features as target
        graph_data['y'] = x.clone() # Original features as target

        # Store graph-level attributes if available
        if 'inputs' in nx_graph.graph:
            graph_data['num_inputs'] = nx_graph.graph['inputs']
        if 'outputs' in nx_graph.graph:
            graph_data['num_outputs'] = nx_graph.graph['outputs']

        return Data(**graph_data)

    def apply_masking(self, data: Data) -> Data:
        """
        Apply masking to node features based on the instance's mask_ratio.

        This keeps the original features as the target (y) and creates a mask
        indicating which nodes have been masked.

        Args:
            data: PyG Data object containing the graph

        Returns:
            PyG Data object with masked features and mask indicator
        """
        x = data.x
        num_nodes = x.size(0)

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
        """Gets the masked AIG at the specified index and applies masking."""
        # Load data if not already loaded
        if self._data_list is None:
            self._load_processed_data()
        data = self._data_list[idx]
        data = self.apply_masking(data) # Apply masking here
        return data
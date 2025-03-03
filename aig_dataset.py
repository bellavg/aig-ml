
import torch.serialization

import pickle
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import torch.serialization
import os


class AIGDataset(Dataset):
    def __init__(self, file_path=None, num_graphs=None, transform=None, pre_transform=None,
                 root="", processed_file="data.pt"):
        """
        Dataset for And-Inverter Graphs (AIGs) with an option to load directly from processed data.pt.

        Args:
            file_path (str, optional): Path to the original pickle file (not needed if processed data exists)
            num_graphs (int, optional): Number of graphs to load (default: None, loads all)
            transform (callable, optional): A function/transform applied to a `torch_geometric.data.Data` object
            pre_transform (callable, optional): A function/transform applied before saving data
            root (str): Root directory where processed data is stored (default: ".processed/")
            processed_file (str): Name of the processed data file (default: "data.pt")
        """
        self.file_path = file_path
        self.num_graphs = num_graphs
        self._processed_file = processed_file
        self._data_list = None

        # Initialize PyG Dataset with the root directory
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

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
        Processing is only needed if data.pt doesn't exist and file_path is provided.
        The original pickle-to-PyG conversion code would go here.
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
        print(f"Processing data from {self.file_path}...")

        with open(self.file_path, 'rb') as f:
            nx_graphs = pickle.load(f)

        # If num_graphs is specified, take only the first `num_graphs` graphs
        if self.num_graphs is not None:
            nx_graphs = nx_graphs[:self.num_graphs]
            print(f"Limited dataset to {self.num_graphs} graphs")

        data_list = []
        for nx_graph in nx_graphs:
            data = self.convert_to_pyg_data(nx_graph)
            data_list.append(data)

        # Save processed data
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(data_list, processed_path)
        self._data_list = data_list
        print(f"Saved processed dataset to {processed_path}")

    def convert_to_pyg_data(self, nx_graph):
        """Convert networkx graph to PyG Data object."""
        node_features = []
        for n in nx_graph.nodes():
            node_type = np.array(nx_graph.nodes[n]['type'], dtype=np.float32)
            node_feature = np.array([nx_graph.nodes[n]['feature']], dtype=np.float32)
            full_feature = np.concatenate([node_type, node_feature])
            node_features.append(full_feature)

        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()

        # Process edge attributes
        edge_attr_list = [nx_graph[u][v]['type'] for u, v in nx_graph.edges]
        edge_attr = torch.tensor(np.array(edge_attr_list, dtype=np.float32), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def len(self):
        """Returns the number of AIGs in the dataset."""
        # Load data if not already loaded
        if self._data_list is None:
            self._load_processed_data()
        return len(self._data_list)

    def get(self, idx):
        """Gets the AIG at the specified index."""
        # Load data if not already loaded
        if self._data_list is None:
            self._load_processed_data()
        return self._data_list[idx]
# class AIGDataset(Dataset):
#     def __init__(self, file_path, num_graphs=None, transform=None, pre_transform=None):
#         """
#         Dataset for And-Inverter Graphs (AIGs) with an option to limit the number of graphs.
#
#         Args:
#             file_path (str): Path to the pickle file containing the AIGs.
#             num_graphs (int, optional): Number of graphs to load (default: None, loads all).
#             transform (callable, optional): A function/transform applied to a `torch_geometric.data.Data` object.
#             pre_transform (callable, optional): A function/transform applied before saving data.
#         """
#         self.file_path = file_path
#         self.num_graphs = num_graphs  # How many graphs to load
#         self.processed_file = "data.pt"
#
#         super().__init__(root=".", transform=transform, pre_transform=pre_transform)
#
#     @property
#     def processed_file_names(self):
#         """Required for PyG to avoid NotImplementedError."""
#         return [self.processed_file]
#
#     def process(self):
#         """Loads the AIGs from the pickle file and converts them to PyTorch Geometric Data objects."""
#
#         processed_path = self.processed_paths[0]
#         # If data.pt already exists, skip processing
#         if os.path.exists(processed_path):
#             print(f"Found existing processed file at {processed_path}. Skipping processing.")
#             return
#
#
#         with open(self.file_path, 'rb') as f:
#             nx_graphs = pickle.load(f)
#
#         # If num_graphs is specified, take only the first `num_graphs` graphs
#         if self.num_graphs is not None:
#             nx_graphs = nx_graphs[:self.num_graphs]  # ✅ Take only the first `num_graphs` graphs
#
#         data_list = []
#         for nx_graph in nx_graphs:
#             data = self.convert_to_pyg_data(nx_graph)
#             data_list.append(data)
#
#         # ✅ Fix: Use torch.save() instead of self.collate()
#         torch.save(data_list, self.processed_paths[0])  # ✅ Now correctly saves the processed dataset
#
#     def convert_to_pyg_data(self, nx_graph):
#         node_features = []
#         for n in nx_graph.nodes():
#             node_type = np.array(nx_graph.nodes[n]['type'], dtype=np.float32)
#             node_feature = np.array([nx_graph.nodes[n]['feature']], dtype=np.float32)
#             full_feature = np.concatenate([node_type, node_feature])
#             node_features.append(full_feature)
#
#         x = torch.tensor(np.array(node_features), dtype=torch.float)
#
#         edge_index = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()
#
#         # 🔍 Debug edge attributes
#         edge_attr_list = [nx_graph[u][v]['type'] for u, v in nx_graph.edges]
#         edge_attr = torch.tensor(np.array(edge_attr_list, dtype=np.float32), dtype=torch.float)
#
#         return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#
#
#     def len(self):
#         """Returns the number of AIGs in the dataset."""
#         with torch.serialization.safe_globals(["torch_geometric.data.Data"]):  # ✅ Fix for PyG Data object
#             data_list = torch.load(self.processed_paths[0], weights_only=False)  # ✅ Ensure full data loading
#         return len(data_list)
#
#     def get(self, idx):
#         """Gets the AIG at the specified index."""
#         with torch.serialization.safe_globals(["torch_geometric.data.Data"]):  # ✅ Fix for PyG Data object
#             data_list = torch.load(self.processed_paths[0], weights_only=False)  # ✅ Ensure full data loading
#         return data_list[idx]


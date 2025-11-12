"""
Simple GraphML dataset loader for cycle detection.
Creates basic torch geometric objects with just nodes and edges, no features.
"""
import os
import os.path as osp
import glob
from typing import Optional, Callable, List
import logging

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
from torch_geometric.graphgym.config import cfg

ALG_OFFSETS = {
    'er': 0,
    'ba': 500,
    'sbm': 1000,
    'sfn': 1500,
    'complete': 2000,
    'star': 2500,
    'path': 3000
}

class SimpleGraphMLDataset(InMemoryDataset):
    """
    Simple dataset class for loading GraphML files without node features.
    
    This dataset creates basic torch geometric objects containing only:
    - Node indices
    - Edge indices
    - Graph labels (for cycle detection or custom labels)
    
    Args:
        root (str): Root directory where the dataset should be saved.
        graphml_dir (str): Directory containing GraphML files.
        label_map (dict, optional): Dictionary mapping filename to label.
        test_files (list, optional): List of filenames to use as test set.
        test_dir (str, optional): Directory containing test GraphML files (alternative to test_files).
        transform (callable, optional): A function/transform for the data objects.
        pre_transform (callable, optional): A function/transform for preprocessing.
        pre_filter (callable, optional): A function to filter data objects.
        train_ratio (float): Ratio for training split (applied to non-test files).
        val_ratio (float): Ratio for validation split (applied to non-test files).
        random_seed (int): Random seed for splits.
    """
    
    def __init__(
        self,
        root: str,
        graph_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        train_ratio: float = 0.8, 
        random_seed: int = 42
    ):
        self.graph_dir = graph_dir
        self.label_dir = label_dir
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        self.random_seed = random_seed

        # Ensure train/test ratios sum to 1 (val is handled separately)
        assert 0.0 < train_ratio < 1.0, "train_ratio must be less than 1.0 and greater than 0.0"
        
        super().__init__(root)
        # Load processed dataset. If processed files were created by an
        # earlier run that didn't include node features, `data.x` can be
        # missing (None). In that case, re-run `process()` to regenerate
        # processed files (which will include node-index features we add
        # in `create_graph_object`). This avoids returning batches with
        # `batch.x is None` due to stale cache.
        self._data, self.slices = torch.load(self.processed_paths[0])
        self.split_idxs = torch.load(self.processed_paths[1])

        # If loaded processed data has no node features or no edge features,
        # either reprocess or populate them in-memory so batches won't have
        # None attributes that break downstream layers (e.g., GatedGCN).
        missing_x = not hasattr(self._data, 'x') or self._data.x is None
        missing_e = not hasattr(self._data, 'edge_attr') or self._data.edge_attr is None
        if missing_x or missing_e:
            logging.warning(
                "Processed data missing node or edge features (data.x or data.edge_attr is None). "
                "Attempting to populate missing attributes in-memory.")
            try:
                # Populate node features if missing
                if missing_x:
                    total_nodes = int(self._data.num_nodes)
                    self._data.x = torch.arange(total_nodes, dtype=torch.float).unsqueeze(1)

                # Populate edge attributes if missing
                if missing_e:
                    total_edges = int(self._data.edge_index.size(1))
                    edge_dim = getattr(cfg.gnn, 'dim_inner', 1)
                    self._data.edge_attr = torch.zeros((total_edges, edge_dim), dtype=torch.float)

                # Also save back to processed files for consistency next run
                data, slices = self._data, self.slices
                torch.save((data, slices), self.processed_paths[0])
                torch.save(self.split_idxs, self.processed_paths[1])
                # Reload into the usual attributes
                self.data, self.slices = torch.load(self.processed_paths[0])
                self.split_idxs = torch.load(self.processed_paths[1])
            except Exception as e:
                logging.error(f"Failed to populate/re-save dataset attributes: {e}")
                raise
    
    @property
    def train_graph_names(self) -> List[str]:
        """Return list of training GraphML files."""
        if not osp.exists(self.train_graph_dir):
            return []
        return [f for f in os.listdir(self.train_graph_dir) if f.endswith('.graphml')]
    
    @property
    def test_graph_names(self) -> List[str]:
        """Return list of test GraphML files."""
        if not osp.exists(self.test_graph_dir):
            return []
        return [f for f in os.listdir(self.test_graph_dir) if f.endswith('.graphml')]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_idxs.pt']

    @property
    def raw_file_names(self) -> List[str]:
        """No raw files are required/downloaded for this dataset.

        The dataset reads GraphML files directly from a local directory structure
        (provided via `graph_dir`), so we return an empty list to satisfy the
        PyG Dataset API and avoid triggering a download step.
        """
        return []
    
    def download(self):
        """No download required for local GraphML files."""
        if not osp.exists(self.graph_dir) or not osp.isdir(self.label_dir):
            raise FileNotFoundError(f"GraphML directory not found: {self.graph_dir} or label directory not found: {self.label_dir}")
        logging.info(f"Using GraphML files from: {self.graph_dir}")

    def process(self):
        """Process GraphML files into simple PyTorch Geometric Data objects."""
        data_list = []
        test_idx = []
        train_val_idx = []
        curr_idx = 0

        train_graph_files, test_graph_files = {}, {}
        assert self.graph_dir and osp.exists(self.graph_dir), "Graph directory not found"
        algs = [a for a in os.listdir(self.graph_dir) if osp.isdir(osp.join(self.graph_dir, a))]
        for alg in algs:
            train_path = osp.join(self.graph_dir, alg, "train")
            test_path = osp.join(self.graph_dir, alg, "test")
            assert osp.exists(train_path), f"Train directory {train_path} not found"
            assert osp.exists(test_path), f"Test directory {test_path} not found"
            train_graph_files[alg] = glob.glob(osp.join(train_path, "*.graphml"))
            test_graph_files[alg] = glob.glob(osp.join(test_path, "*.graphml"))

        train_label_path = os.path.join(self.label_dir, "train")
        test_label_path = os.path.join(self.label_dir, "test")
        train_labels = {}
        test_labels = {}
        for alg in algs:
            train_labels[alg] = self.obtain_labels(train_label_path, alg)
            test_labels[alg] = self.obtain_labels(test_label_path, alg)

        if not train_graph_files and not test_graph_files:
            raise ValueError(f"No GraphML files found in {self.graph_dir}")

        for alg in algs:
            for graphml_file in train_graph_files[alg]:
                filename = osp.basename(graphml_file).replace('.graphml', '')
                alg_offset = ALG_OFFSETS[alg]
                file_index = int(filename) + alg_offset
                extended_filename = f"{alg}_train_{file_index}"
                try:
                    label = train_labels[alg][extended_filename]
                except KeyError as e:
                    print('KeyError:', extended_filename)
                    raise e
                data_list.append(self.create_graph_object(graphml_file, label))
                train_val_idx.append(curr_idx)
                curr_idx += 1
                
            for graphml_file in test_graph_files[alg]:
                filename = osp.basename(graphml_file).replace('.graphml', '')
                alg_offset = ALG_OFFSETS[alg]
                file_index = int(filename) + alg_offset
                extended_filename = f"{alg}_test_{file_index}"
                try:
                    label = test_labels[alg][extended_filename]
                except KeyError as e:
                    print('KeyError:', extended_filename)
                    raise e
                data_list.append(self.create_graph_object(graphml_file, label))
                test_idx.append(curr_idx)
                curr_idx += 1
        
        if not data_list:
            raise ValueError("No valid graphs were processed from the GraphML files")
        
        logging.info(f"Successfully processed {len(data_list)} graphs")
        
        # Print label distribution
        labels = [data.y.item() for data in data_list]
        unique_labels, counts = torch.unique(torch.tensor(labels), return_counts=True)
        logging.info(f"Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        
        logging.info(f"Test files specified: {len(test_idx)} graphs")
        logging.info(f"Train/Val files: {len(train_val_idx)} graphs")
        
        # Split the remaining files into train and validation
        if len(train_val_idx) > 1:
            # Check if we have enough samples for stratified splitting in train/val
            train_val_labels = [labels[i] for i in train_val_idx]
            unique_labels_tv, counts_tv = torch.unique(torch.tensor(train_val_labels), return_counts=True)
            min_count_tv = counts_tv.min().item()
            use_stratify = min_count_tv >= 2 and len(train_val_idx) >= 4  # Need at least 2 per class and 4 total
            
            if use_stratify:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    train_size=self.train_ratio,
                    random_state=self.random_seed,
                    stratify=train_val_labels
                )
            else:
                logging.warning(f"Not enough samples for stratified splitting in train/val (min_count={min_count_tv}), using random splits")
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    train_size=self.train_ratio,
                    random_state=self.random_seed
                )
        else:
            raise ValueError("Not enough training/validation samples to perform split")
        
        # Convert to tensors
        split_idxs = {
            'train': torch.tensor(train_idx),
            'val': torch.tensor(val_idx), 
            'test': torch.tensor(test_idx)
        }
        
        logging.info(f"Dataset splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(split_idxs, self.processed_paths[1])
    
    def get_idx_split(self):
        """Return the train/val/test split indices."""
        return {
            'train': self.split_idxs['train'],
            'valid': self.split_idxs['val'],  # Note: GraphGPS expects 'valid', not 'val'
            'test': self.split_idxs['test']
        }

    def obtain_labels(self, path: str, alg: str) -> dict:
        result = {}
        with open(os.path.join(path, f"{alg}.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                graph_id = parts[0]
                label = parts[1]
                result[graph_id] = int(label)

        return result
    
    def create_graph_object(self, graphml_file: str, label: int) -> Data:
        """Create a PyTorch Geometric Data object from a GraphML file."""
        # Load graph with NetworkX
        G = nx.read_graphml(graphml_file)
        
        # Convert to undirected if needed
        if G.is_directed():
            G = G.to_undirected()
        
        # Convert to PyTorch Geometric Data object
        data = from_networkx(G)
        
        # Ensure we have num_nodes
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            data.num_nodes = G.number_of_nodes()
        
        # Use node index as the only node feature when none are present.
        # This creates a single-column feature tensor with values [0, 1, ..., N-1].
        if getattr(data, 'x', None) is None:
            data.x = torch.arange(data.num_nodes, dtype=torch.float).unsqueeze(1)

        # Ensure edge attributes exist (some local MPNNs expect edge_attr).
        # Create zero edge features of size cfg.gnn.dim_inner if missing so
        # layers such as GatedGCN (which expect edge_attr) can operate.
        if getattr(data, 'edge_attr', None) is None:
            num_edges = data.edge_index.size(1)
            edge_dim = getattr(cfg.gnn, 'dim_inner', 1)
            data.edge_attr = torch.zeros((num_edges, edge_dim), dtype=torch.float)

        # Set label
        data.y = torch.tensor([label], dtype=torch.long)
        
        return data

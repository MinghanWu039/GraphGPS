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
        auto_cycle_detection (bool): If True, automatically detect cycles for labels.
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
        graphml_dir: str,
        label_map: Optional[dict] = None,
        auto_cycle_detection: bool = True,
        test_files: Optional[List[str]] = None,
        test_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        train_ratio: float = 0.8,  # Default to 80% train, 20% val for non-test files
        val_ratio: float = 0.2,
        random_seed: int = 42
    ):
        self.graphml_dir = graphml_dir
        self.label_map = label_map or {}
        self.auto_cycle_detection = auto_cycle_detection
        self.test_dir = test_dir
        self.test_files = set(test_files or [])  # Convert to set for faster lookup
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # If test_dir is provided, get all GraphML files from that directory
        if self.test_dir and osp.exists(self.test_dir):
            test_files_from_dir = [f for f in os.listdir(self.test_dir) if f.endswith('.graphml')]
            self.test_files.update(test_files_from_dir)
            logging.info(f"Found {len(test_files_from_dir)} test files in {self.test_dir}")
        elif self.test_dir:
            logging.warning(f"Test directory {self.test_dir} not found, ignoring")
        
        # Ensure train/val ratios sum to 1 (test is handled separately)
        total_ratio = train_ratio + val_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Train and val ratios must sum to 1.0, got {total_ratio}")
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.split_idxs = torch.load(self.processed_paths[1])
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of GraphML files."""
        if not osp.exists(self.graphml_dir):
            return []
        return [f for f in os.listdir(self.graphml_dir) if f.endswith('.graphml')]
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_idxs.pt']
    
    def download(self):
        """No download required for local GraphML files."""
        if not osp.exists(self.graphml_dir):
            raise FileNotFoundError(f"GraphML directory not found: {self.graphml_dir}")
        logging.info(f"Using GraphML files from: {self.graphml_dir}")
    
    def process(self):
        """Process GraphML files into simple PyTorch Geometric Data objects."""
        data_list = []
        
        # Collect GraphML files from main directory
        graphml_files = glob.glob(osp.join(self.graphml_dir, "*.graphml"))
        
        # Also collect GraphML files from test directory if specified
        test_graphml_files = []
        if self.test_dir and osp.exists(self.test_dir):
            test_graphml_files = glob.glob(osp.join(self.test_dir, "*.graphml"))
        
        # Combine all files
        all_graphml_files = graphml_files + test_graphml_files
        
        if not all_graphml_files:
            raise ValueError(f"No GraphML files found in {self.graphml_dir}" + 
                           (f" or {self.test_dir}" if self.test_dir else ""))
        
        logging.info(f"Processing {len(all_graphml_files)} GraphML files...")
        logging.info(f"  Main directory: {len(graphml_files)} files")
        if test_graphml_files:
            logging.info(f"  Test directory: {len(test_graphml_files)} files")
        
        for i, graphml_file in enumerate(all_graphml_files):
            try:
                # Load graph with NetworkX
                G = nx.read_graphml(graphml_file)
                
                # Convert to undirected if needed (most GNN methods work better with undirected graphs)
                if G.is_directed():
                    G = G.to_undirected()
                
                # Convert to PyTorch Geometric Data object
                data = from_networkx(G)
                
                # Remove any attributes that from_networkx might have added
                # We only want nodes and edges
                if hasattr(data, 'x'):
                    delattr(data, 'x')
                if hasattr(data, 'edge_attr'):
                    delattr(data, 'edge_attr')
                
                # Ensure we have num_nodes
                if not hasattr(data, 'num_nodes') or data.num_nodes is None:
                    data.num_nodes = G.number_of_nodes()
                
                # Set label
                filename = osp.basename(graphml_file)
                if filename in self.label_map:
                    label = self.label_map[filename]
                elif self.auto_cycle_detection:
                    # Automatically detect if the graph has cycles
                    label = 1 if not nx.is_forest(G) else 0  # 1 if has cycles, 0 if acyclic
                else:
                    # Try to extract label from filename pattern (e.g., class_1.graphml -> label 1)
                    try:
                        if '_' in filename:
                            parts = filename.split('_')
                            # Look for numeric parts
                            for part in parts:
                                try:
                                    label = int(part.split('.')[0])
                                    break
                                except ValueError:
                                    continue
                            else:
                                label = 0  # Default if no numeric part found
                        else:
                            label = 0  # Default label
                    except ValueError:
                        label = i % 2  # Alternate between 0 and 1 as dummy labels
                
                data.y = torch.tensor([label], dtype=torch.long)
                
                data_list.append(data)
                
            except Exception as e:
                logging.error(f"Error processing {graphml_file}: {str(e)}")
                continue
        
        if not data_list:
            raise ValueError("No valid graphs were processed from the GraphML files")
        
        logging.info(f"Successfully processed {len(data_list)} graphs")
        
        # Print label distribution
        labels = [data.y.item() for data in data_list]
        unique_labels, counts = torch.unique(torch.tensor(labels), return_counts=True)
        logging.info(f"Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        
        # Create train/val/test splits based on specified test files
        num_graphs = len(data_list)
        
        # Separate test indices from train/val indices
        test_idx = []
        train_val_idx = []
        
        for i, graphml_file in enumerate(all_graphml_files):
            filename = osp.basename(graphml_file)
            # Check if file is from test directory or in test_files list
            is_test_file = (filename in self.test_files or 
                           (self.test_dir and graphml_file.startswith(self.test_dir)))
            
            if is_test_file:
                test_idx.append(i)
            else:
                train_val_idx.append(i)
        
        logging.info(f"Test files specified: {len(test_idx)} graphs")
        logging.info(f"Train/Val files: {len(train_val_idx)} graphs")
        
        if len(train_val_idx) == 0:
            raise ValueError("No files available for training/validation after removing test files")
        
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
            # Only one file for train/val, put it in train
            train_idx = train_val_idx
            val_idx = []
            logging.warning("Only one file available for train/val, putting it in training set")
        
        # Convert to tensors
        split_idxs = {
            'train': torch.tensor(train_idx),
            'val': torch.tensor(val_idx), 
            'test': torch.tensor(test_idx)
        }
        
        logging.info(f"Dataset splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

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
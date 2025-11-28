"""
GraphML dataset loader for shortest-path node classification.
Each graph comes with a designated start node and per-node target distances.
"""
import os
import os.path as osp
import glob
import json
from typing import Optional, Callable, List, Dict, Any
import logging

import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split

INITIAL_DIST = 1000.0

class ShortestPathsGraphMLDataset(InMemoryDataset):
    """
    Dataset class for loading GraphML graphs paired with shortest-path labels.

    Each graph is accompanied by a JSON file containing:
        - start/target node identifier
        - dictionary of node_id -> shortest-path distance from start node

    The resulting PyG `Data` objects include:
        - edge_index / edge_attr
        - node features with the original features plus a start-node indicator column
        - per-node targets (`data.y`) storing the shortest-path distances (or -1 for unreachable)
        - `data.start_node` (LongTensor) with the index of the sampled start node

    Args mirror PyG datasets, with `graph_dir` and `label_dir` pointing to the
    directory trees containing `.graphml` and `.json` files respectively.
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
        random_seed: int = 42,
    ):
        self.graph_dir = graph_dir
        self.label_dir = label_dir
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        self.random_seed = random_seed

        # Ensure train/test ratios sum to 1 (val is handled separately)
        assert 0.0 < train_ratio < 1.0, "train_ratio must be less than 1.0 and greater than 0.0"
        
        logging.info(
            "Initializing ShortestPathsGraphMLDataset root=%s graph_dir=%s label_dir=%s",
            root,
            graph_dir,
            label_dir,
        )
        root = os.path.join(root, "shortest-paths")
        super().__init__(root)
        # Load processed dataset. If processed files were created by an
        # earlier run that didn't include node features, `data.x` can be
        # missing (None). In that case, re-run `process()` to regenerate
        # processed files (which will include node-index features we add
        # in `create_graph_object`). This avoids returning batches with
        # `batch.x is None` due to stale cache.
        self._data, self.slices = torch.load(self.processed_paths[0])
        self.split_idxs = torch.load(self.processed_paths[1])
        # self._ensure_consistent_cache()

        # If loaded processed data has no node features or no edge features,
        # either reprocess or populate them in-memory so batches won't have
        # None attributes that break downstream layers (e.g., GatedGCN).
        # missing_x = not hasattr(self._data, 'x') or self._data.x is None
        # missing_e = not hasattr(self._data, 'edge_attr') or self._data.edge_attr is None
        # if missing_x or missing_e:
        #     logging.warning(
        #         "Processed data missing node or edge features (data.x or data.edge_attr is None). "
        #         "Attempting to populate missing attributes in-memory.")
        #     try:
        #         # Populate node features if missing
        #         if missing_x:
        #             total_nodes = int(self._data.num_nodes)
        #             self._data.x = torch.arange(total_nodes, dtype=torch.float).unsqueeze(1)

        #         # Populate edge attributes if missing
        #         if missing_e:
        #             total_edges = int(self._data.edge_index.size(1))
        #             edge_dim = self.dim_inner
        #             self._data.edge_attr = torch.zeros((total_edges, edge_dim), dtype=torch.float)

        #         # Also save back to processed files for consistency next run
        #         data, slices = self._data, self.slices
        #         torch.save((data, slices), self.processed_paths[0])
        #         torch.save(self.split_idxs, self.processed_paths[1])
        #         # Reload into the usual attributes
        #         self.data, self.slices = torch.load(self.processed_paths[0])
        #         self.split_idxs = torch.load(self.processed_paths[1])
        #     except Exception as e:
        #         logging.error(f"Failed to populate/re-save dataset attributes: {e}")
        #         raise
    
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
        """Process GraphML files plus JSON labels into PyG Data objects."""
        data_list: List[Data] = []
        test_idx: List[int] = []
        train_val_idx: List[int] = []
        curr_idx = 0

        assert self.graph_dir and osp.exists(self.graph_dir), "Graph directory not found"
        assert self.label_dir and osp.exists(self.label_dir), "Label directory not found"

        algs = sorted([a for a in os.listdir(self.graph_dir) if osp.isdir(osp.join(self.graph_dir, a))])
        if not algs:
            raise ValueError(f"No algorithm subdirectories found under {self.graph_dir}")

        for alg in algs:
            logging.info("Processing algorithm bucket '%s'", alg)
            for split in ("train", "test"):
                graph_split_dir = osp.join(self.graph_dir, alg, split)
                label_split_dir = osp.join(self.label_dir, alg, split)
                if not osp.exists(graph_split_dir):
                    raise FileNotFoundError(f"Missing graph split directory: {graph_split_dir}")
                if not osp.exists(label_split_dir):
                    raise FileNotFoundError(f"Missing label split directory: {label_split_dir}")

                graph_files = sorted(glob.glob(osp.join(graph_split_dir, "*.graphml")))
                if not graph_files:
                    logging.warning(f"No GraphML files found in {graph_split_dir}")
                    continue

                logging.info(
                    "  Split '%s': %d graphs | labels dir=%s",
                    split,
                    len(graph_files),
                    label_split_dir,
                )

                for graphml_file in graph_files:
                    label_path = self._label_path_for(graphml_file)
                    if not osp.exists(label_path):
                        raise FileNotFoundError(
                            f"Missing JSON label for {graphml_file}: expected {label_path}"
                        )
                    label_payload = self._load_label_payload(label_path)
                    logging.debug("    Converting %s with label %s", graphml_file, label_path)
                    data_obj = self.create_graph_object(graphml_file, label_payload)
                    data_list.append(data_obj)
                    if split == "train":
                        train_val_idx.append(curr_idx)
                    else:
                        test_idx.append(curr_idx)
                    curr_idx += 1

        if not data_list:
            raise ValueError("No valid graphs were processed from the GraphML files")

        logging.info(
            "Processed %d graphs (%d train/val, %d test)",
            len(data_list),
            len(train_val_idx),
            len(test_idx),
        )

        if len(train_val_idx) < 2:
            raise ValueError("Need at least two training graphs to create train/val split")

        train_idx, val_idx = train_test_split(
            train_val_idx,
            train_size=self.train_ratio,
            random_state=self.random_seed,
            shuffle=True,
        )

        split_idxs = {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'val': torch.tensor(val_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long),
        }

        self._assign_node_split_masks(data_list, train_idx, val_idx, test_idx)

        logging.info(
            "Dataset splits -> Train: %d | Val: %d | Test: %d",
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

        logging.info("Inspecting first graph object:")
        logging.info(data_list[0])

        logging.info("Collating PyG dataset with %d graphs", len(data_list))
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

    def _ensure_consistent_cache(self) -> None:
        """Verify cached data matches stored split indices; reprocess otherwise."""
        ref_attr = None
        for attr, slice_tensor in self.slices.items():
            if torch.is_tensor(slice_tensor) and slice_tensor.numel() > 0:
                ref_attr = attr
                break

        if ref_attr is None:
            logging.warning("Unable to validate cached dataset; no slice tensors found.")
            return

        num_graphs = int(self.slices[ref_attr].numel() - 1)
        required_graphs = self._max_required_index(self.split_idxs) + 1

        if required_graphs > num_graphs:
            logging.warning(
                "Split indices reference %d graphs but only %d are cached. "
                "Reprocessing dataset to refresh cache.",
                required_graphs,
                num_graphs,
            )
            self.process()
            self._data, self.slices = torch.load(self.processed_paths[0])
            self.split_idxs = torch.load(self.processed_paths[1])
        else:
            logging.debug(
                "Cache validation OK: %d graphs available, %d referenced.",
                num_graphs,
                required_graphs,
            )

    @staticmethod
    def _max_required_index(split_dict: Dict[str, torch.Tensor]) -> int:
        """Return the maximum graph index referenced by the split tensors."""
        max_idx = -1
        for key in ('train', 'val', 'test'):
            split_tensor = split_dict.get(key)
            if split_tensor is None or split_tensor.numel() == 0:
                continue
            max_idx = max(max_idx, int(split_tensor.max().item()))
        return max_idx

    def _assign_node_split_masks(
        self,
        graphs: List[Data],
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
    ) -> None:
        """Attach boolean masks to each graph for node-level heads."""
        graph_split_lookup = {
            idx: 'train' for idx in train_idx
        }
        graph_split_lookup.update({idx: 'val' for idx in val_idx})
        graph_split_lookup.update({idx: 'test' for idx in test_idx})

        for graph_idx, graph in enumerate(graphs):
            split_name = graph_split_lookup.get(graph_idx)
            graph.train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
            graph.val_mask = torch.zeros_like(graph.train_mask)
            graph.test_mask = torch.zeros_like(graph.train_mask)

            if split_name == 'train':
                graph.train_mask[:] = True
            elif split_name == 'val':
                graph.val_mask[:] = True
            elif split_name == 'test':
                graph.test_mask[:] = True

    def _label_path_for(self, graphml_file: str) -> str:
        rel_path = osp.relpath(graphml_file, self.graph_dir)
        rel_root, _ = osp.splitext(rel_path)
        return osp.join(self.label_dir, rel_root + ".json")

    @staticmethod
    def _canonical_node_id(node_id: Any) -> str:
        """Normalize node identifiers to string keys for easy comparison."""
        return str(node_id)

    def _load_label_payload(self, label_path: str) -> Dict[str, Any]:
        with open(label_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)

        if "target_node" not in payload or "distances" not in payload:
            raise ValueError(f"Invalid label payload (missing keys) in {label_path}")
        return payload

    def create_graph_object(self, graphml_file: str, label_payload: Dict[str, Any]) -> Data:
        """Create a PyTorch Geometric Data object from a GraphML file plus label payload."""
        # Load graph with NetworkX
        G = nx.read_graphml(graphml_file)
        
        # Convert to undirected if needed
        if G.is_directed():
            G = G.to_undirected()

        node_order = list(G.nodes())
        node_to_idx = {self._canonical_node_id(node): idx for idx, node in enumerate(node_order)}
        
        # Convert to PyTorch Geometric Data object
        data = from_networkx(G)
        
        # Ensure we have num_nodes
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            data.num_nodes = G.number_of_nodes()
        
        # Use node index as the base node feature when none are present.
        if getattr(data, 'x', None) is None:
            data.x = torch.arange(data.num_nodes, dtype=torch.float).unsqueeze(1)
        else:
            data.x = data.x.to(torch.float)

        start_node_id = self._canonical_node_id(label_payload["target_node"])
        if start_node_id not in node_to_idx:
            raise ValueError(
                f"Start node {start_node_id} not found in graph {graphml_file}"
            )
        start_idx = node_to_idx[start_node_id]
        logging.debug(
            "Graph %s start node %s -> idx %d (num_nodes=%d)",
            graphml_file,
            start_node_id,
            start_idx,
            data.num_nodes,
        )

        distances = label_payload.get("distances", {})
        y = torch.full((data.num_nodes,), fill_value=-1, dtype=torch.long)
        for node_id, distance in distances.items():
            canonical_id = self._canonical_node_id(node_id)
            node_idx = node_to_idx.get(canonical_id)
            if node_idx is None:
                logging.debug(
                    "Node id %s from label missing in graph %s", canonical_id, graphml_file
                )
                continue
            if distance is None:
                y[node_idx] = -1
            else:
                y[node_idx] = int(distance)

        data.y = y
        data.start_node = torch.tensor([start_idx], dtype=torch.long)

        # Augment node features with start-node indicator feature.
        start_feature = torch.full((data.num_nodes, 1), INITIAL_DIST, dtype=torch.float)
        start_feature[start_idx, 0] = 0.0
        data.x = torch.cat([data.x, start_feature], dim=1)

        if getattr(data, 'edge_attr', None) is None:
            num_edges = data.edge_index.size(1)
            edge_dim = 1
            data.edge_attr = torch.zeros((num_edges, edge_dim), dtype=torch.float)
        
        return data

def __getitem__(self, idx):
    # Let the parent handle slices, lists, tensors of indices, etc.
    if isinstance(idx, slice) or isinstance(idx, (list, tuple, torch.Tensor)):
        return super().__getitem__(idx)

    # Now idx is a single integer index -> we can safely debug
    idx_int = int(idx)

    # DEBUG: check for any bad slices
    for key, sl in self.slices.items():
        if torch.is_tensor(sl) and sl.numel() <= idx_int + 1:
            print(f"[BAD] key={key!r}, slices.shape={tuple(sl.shape)}, len={sl.numel()}")

    return super().__getitem__(idx_int)

#!/usr/bin/env python3
"""
Generate simple test GraphML files for cycle detection and test the loader.
"""
import os
import sys
import networkx as nx
import numpy as np

def create_simple_test_graphs(output_dir="simple_test_graphs", num_graphs=20):
    """Create simple test GraphML files in separate train and test directories."""
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # Split graphs between train and test
    num_test = num_graphs // 4  # 25% for test
    num_train = num_graphs - num_test
    
    # Create training graphs
    for i in range(num_train):
    # Create training graphs
    for i in range(num_train):
        if i < num_train // 2:
            # Create acyclic graphs (trees)
            n_nodes = np.random.randint(5, 15)
            # Create a simple tree structure (star or path)
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            
            if n_nodes > 1:
                if np.random.random() < 0.5:
                    # Create a star graph (tree)
                    center = 0
                    for j in range(1, n_nodes):
                        G.add_edge(center, j)
                else:
                    # Create a path graph (tree)
                    for j in range(n_nodes - 1):
                        G.add_edge(j, j + 1)
            
            has_cycle = False
            filename = f"train_acyclic_{i:03d}.graphml"
            save_dir = train_dir
        else:
            # Create graphs with cycles
            n_nodes = np.random.randint(5, 15)
            # Create a cycle and add some extra nodes
            cycle_length = min(np.random.randint(3, 6), n_nodes)
            G = nx.cycle_graph(cycle_length)
            
            # Add remaining nodes and connect them
            for node in range(cycle_length, n_nodes):
                G.add_node(node)
                # Connect to a random existing node
                connect_to = np.random.choice(list(G.nodes())[:-1])
                G.add_edge(node, connect_to)
            
            has_cycle = True
            filename = f"train_cyclic_{i:03d}.graphml"
            save_dir = train_dir
        
        # Verify our cycle detection
        actual_has_cycle = not nx.is_forest(G)
        assert actual_has_cycle == has_cycle, f"Cycle detection mismatch for {filename}"
        
        # Save to GraphML
        filepath = os.path.join(save_dir, filename)
        nx.write_graphml(G, filepath)
        
        print(f"Created {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, has_cycle={has_cycle}")
    
    # Create test graphs
    for i in range(num_test):
        if i < num_test // 2:
            # Create acyclic test graphs
            n_nodes = np.random.randint(5, 15)
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            
            if n_nodes > 1:
                # Create a path graph (tree)
                for j in range(n_nodes - 1):
                    G.add_edge(j, j + 1)
            
            has_cycle = False
            filename = f"test_acyclic_{i:03d}.graphml"
        else:
            # Create cyclic test graphs
            n_nodes = np.random.randint(5, 15)
            cycle_length = min(np.random.randint(3, 6), n_nodes)
            G = nx.cycle_graph(cycle_length)
            
            # Add remaining nodes
            for node in range(cycle_length, n_nodes):
                G.add_node(node)
                connect_to = np.random.choice(list(G.nodes())[:-1])
                G.add_edge(node, connect_to)
            
            has_cycle = True
            filename = f"test_cyclic_{i:03d}.graphml"
        
        # Verify cycle detection
        actual_has_cycle = not nx.is_forest(G)
        assert actual_has_cycle == has_cycle, f"Cycle detection mismatch for {filename}"
        
        # Save to test directory
        filepath = os.path.join(test_dir, filename)
        nx.write_graphml(G, filepath)
        
        print(f"Created {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, has_cycle={has_cycle}")
    
    print(f"\nGenerated {num_graphs} graphs in '{output_dir}'")
    print(f"  Training graphs: {num_train} in {train_dir}")
    print(f"  Test graphs: {num_test} in {test_dir}")
    return output_dir, train_dir, test_dir

def test_simple_loader():
    """Test the simple GraphML loader."""
    print("\n=== Testing Simple GraphML Loader ===")
    
    # Add project root to path
    sys.path.insert(0, '/dsmlp/home-fs03/95/495/miw039/DSC180A/GraphGPS')
    
    try:
        from graphgps.loader.dataset.simple_graphml import SimpleGraphMLDataset
        print("✓ Successfully imported SimpleGraphMLDataset")
    except ImportError as e:
        print(f"✗ Failed to import SimpleGraphMLDataset: {e}")
        return False
    
    # Generate test data
    base_dir, train_dir, test_dir = create_simple_test_graphs(num_graphs=20)
    print(f"Test directory: {test_dir}")
    
    # Test dataset loading with test_dir
    try:
        dataset = SimpleGraphMLDataset(
            root="test_simple_processed",
            graphml_dir=train_dir,  # Training files
            test_dir=test_dir,      # Test files directory
            auto_cycle_detection=True
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Number of graphs: {len(dataset)}")
        print(f"  Number of node features: {dataset.num_node_features}")
        print(f"  Number of edge features: {dataset.num_edge_features}")
        
        # Test a sample
        sample = dataset[0]
        print(f"  Sample graph - nodes: {sample.num_nodes}, edges: {sample.num_edges}")
        if hasattr(sample, 'x') and sample.x is not None:
            print(f"  Node features shape: {sample.x.shape}")
        else:
            print(f"  No node features (as expected)")
        print(f"  Label: {sample.y}")
        
        # Test splits
        splits = dataset.get_idx_split()
        print(f"  Train/Val/Test splits: {len(splits['train'])}/{len(splits['valid'])}/{len(splits['test'])}")
        
        # Verify cycle detection
        acyclic_count = 0
        cyclic_count = 0
        for i in range(len(dataset)):
            label = dataset[i].y.item()
            if label == 0:
                acyclic_count += 1
            else:
                cyclic_count += 1
        print(f"  Detected {acyclic_count} acyclic graphs, {cyclic_count} cyclic graphs")
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_loader()
    
    if success:
        # Create a config file for this test
        base_dir = "simple_test_graphs"
        train_dir = os.path.join(base_dir, "train")
        test_dir = os.path.join(base_dir, "test")
        
        config_content = f"""out_dir: results
metric_best: accuracy
wandb:
  use: False
dataset:
  format: PyG-SimpleGraphML
  name: simple_cycle_test
  task: graph
  task_type: classification
  transductive: False
  node_encoder: True
  node_encoder_name: TypeDictNode
  node_encoder_bn: False
  edge_encoder: False
  graphml_dir: {os.path.abspath(train_dir)}  # Training files directory
  test_dir: {os.path.abspath(test_dir)}      # Test files directory
  auto_cycle_detection: True
  train_ratio: 0.75  # Applied to training files for train/val split
  val_ratio: 0.25   # Applied to training files for train/val split
  split_seed: 42
posenc_LapPE:
  enable: True
  eigen:
    laplacian_norm: none
    eigvec_norm: L2
    max_freqs: 8
  model: DeepSet
  dim_pe: 8
  layers: 2
  raw_norm_type: none
train:
  mode: custom
  batch_size: 4
  eval_period: 1
  ckpt_period: 100
model:
  type: GPSModel
  loss_fun: cross_entropy
  edge_decoding: dot
  graph_pooling: mean
gt:
  layer_type: CustomGatedGCN+Transformer
  layers: 2
  n_heads: 4
  dim_hidden: 32
  dropout: 0.1
  attn_dropout: 0.1
  layer_norm: False
  batch_norm: True
gnn:
  head: default
  layers_pre_mp: 0
  layers_post_mp: 1
  dim_inner: 32
  batchnorm: True
  act: relu
  dropout: 0.0
  agg: mean
  normalize_adj: False
optim:
  clip_grad_norm: True
  optimizer: adamW
  weight_decay: 1e-5
  base_lr: 0.001
  max_epoch: 20
  scheduler: cosine_with_warmup
  num_warmup_epochs: 2"""
      
        config_path = "configs/GPS/test-simple-cycle-GPS.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"\nCreated test config file: {config_path}")
        print(f"\nTo test with GraphGPS, run:")
        print(f"python main.py --cfg {config_path}")
    
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Generate sample GraphML files for testing the custom GraphML dataset loader.
"""
import os
import networkx as nx
import numpy as np

def create_sample_graphs(output_dir="sample_graphml_data", num_graphs=20):
    """Create sample GraphML files for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    for i in range(num_graphs):
        # Create random graphs with different properties for two classes
        if i < num_graphs // 2:
            # Class 0: Smaller, more connected graphs
            n_nodes = np.random.randint(10, 20)
            p = 0.3  # Higher connection probability
            label = 0
            class_name = "class_0"
        else:
            # Class 1: Larger, less connected graphs
            n_nodes = np.random.randint(20, 35)
            p = 0.15  # Lower connection probability
            label = 1
            class_name = "class_1"
        
        # Generate random graph
        G = nx.erdos_renyi_graph(n_nodes, p)
        
        # Add node features
        for node in G.nodes():
            G.nodes[node]['degree'] = G.degree(node)
            G.nodes[node]['clustering'] = nx.clustering(G, node)
            G.nodes[node]['random_feature'] = np.random.normal(0, 1)
        
        # Add edge features  
        for edge in G.edges():
            G.edges[edge]['weight'] = np.random.uniform(0.1, 1.0)
            G.edges[edge]['edge_type'] = np.random.choice(['type_A', 'type_B'])
        
        # Save to GraphML
        filename = f"{class_name}_{i:03d}.graphml"
        filepath = os.path.join(output_dir, filename)
        nx.write_graphml(G, filepath)
        
        print(f"Created {filename}: {n_nodes} nodes, {G.number_of_edges()} edges, label={label}")
    
    print(f"\nGenerated {num_graphs} sample GraphML files in '{output_dir}'")
    print(f"Class 0: {num_graphs//2} graphs (smaller, more connected)")
    print(f"Class 1: {num_graphs//2} graphs (larger, less connected)")
    
    return output_dir

if __name__ == "__main__":
    output_dir = create_sample_graphs()
    
    # Create a simple config file for this example
    config_content = f"""out_dir: results
metric_best: accuracy
wandb:
  use: False
dataset:
  format: PyG-GraphML
  name: sample_data
  task: graph
  task_type: classification
  transductive: False
  node_encoder: True
  node_encoder_name: LinearNode
  node_encoder_bn: False
  edge_encoder: True
  edge_encoder_name: LinearEdge
  edge_encoder_bn: False
  graphml_dir: {os.path.abspath(output_dir)}
  node_features: ['degree', 'clustering', 'random_feature']
  edge_features: ['weight']
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
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
  
    config_path = "configs/GPS/sample-graphml-GPS.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\nCreated config file: {config_path}")
    print(f"\nTo test the setup, run:")
    print(f"python main.py --cfg {config_path}")
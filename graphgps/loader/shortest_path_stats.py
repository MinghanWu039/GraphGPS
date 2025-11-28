#!/usr/bin/env python3
"""
Utility script to summarize the label distribution for the shortest-path dataset.

It instantiates `ShortestPathsGraphMLDataset`, iterates through the graphs, and
accumulates counts for every observed distance label (including -1 for
unreachable nodes). The counts can optionally be saved to a JSON file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt
import torch

from .dataset.shortest_paths_graphml import ShortestPathsGraphMLDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize shortest-path label distribution."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/ShortestPathInspection",
        help="Root directory where the processed dataset cache is (or will be) stored.",
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Directory containing the GraphML graphs (same as cfg.data_src.graphml_dir).",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing JSON labels (same as cfg.data_src.label_dir).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (must match the dataset you processed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when generating the dataset splits.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file to store the histogram {label: count}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ShortestPathsGraphMLDataset(
        root=args.root,
        graph_dir=args.graph_dir,
        label_dir=args.label_dir,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
    )

    label_counter: Counter[int] = Counter()
    total_nodes = 0
    for graph in dataset:
        labels = graph.y.view(-1).to(torch.long)
        total_nodes += labels.numel()
        label_counter.update(int(label.item()) for label in labels)

    print(f"Graphs analyzed : {len(dataset)}")
    print(f"Total nodes     : {total_nodes}")
    for label, count in sorted(label_counter.items()):
        print(f"Label {label:>3}: {count}")

    # Plot histogram of label counts to visualize distribution.
    labels_sorted, counts_sorted = zip(*sorted(label_counter.items()))
    plt.figure(figsize=(10, 5))
    plt.bar(labels_sorted, counts_sorted, width=0.8)
    plt.xlabel("Shortest-path label")
    plt.ylabel("Count")
    plt.title("Shortest-path label distribution")
    plt.tight_layout()
    plt.savefig("shortest_path_label_distribution.png")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "num_graphs": len(dataset),
                    "total_nodes": total_nodes,
                    "label_counts": dict(sorted(label_counter.items())),
                },
                fp,
                indent=2,
            )
        print(f"Wrote histogram to {args.output}")


if __name__ == "__main__":
    main()

"""
Graphs analyzed : 8000
Total nodes     : 95744
Label  -1: 469
Label   0: 8000
Label   1: 16456
Label   2: 27360
Label   3: 16791
Label   4: 6763
Label   5: 4293
Label   6: 3424
Label   7: 2826
Label   8: 2305
Label   9: 1852
Label  10: 1457
Label  11: 1154
Label  12: 872
Label  13: 653
Label  14: 463
Label  15: 307
Label  16: 175
Label  17: 98
Label  18: 26
"""

import os
from typing import Any, List

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data
from tqdm import tqdm

from src.data.base import BaseDataset

from torch_geometric.utils import k_hop_subgraph


def stratified_subsample(data, samples_per_class=50, num_hops=3):
    y = data.y
    print(y)
    classes = y.unique()
    sampled_nodes = []

    for cls in tqdm(classes, desc="Stratified Sampling"):
        cls_nodes = (y == cls).nonzero(as_tuple=True)[0]
        
        # seed the random number generator for reproducibility
        torch.manual_seed(42)
        
        perm = torch.randperm(cls_nodes.size(0))[:samples_per_class]
        sampled_nodes.extend(cls_nodes[perm].tolist())

    # Merge k-hop neighborhoods of sampled nodes
    subset, edge_index, _, mask = k_hop_subgraph(
        node_idx=torch.tensor(sampled_nodes),
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )
    
    print(f"Subset size: {subset.size()}")
    print(f"Edge index size: {edge_index.size()}")

    # Now build a new Data object manually
    new_data = Data()
    for key in data.keys():
        item = data[key]
        if torch.is_tensor(item) and item.size(0) == data.num_nodes:
            new_data[key] = item[subset]
        else:
            new_data[key] = item

    new_data.edge_index = edge_index
    
    return new_data


class Computers_for_transfer_Dataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        preprocessor_llm: Any = None,
        computers_dataset_root: str = os.path.expandvars('/scratch/$USER/datasets/computers'),  # Add this parameter
    ):
        if computers_dataset_root is None:
            # Default: assume computers dataset is in same parent directory
            parent_dir = os.path.dirname(root)
            self.computers_raw_path = os.path.join(parent_dir, "computers", "raw")
        else:
            self.computers_raw_path = os.path.join(computers_dataset_root, "raw")
    
        super().__init__(
            dataset_name="computers-for-transfer", 
            root=root,
            transform=transform,
            force_reload=force_reload,
        )

        self.type = "node"
        self.preprocessor_llm = preprocessor_llm
        

        self.load(self.processed_paths[1])
        self.graph_data = torch.load(self.processed_paths[0], weights_only=False)
        
    @property
    def processed_file_names(self) -> List[str]:
        return ["graph_data.pt", "data_list.pt"]

    def process(self):
        # Use the original computers dataset raw data
        raw_path = self.computers_raw_path
        assert os.path.exists(raw_path), f"Original computers raw data not found at {raw_path}"

        # Load from original computers dataset
        graph = torch.load(f"{raw_path}/pyg_graph.pt")
        
        graph.x = np.load(f"{raw_path}/Computers_roberta_base_512_cls.npy")
        # convert to torch tensor
        graph.x = torch.tensor(graph.x, dtype=torch.float32)
        graph.y = torch.tensor(graph.label, dtype=torch.long)
        
        del graph.label

        # Save to NEW dataset's processed directory
        torch.save(graph, self.processed_paths[0])

        # NEW class names for transfer learning
        class_names = [
            "computer accessories and peripherals",
            "tablet accessories", 
            "laptop accessories",
            "computers and tablets",
            "computer components",
            "data storage",
            "networking products",
            "monitors",
            "servers",
            "tablet replacement parts",

            #---photo labels----
            'Film Photography', 'Video', 'Digital Cameras', 'Accessories',
            'Binoculars & Scopes', 'Lenses', 'Bags & Cases', 'Lighting & Studio',
            'Flashes', 'Tripods & Monopods', 'Underwater Photography',
            'Video Surveillance',

            #----history labels----
            'World', 'Americas', 'Asia', 'Military', 'Europe', 'Russia', 'Africa',
            'Ancient Civilizations', 'Middle East',
            'Historical Study & Educational Resources', 'Australia & Oceania',
            'Arctic & Antarctica'
        ]

        data_list = []

        # NEW question for transfer learning
        QUESTION = f"Answer the following question: Which subcategory does this representation/description belong to? Please only output the most likely answer from the following subcategories and nothing else: {', '.join(class_names)} \n\n Answer:"

        # Load text data from original computers dataset
        text_path = os.path.join(raw_path, "text.csv")
        text_df = pd.read_csv(text_path, sep=",")

        for index, row in text_df.iterrows():
            # Create a Data object with NEW labels and question
            data_item = Data(
                node_id=int(row["node_id"]),
                label=class_names[row["label"]] if row["label"] < len(class_names) else "unknown",
                context=row["text"],
                question=QUESTION,
            )
            data_list.append(data_item)

        # Save to NEW dataset's processed directory
        self.save(data_list, self.processed_paths[1])

    def split(self, seed, split_ratio: List[float] = [0.6, 0.2, 0.2]):
        num_samples = len(self.graph_data.y)
        indices = np.random.RandomState(seed).permutation(num_samples)

        train_split = int(num_samples * split_ratio[0])
        val_split = int(num_samples * (split_ratio[0] + split_ratio[1]))

        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
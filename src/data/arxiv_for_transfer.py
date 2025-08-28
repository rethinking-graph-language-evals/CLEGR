import os
import csv
import torch

import numpy as np
import pandas as pd

from typing import Any, List
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from torch_geometric.transforms import IndexToMask
from torch_geometric.data.dataset import Dataset

from src.data.base import BaseDataset

class Arxiv_for_transfer_Dataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        preprocessor_llm: Any = None,
        arxiv_dataset_root: str = os.path.expandvars('/scratch/$USER/datasets/arxiv'), 
    ):

        # Store the path to the original arxiv dataset
        if arxiv_dataset_root is None:
            # Default: assume arxiv dataset is in same parent directory
            parent_dir = os.path.dirname(root)
            self.arxiv_root_path = os.path.join(parent_dir, "arxiv")
        else:
            self.arxiv_root_path = arxiv_dataset_root

        super().__init__(
            dataset_name="arxiv-for-transfer", 
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
        # Use the original arxiv dataset root path
        arxiv_raw_path = os.path.join(self.arxiv_root_path, "arxiv_raw")
        assert os.path.exists(arxiv_raw_path), f"Original arxiv raw data not found at {arxiv_raw_path}"

        # Load from original arxiv dataset location
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv",
            root=self.arxiv_root_path  # Use original arxiv dataset root
        )
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = dataset[0]
        graph.train_index = train_idx
        graph.val_index = valid_idx
        graph.test_index = test_idx
        
        # pca features to 500 if greater
        if graph.x.shape[1] > 500:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=500)
            new_x = pca.fit_transform(graph.x)
            graph.x = torch.tensor(new_x).to(torch.float)

        # Save to NEW dataset's processed directory
        torch.save(graph, self.processed_paths[0])

        common_y = graph.y

        # Get raw text features from original arxiv dataset
        with open(os.path.join(self.arxiv_root_path, "arxiv_raw/titleabs.tsv")) as f:
            csvreader = csv.reader(f, delimiter="\t")
            magid_to_titleabs = {int(row[0]): {"Title": row[1].strip(), "Abstract": row[2].strip()} for row in csvreader if len(row) == 3}
        
        nodeid_to_magid = pd.read_csv(os.path.join(self.arxiv_root_path, "ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"), compression="gzip", converters={0:int, 1:int})
        nodeid_to_magid = dict(zip(nodeid_to_magid['node idx'], nodeid_to_magid['paper id']))
        
        # Load the label-to-category mapping from original arxiv dataset
        label_to_category = pd.read_csv(os.path.join(self.arxiv_root_path, "ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz"), compression="gzip")

        def format_category(category):
            category = category.replace("arxiv ", "")
            category = category.replace(" ", ".")
            category = category.upper()
            return f"Arxiv {category}"
        label_to_category['formatted_category'] = label_to_category['arxiv category'].apply(format_category)

        original_arxiv_categories = list(label_to_category['formatted_category'])
        
        # NEW: Add Cora classes for transfer learning
        cora_classes = [
            "theory",
            "reinforcement learning", 
            "genetic algorithms",
            "neural networks",
            "probabilistic methods",
            "case based",
            "rule learning",
        ]
        
        # Combine original arxiv categories with cora classes
        all_categories = original_arxiv_categories + cora_classes
        
        # Create label mapping (keeping original arxiv labels intact)
        label_category_dict = dict(zip(label_to_category['label idx'], label_to_category['formatted_category']))

        # NEW question for transfer learning including both arxiv and cora categories
        QUESTION = f"Answer the following question: Which subcategory does this paper belong to? Please only output the most likely answer from the following subcategories and nothing else: {', '.join(all_categories)}.\nAnswer: "
        
        data_list = []
        for nodeid in range(graph.num_nodes):
            data_list.append(
                Data(
                    id=nodeid_to_magid[nodeid],
                    question=QUESTION,
                    context=magid_to_titleabs[nodeid_to_magid[nodeid]],
                    label=label_category_dict[common_y[nodeid].item()],
                    node_id=nodeid
                )
            )

        # Save to NEW dataset's processed directory
        self.save(data_list, self.processed_paths[1])
    
    def split(self, seed, split_ratio: List[float] = [0.6, 0.2, 0.2]):
        train_indices = self.graph_data.train_index
        val_indices = self.graph_data.val_index
        test_indices = self.graph_data.test_index

        return {
            "train": train_indices.numpy(force=True).tolist(),
            "val": val_indices.numpy(force=True).tolist(),
            "test": test_indices.numpy(force=True).tolist(),
        }
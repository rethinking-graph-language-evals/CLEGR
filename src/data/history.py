import os
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset

from torch_geometric.data import Data
from torch_geometric.datasets import Amazon

from src.data.base import BaseDataset

class HistoryDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        preprocessor_llm: Any = None,
    ):
        super().__init__(
            dataset_name="history",
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
        # assert raw data is present
        raw_path = os.path.join(self.root, "raw")
        assert os.path.exists(raw_path), f"Raw data not found at {raw_path}"

        graph = torch.load(f"{raw_path}/pyg_graph.pt")
        
        graph.x = np.load(f"{raw_path}/History_roberta_base_512_cls.npy")
        # convert to torch tensor
        graph.x = torch.tensor(graph.x, dtype=torch.float32)
            
        # graph = torch.zeros((2, 2))
        # # save graph data
        torch.save(graph, self.processed_paths[0])

        class_names = ['World', 'Americas', 'Asia', 'Military', 'Europe', 'Russia', 'Africa',
        'Ancient Civilizations', 'Middle East',
        'Historical Study & Educational Resources', 'Australia & Oceania',
        'Arctic & Antarctica']

        data_list = []

        QUESTION = f"Answer the following question: Which history related subcategory does this description belong to? Please only output the most likely answer from the following subcategories and nothing else: {', '.join(class_names)} \n\n Answer:"
        
        # load the csv containing text data
        text_path = os.path.join(raw_path, "/home/datasets/history/raw/History.csv")
        text_df = pd.read_csv(text_path, sep=',')
        
        for index, row in text_df.iterrows():
            # Create a Data object that contains all the necessary information for the example
            # print(len(class_names), row['label'])
            data_item = Data(
                node_id=int(row['node_id']),
                label=class_names[row['label']],
                context=row['text'],
                question=QUESTION,
            )
            data_list.append(data_item)

        # save data list
        self.save(data_list, self.processed_paths[1])
    
    def split(self, seed, split_ratio: List[float] = [0.6, 0.2, 0.2]):
        num_samples = len(self)
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
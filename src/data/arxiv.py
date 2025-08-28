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

class ArxivDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        preprocessor_llm: Any = None,
    ):
        super().__init__(
            dataset_name="arxiv",
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
        raw_path = os.path.join(self.root, "arxiv_raw")
        assert os.path.exists(raw_path), f"Raw data not found at {raw_path}"

        dataset = PygNodePropPredDataset(
            name = "ogbn-arxiv",
            root=self.root
        )
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = dataset[0]
        graph.train_index = train_idx
        graph.val_index = valid_idx
        graph.test_index = test_idx
        # idx2mask = IndexToMask()
        # graph = idx2mask(graph)
        
        # pca features to 500 if greater
        if graph.x.shape[1] > 500:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=500)
            new_x = pca.fit_transform(graph.x)
            graph.x = torch.tensor(new_x).to(torch.float)

        # save graph data
        torch.save(graph, self.processed_paths[0])

        common_y = graph.y

        # Get raw text features
        with open(os.path.join(self.root, "arxiv_raw/titleabs.tsv")) as f:
            csvreader = csv.reader(f, delimiter="\t")
            magid_to_titleabs = {int(row[0]): {"Title": row[1].strip(), "Abstract": row[2].strip()} for row in csvreader if len(row) == 3}
        
        nodeid_to_magid = pd.read_csv(os.path.join(self.root, "ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"), compression="gzip", converters={0:int, 1:int})
        nodeid_to_magid = dict(zip(nodeid_to_magid['node idx'], nodeid_to_magid['paper id']))
        
        # Load the label-to-category mapping
        label_to_category = pd.read_csv(os.path.join(self.root, "ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz"), compression="gzip")

        def format_category(category):
            category = category.replace("arxiv ", "")
            category = category.replace(" ", ".")
            category = category.upper()
            return f"Arxiv {category}"
        label_to_category['formatted_category'] = label_to_category['arxiv category'].apply(format_category)

        label_category_dict = dict(zip(label_to_category['label idx'], label_to_category['formatted_category']))

        QUESTION = f"Answer the following question: Which subcategory does this paper belong to? Please only output the most likely answer from the following subcategories and nothing else: {', '.join(list(label_category_dict.values()))}.\nAnswer: "
        
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

        # save data list
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
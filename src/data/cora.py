import os
from typing import Any, List

import numpy as np
import torch
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from src.data.base import BaseDataset

class CoraDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        preprocessor_llm: Any = None,
    ):
        super().__init__(
            dataset_name="cora",
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
        raw_path = os.path.join(self.root, "cora_raw")
        assert os.path.exists(raw_path), f"Raw data not found at {raw_path}"

        dataset = Planetoid(
            root=os.path.join(self.root, "cora_pyg"),
            name="Cora",
            transform=self.transform,
        )

        graph: Dataset | BaseData = dataset[0]

        # First, let's get the original paper IDs
        path = os.path.join(self.root, "cora_raw", "cora.content")
        idx_features_labels = np.genfromtxt(path, dtype=np.dtype(str))
        original_ids = idx_features_labels[:, 0]  # First column contains the paper IDs

        # These are the groups of duplicate nodes identified in the original preprocessing
        duplicate_groups = [
            [137, 2144],
            [283, 1260],
            [366, 1127, 1995],
            [503, 574, 2348],
            [995, 1338, 2102, 2103],
            [709, 1897],
            [950, 1495],
            [1020, 2076],
            [1040, 1719, 1720],
            [1033, 1586],
            [1031, 2205],
        ]

        # Keep only the first node from each duplicate group
        nodes_to_remove = [node for group in duplicate_groups for node in group[1:]]
        mask = torch.ones(graph.num_nodes, dtype=torch.bool)
        mask[nodes_to_remove] = False

        # Update graph graph
        edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
        graph.edge_index = graph.edge_index[:, edge_mask]

        # Create new node mapping after removal
        new_indices = -torch.ones(graph.num_nodes, dtype=torch.long)
        new_indices[mask] = torch.arange(mask.sum())

        # Update edge indices
        graph.edge_index = new_indices[graph.edge_index]

        # Update node features, labels, and masks
        graph.x = graph.x[mask]
        graph.y = graph.y[mask]
        graph.train_mask = graph.train_mask[mask]
        graph.val_mask = graph.val_mask[mask]
        graph.test_mask = graph.test_mask[mask]
        graph.num_nodes = mask.sum()
        
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
        with open(os.path.join(self.root, "cora_raw/mccallum/cora/papers")) as f:
            lines = f.readlines()

        pid_filename = {}
        for line in lines:
            pid = line.split("\t")[0]
            fn = line.split("\t")[1].split(":", 1)
            pid_filename[pid] = fn[0] + "_" + fn[1] if len(fn) == 2 else fn[0]

        class_names = [
            "theory",
            "reinforcement learning",
            "genetic algorithms",
            "neural networks",
            "probabilistic methods",
            "case based",
            "rule learning",
        ]

        data_list = []

        QUESTION = "Answer the following question: Which subcategory does this paper belong to? Please only output the most likely answer from the following subcategories and nothing else: theory, reinforcement learning, genetic algorithms, neural networks, probabilistic methods, case based, rule learning.\nAnswer: "

        # Read titles and abstracts
        current_idx = 0

        for i in range(len(original_ids)):
            if not mask[i]:
                continue

            original_id = original_ids[i]
            try:
                if original_id in pid_filename:
                    fn = pid_filename[original_id]
                    with open(
                        os.path.join(
                            self.root, "cora_raw/mccallum/cora/extractions", fn
                        )
                    ) as f:
                        lines = f.read().splitlines()
                        title = ""
                        abstract = ""
                        for line in lines:
                            if "Title:" in line:
                                title = line[7:]
                            if "Abstract:" in line:
                                abstract = line[10:]

                        context = {
                            "Title": title.strip(),
                            "Abstract": abstract.strip(),
                        }

                        data_list.append(
                            Data(
                                id=original_id,
                                question=QUESTION,
                                context=context,
                                label=class_names[common_y[current_idx].item()],
                                node_id=current_idx
                            )
                        )
                else:
                    data_list.append(
                        Data(
                            id=original_id,
                            question=QUESTION,
                            context={
                                "Title": "No available title",
                                "Abstract": "No available abstract",
                            },
                            label=class_names[common_y[current_idx].item()],
                            node_id=current_idx
                        )
                    )
                current_idx += 1
            except FileNotFoundError:
                data_list.append(
                    Data(
                        id=original_id,
                        question=QUESTION,
                        context={
                            "Title": "No available title",
                            "Abstract": "No available abstract",
                        },
                        label=class_names[common_y[current_idx].item()],
                        node_id=current_idx
                    )
                )
                current_idx += 1

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
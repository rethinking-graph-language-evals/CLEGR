import os
import pickle
from typing import Any

import torch
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.io import fs
import numpy as np

from src.data.base import BaseDataset
from src.data.gqa.generate import generate_pyg_dataset, filter_question_forms
from src.data.gqa.args import get_args

from transformers import AutoTokenizer, AutoModel

class CLEGRFullDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        size: str = "small",
        seed: int = 42,
    ):
        self.draw_dir = os.path.join(root, "clegr-full", "drawings")
        prefix = size+"-" if size != "small" else ""
        self.size = size
        self.seed = seed
        super().__init__(
            dataset_name=prefix + "clegr-full",
            root=root,
            transform=transform,
            force_reload=force_reload,
        )
        self.type = "graph"
        # Load processed data

        # map dataset to cpu for start
        path = self.processed_paths[0]
        out = fs.torch_load(path, map_location="cpu")
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    @property
    def processed_file_names(self) -> list[str]:
        return ["data_list.pt", "mappers.pkl"]

    def process(self):
        args = get_args()
        args.seed = self.seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # set to device
        self.model = self.model.to(self.device)

        setattr(args, "group", None)

        if self.size == "small":
            setattr(args, "small", True)
        elif self.size == "medium":
            setattr(args, "medium", True)
        elif self.size == "mixed":
            setattr(args, "mixed", True)
        elif self.size == "large":
            print("Using large dataset")

        forms_to_use = filter_question_forms(args)
        dataset, feature_mappers = generate_pyg_dataset(
            args=args, forms_to_use=forms_to_use, output_dir=self.draw_dir
        )

        print("Dataset loaded")
        print(len(dataset))
        print(dataset[0])

        data_list = []
        question_prefix = "Above is the representation of a synthetic subway network. All stations and lines are completely fictional. Keep in mind that the subway network is not real. All information necessary to answer the question is present in the above representation. The question is: "

        for data in tqdm(dataset):
            new_data = data.clone()
            node_features = self.embed_sentences(data.node_texts)
            edge_attr = self.embed_sentences(data.edge_texts)

            new_question = question_prefix + data.question
            new_data.question = new_question

            new_data.x = node_features
            new_data.edge_attr = edge_attr
            data_list.append(new_data)

        print("Data list created")

        # Save the dataset
        self.save(
            data_list,
            self.processed_paths[0],
        )
        # Save the mappers
        with open(self.processed_paths[1], "wb") as f:
            pickle.dump(feature_mappers, f)

    def split_by_graph(self, seed: int, split_ratio: tuple = (0.6, 0.2, 0.2)):
        # 1. Collect all graph_ids
        graph_ids = [self[i].graph_id for i in range(len(self))]
        # for i in range(len(self)):
        #     print(self[i].graph_id)
        unique_graph_ids = np.unique(graph_ids)    # 2. Shuffle graph_ids
        rng = np.random.RandomState(seed)
        shuffled_graph_ids = rng.permutation(unique_graph_ids)    # 3. Split graph_ids
        n = len(shuffled_graph_ids)
        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))
        train_graphs = set(shuffled_graph_ids[:train_end])
        val_graphs = set(shuffled_graph_ids[train_end:val_end])
        test_graphs = set(shuffled_graph_ids[val_end:])    # 4. Assign samples to splits based on graph_id
        train_indices = [i for i, gid in enumerate(graph_ids) if gid in train_graphs]
        val_indices = [i for i, gid in enumerate(graph_ids) if gid in val_graphs]
        test_indices = [i for i, gid in enumerate(graph_ids) if gid in test_graphs]    
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

    def split(self, seed, split_ratio=(0.6, 0.2, 0.2)):
        # Delegate to BaseDataset.split so the permutation is seed-dependent
        return self.split_by_graph(seed, split_ratio)

    def embed_sentences(self, sentences):
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = (
                outputs.last_hidden_state
            )  # (batch_size, seq_len, hidden_size)

        # Mean pooling (ignoring padding)
        attention_mask = inputs["attention_mask"].unsqueeze(
            -1
        )  # (batch_size, seq_len, 1)
        masked_hidden = last_hidden_state * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1)  # (batch_size, 1)
        embedding = sum_hidden / lengths  # (batch_size, hidden_size)

        return embedding  # Tensor of shape (batch_size, hidden_size)


class CLEGRReasoningDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        size: str = "small",
        seed: int = 42,
    ):
        self.draw_dir = os.path.join(root, "clegr-reasoning", "drawings")
        prefix = size+"-" if size != "small" else ""
        self.size = size
        self.seed = seed
        super().__init__(
            dataset_name=prefix + "clegr-reasoning",
            root=root,
            transform=transform,
            force_reload=force_reload,
        )
        self.type = "graph"
        # Load processed data

        # map dataset to cpu for start
        path = self.processed_paths[0]
        out = fs.torch_load(path, map_location="cpu")
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

class CLEGRFactsDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        size: str = "small",
        seed: int = 42,
    ):
        self.draw_dir = os.path.join(root, "clegr-facts", "drawings")
        prefix = size+"-" if size != "small" else ""
        self.size = size
        self.seed = seed
        super().__init__(
            dataset_name=prefix + "clegr-facts",
            root=root,
            transform=transform,
            force_reload=force_reload,
        )
        self.type = "graph"
        # Load processed data

        # map dataset to cpu for start
        path = self.processed_paths[0]
        out = fs.torch_load(path, map_location="cpu")
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    @property
    def processed_file_names(self) -> list[str]:
        return ["data_list.pt", "mappers.pkl"]

    def process(self):
        args = get_args()
        args.seed = self.seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # set to device
        self.model = self.model.to(self.device)

        setattr(args, "group", "FactBased")

        if self.size == "small":
            setattr(args, "small", True)
        elif self.size == "medium":
            setattr(args, "medium", True)
        elif self.size == "mixed":
            setattr(args, "mixed", True)
        elif self.size == "large":
            print("Using large dataset")

        forms_to_use = filter_question_forms(args)
        dataset, feature_mappers = generate_pyg_dataset(
            args=args, forms_to_use=forms_to_use, output_dir=self.draw_dir
        )

        print("Dataset loaded")
        print(len(dataset))
        print(dataset[0])

        data_list = []
        question_prefix = "Above is the representation of a synthetic subway network. All stations and lines are completely fictional. Keep in mind that the subway network is not real. All information necessary to answer the question is present in the above representation. The question is: "

        for data in tqdm(dataset):
            new_data = data.clone()
            node_features = self.embed_sentences(data.node_texts)
            edge_attr = self.embed_sentences(data.edge_texts)

            new_question = question_prefix + data.question
            new_data.question = new_question

            new_data.x = node_features
            new_data.edge_attr = edge_attr
            data_list.append(new_data)

        print("Data list created")

        # Save the dataset
        self.save(
            data_list,
            self.processed_paths[0],
        )
        # Save the mappers
        with open(self.processed_paths[1], "wb") as f:
            pickle.dump(feature_mappers, f)

    def split_by_graph(self, seed: int, split_ratio: tuple = (0.6, 0.2, 0.2)):
        # 1. Collect all graph_ids
        graph_ids = [self[i].graph_id for i in range(len(self))]
        # for i in range(len(self)):
        #     print(self[i].graph_id)
        unique_graph_ids = np.unique(graph_ids)    # 2. Shuffle graph_ids
        rng = np.random.RandomState(seed)
        shuffled_graph_ids = rng.permutation(unique_graph_ids)    # 3. Split graph_ids
        n = len(shuffled_graph_ids)
        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))
        train_graphs = set(shuffled_graph_ids[:train_end])
        val_graphs = set(shuffled_graph_ids[train_end:val_end])
        test_graphs = set(shuffled_graph_ids[val_end:])    # 4. Assign samples to splits based on graph_id
        train_indices = [i for i, gid in enumerate(graph_ids) if gid in train_graphs]
        val_indices = [i for i, gid in enumerate(graph_ids) if gid in val_graphs]
        test_indices = [i for i, gid in enumerate(graph_ids) if gid in test_graphs]    
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

    def split(self, seed, split_ratio=(0.6, 0.2, 0.2)):
        # Delegate to BaseDataset.split so the permutation is seed-dependent
        return self.split_by_graph(seed, split_ratio)

    def embed_sentences(self, sentences):
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = (
                outputs.last_hidden_state
            )  # (batch_size, seq_len, hidden_size)

        # Mean pooling (ignoring padding)
        attention_mask = inputs["attention_mask"].unsqueeze(
            -1
        )  # (batch_size, seq_len, 1)
        masked_hidden = last_hidden_state * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1)  # (batch_size, 1)
        embedding = sum_hidden / lengths  # (batch_size, hidden_size)

        return embedding  # Tensor of shape (batch_size, hidden_size)


class CLEGRReasoningDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        size: str = "small",
        seed: int = 42,
    ):
        self.draw_dir = os.path.join(root, "clegr-reasoning", "drawings")
        prefix = size+"-" if size != "small" else ""
        self.size = size
        self.seed = seed
        super().__init__(
            dataset_name=prefix + "clegr-reasoning",
            root=root,
            transform=transform,
            force_reload=force_reload,
        )
        self.type = "graph"
        # Load processed data

        # map dataset to cpu for start
        path = self.processed_paths[0]
        out = fs.torch_load(path, map_location="cpu")
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    @property
    def processed_file_names(self) -> list[str]:
        return ["data_list.pt", "mappers.pkl"]

    def process(self):
        args = get_args()
        args.seed = self.seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # set to device
        self.model = self.model.to(self.device)

        setattr(args, "group", "ReasoningBased")

        if self.size == "small":
            setattr(args, "small", True)
        elif self.size == "medium":
            setattr(args, "medium", True)
        elif self.size == "mixed":
            setattr(args, "mixed", True)
        elif self.size == "large":
            print("Using large dataset")

        forms_to_use = filter_question_forms(args)
        dataset, feature_mappers = generate_pyg_dataset(
            args=args, forms_to_use=forms_to_use, output_dir=self.draw_dir
        )

        print("Dataset loaded")
        print(len(dataset))
        print(dataset[0])

        data_list = []

        for data in tqdm(dataset):
            new_data = data.clone()
            node_features = self.embed_sentences(data.node_texts)
            edge_attr = self.embed_sentences(data.edge_texts)

            # new_question = data.question + question_suffix
            # new_data.question = new_question

            new_data.x = node_features
            new_data.edge_attr = edge_attr
            data_list.append(new_data)

        print("Data list created")

        # Save the dataset
        self.save(
            data_list,
            self.processed_paths[0],
        )
        # Save the mappers
        with open(self.processed_paths[1], "wb") as f:
            pickle.dump(feature_mappers, f)

    def split_by_graph(self, seed: int, split_ratio: tuple = (0.6, 0.2, 0.2)):
        # 1. Collect all graph_ids
        graph_ids = [self[i].graph_id for i in range(len(self))]
        # for i in range(len(self)):
        #     print(self[i].graph_id)
        unique_graph_ids = np.unique(graph_ids)    # 2. Shuffle graph_ids
        rng = np.random.RandomState(seed)
        shuffled_graph_ids = rng.permutation(unique_graph_ids)    # 3. Split graph_ids
        n = len(shuffled_graph_ids)
        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))
        train_graphs = set(shuffled_graph_ids[:train_end])
        val_graphs = set(shuffled_graph_ids[train_end:val_end])
        test_graphs = set(shuffled_graph_ids[val_end:])    # 4. Assign samples to splits based on graph_id
        train_indices = [i for i, gid in enumerate(graph_ids) if gid in train_graphs]
        val_indices = [i for i, gid in enumerate(graph_ids) if gid in val_graphs]
        test_indices = [i for i, gid in enumerate(graph_ids) if gid in test_graphs]    
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

    def split(self, seed, split_ratio=(0.6, 0.2, 0.2)):
        # Delegate to BaseDataset.split so the permutation is seed-dependent
        return self.split_by_graph(seed, split_ratio)

    def embed_sentences(self, sentences):
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = (
                outputs.last_hidden_state
            )  # (batch_size, seq_len, hidden_size)

        # Mean pooling (ignoring padding)
        attention_mask = inputs["attention_mask"].unsqueeze(
            -1
        )  # (batch_size, seq_len, 1)
        masked_hidden = last_hidden_state * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1)  # (batch_size, 1)
        embedding = sum_hidden / lengths  # (batch_size, hidden_size)

        return embedding  # Tensor of shape (batch_size, hidden_size)


# diff size datasets
class TinyCLEGRFactsDataset(CLEGRFactsDataset):
    def __init__(self, root: str, transform=None, force_reload: bool = False, seed: int = 42):
        super().__init__(
            root=root, transform=transform, force_reload=force_reload, size="tiny", seed=seed
        )


class LargeCLEGRFactsDataset(CLEGRFactsDataset):
    def __init__(self, root: str, transform=None, force_reload: bool = False, seed: int = 42):
        super().__init__(
            root=root, transform=transform, force_reload=force_reload, size="large", seed=seed
        )


class TinyCLEGRReasoningDataset(CLEGRReasoningDataset):
    def __init__(self, root: str, transform=None, force_reload: bool = False, seed: int = 42):
        super().__init__(
            root=root, transform=transform, force_reload=force_reload, size="tiny", seed=seed
        )


class LargeCLEGRReasoningDataset(CLEGRReasoningDataset):
    def __init__(self, root: str, transform=None, force_reload: bool = False, seed: int = 42):
        super().__init__(
            root=root, transform=transform, force_reload=force_reload, size="large", seed=seed
        )
        
if __name__ == "__main__":
    data = CLEGRFactsDataset(os.path.expandvars("/scratch/$USER/datasets_medium_final/"), size="medium")
    data.process()
import numpy as np
from torch_geometric.data import InMemoryDataset

class BaseDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name: str,
        root: str,
        transform=None,
        force_reload: bool = False,
    ):
        self.root = f"{root}/{dataset_name}"
        self.dataset_name = dataset_name
        self.transform = transform
        self.graph_data = None

        super().__init__(self.root, force_reload=force_reload)

    def process(self):
        raise NotImplementedError
    
    def split(self, seed: int, split_ratio: tuple = (0.6, 0.2, 0.2)):
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
    
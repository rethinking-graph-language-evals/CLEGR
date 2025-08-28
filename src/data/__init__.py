from .cora import CoraDataset
from .computers import ComputersDataset
from .computers_for_transfer import Computers_for_transfer_Dataset

from .clegr import (
    CLEGRFactsDataset,
    CLEGRReasoningDataset,
    CLEGRFullDataset,
    TinyCLEGRFactsDataset,
    LargeCLEGRFactsDataset,
    TinyCLEGRReasoningDataset,
    LargeCLEGRReasoningDataset,
)
from .clegr_retrieve import CLEGRRetrievalDataset, CLEGRRetrieveFactsDataset, CLEGRRetrieveReasoningDataset, CLEGRRetrieveFull
from .history import HistoryDataset
from .photo import PhotoDataset
from .arxiv import ArxivDataset
from .arxiv_for_transfer import Arxiv_for_transfer_Dataset

from .collators import NodeDataLoaderCollator


def get_dataset(dataset_name: str):
    mapper = {
        "cora": CoraDataset,
        "arxiv": ArxivDataset,
        "arxiv-for-transfer": Arxiv_for_transfer_Dataset,
        "clegr-facts": CLEGRFactsDataset,
        "clegr-reasoning": CLEGRReasoningDataset,
        "clegr-full": CLEGRFullDataset,
        "clegr-retrieve-full": CLEGRRetrieveFull,
        "clegr-retrieval": CLEGRRetrievalDataset,
        "clegr-retrieve-facts": CLEGRRetrieveFactsDataset,
        "clegr-retrieve-reasoning": CLEGRRetrieveReasoningDataset,
        "computers": ComputersDataset,
        "computers-for-transfer": Computers_for_transfer_Dataset,
        "history": HistoryDataset,
        "photo": PhotoDataset,
        "tiny-clegr-facts": TinyCLEGRFactsDataset,
        "tiny-clegr-reasoning": TinyCLEGRReasoningDataset,
        "large-clegr-facts": LargeCLEGRFactsDataset,
        "large-clegr-reasoning": LargeCLEGRReasoningDataset,
    }

    if dataset_name not in mapper:
        raise ValueError(f"Dataset '{dataset_name}' not found in mapper.")

    return mapper[dataset_name]

from .evaluator import DatasetEvaluator
from .cora_evaluator import CoraEvaluator
from .clegr_evaluator import CLEGREvaluator
from .computers_evaluator import ComputersEvaluator
from .history_evaluator import HistoryEvaluator
from .photo_evaluator import PhotoEvaluator
from .arxiv_evaluator import ArxivEvaluator

def get_evaluator(dataset_name: str) -> DatasetEvaluator:
    mapper = {
        "cora": CoraEvaluator,
        "arxiv": ArxivEvaluator,
        "arxiv-for-transfer": ArxivEvaluator,
        "clegr-facts": CLEGREvaluator,
        "clegr-retrieve-facts": CLEGREvaluator,
        "clegr-reasoning": CLEGREvaluator,
        "clegr-retrieve-reasoning": CLEGREvaluator,
        "clegr-full": CLEGREvaluator,
        "clegr-retrieve-full": CLEGREvaluator,
        "computers": ComputersEvaluator,
        "computers-for-transfer": ComputersEvaluator,
        "history": HistoryEvaluator,
        "photo": PhotoEvaluator,
        "tiny-clegr-facts": CLEGREvaluator,
        "tiny-clegr-reasoning": CLEGREvaluator,
        "large-clegr-facts": CLEGREvaluator,
        "large-clegr-reasoning": CLEGREvaluator,
    }

    if dataset_name not in mapper:
        raise ValueError(f"Dataset '{dataset_name}' not found in mapper.")

    return mapper[dataset_name]()

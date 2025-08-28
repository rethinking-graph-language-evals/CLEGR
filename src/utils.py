import json
import os
import torch
from accelerate.hooks import AlignDevicesHook
from absl import logging, flags
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import lightning as pl


# Import project modules
from src.data import get_dataset
from src.data.base import BaseDataset
from src.data.evaluators.evaluator import DatasetEvaluator
from src.models.glm.base import BaseGLM
from src.models.llm.llm_wrapper import LLM
from src.models import get_glm, get_glm_lightning_module
from src.data import NodeDataLoaderCollator, get_dataset

FLAGS = flags.FLAGS

llm_name_map = json.load(open("src/llm_name_map.json", "r"))


### DATASET UTILITIES ###
def load_dataset(root: str, dataset=None) -> BaseDataset:
    """Loads and verifies the dataset.

    Args:
        root (str): Root directory for the dataset.

    Returns:
        CoraDataset: The loaded dataset.

    Raises:
        AssertionError: If processing fails.
    """
    dataset_name = dataset if dataset else FLAGS.dataset
    try:
        dataset = get_dataset(dataset_name)(root=root)
        if dataset.type == "node":
            dataset.graph_data.share_memory_()
    except AssertionError as error:
        logging.error("Error during processing: %s", error)
        raise

    logging.info("Dataset processing successful.")
    return dataset


def split_dataset(dataset, seed: int):
    """Splits the dataset into train, validation, and test subsets.

    Args:
        dataset: The complete dataset.
        seed: The random seed.

    Returns:
        tuple: train_dataset, val_dataset, test_dataset.
    """
    split = dataset.split(seed=seed)
    train_dataset = [dataset[i] for i in split["train"]]
    val_dataset = [dataset[i] for i in split["val"]]
    test_dataset = [dataset[i] for i in split["test"]]

    logging.info(
        "Dataset split into train: %d, validation: %d, test: %d samples.",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int
):
    """Creates DataLoader objects for training, validation, and testing.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for the DataLoader.

    Returns:
        tuple: train_loader, val_loader, test_loader.
    """

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        if train_dataset is not None
        else None
    )
    val_loader = (
        DataLoader(
            val_dataset,
            num_workers=num_workers,
            shuffle=False,
            batch_size=batch_size,
            persistent_workers=True,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            num_workers=num_workers,
            shuffle=False,
            persistent_workers=True,
        )
        if test_dataset is not None
        else None
    )
    logging.info("Data loaders created.")
    return train_loader, val_loader, test_loader


############ MODEL UTILITIES ###########


def initialize_model(lr: float, graph_data=None, use_lora=False) -> pl.LightningModule:
    """Initializes the model and Lightning module.

    Args:
        lr (float): Learning rate for training.

    Returns:
        pl.LightningModule: The initialized Lightning module.

    Raises:
        ValueError: If the LLM model name is not found in the parameters dictionary.
    """
    # Load the LLM parameters mapping from file
    llm_params = load_llm_params(FLAGS.llm_params_file)
    if FLAGS.llm_model_name not in llm_params:
        logging.error(
            "LLM model name '%s' not found in parameters mapping.", FLAGS.llm_model_name
        )
        raise ValueError(
            f"LLM model name '{FLAGS.llm_model_name}' not found in parameters mapping."
        )

    llm_num_params = llm_params[FLAGS.llm_model_name]
    logging.info(
        "Using LLM model '%s' with %s parameters.", FLAGS.llm_model_name, llm_num_params
    )

    # Initialize LLM with specified parameters from flags
    llm = LLM(
        model_name=llm_name_map[FLAGS.llm_model_name],
        num_params=llm_num_params,
    )
    remove_hook_from_module(llm, recurse=True)

    glm = get_glm(method=FLAGS.method)(
        llm=llm,
        projector_out_channels=llm.llm.config.hidden_size,
        n_gnn_toks=FLAGS.n_gnn_toks,
        pretrain_path=f"{FLAGS.save_dir}/{FLAGS.method}/pretrain_{FLAGS.dataset}_seed_{FLAGS.seed}",
        use_lora=use_lora
    )

    # Freeze LLM parameters.
    for param in glm.llm.parameters():
        param.requires_grad = False
    if FLAGS.method not in ["g-retriever", "node-g-retriever", "llm-only", "soft-prompt"]:
        for param in glm.gnn.parameters():
            param.requires_grad = False

    model = get_glm_lightning_module(method=FLAGS.method)(
        model=glm,
        lr=lr,
        graph_data=graph_data,
        scheduler_step_size=FLAGS.scheduler_step_size,
        scheduler_gamma=FLAGS.scheduler_gamma,
    )
    logging.info("Model initialized successfully.")

    # print all parameters that are being trained
    print("Parameters being trained:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # check if the model subclasses lightning module correctly
    assert isinstance(
        model, pl.LightningModule
    ), f"Model does not subclass LightningModule: {type(model).__name__}"

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size(), "\t", type(model.state_dict()[param_tensor]).__name__)
    # raise NotImplementedError

    logging.info("Model initialized and LLM parameters frozen.")
    return model


def run_inference(
    model: BaseGLM, test_loader, evaluator: DatasetEvaluator, num_samples: int = 5
):
    model.eval()
    results = []
    count = 0

    # Obtain maximum token length using evaluator utility.
    # (Assuming evaluator.max_class_toklen accepts a tokenizer and returns the max token length.)
    max_tok_len = evaluator.max_class_toklen(model.llm.tokenizer)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(model.llm.device)
            # Use model.inference to generate outputs.
            out_seq = model.inference(batch, max_out_tokens=max_tok_len)
            # Here, assuming that batch.label exists and that out_seq is a list or tensor of outputs.
            # You might need to adjust if the data structure differs.
            label = batch.label if hasattr(batch, "label") else "N/A"

            # Append the results in JSON-friendly format.
            results.append(
                {
                    "question": batch.question[0],
                    "label": batch.label[0],
                    "context": batch.context[0],
                    "prediction": out_seq[0],
                    "ground_truth": label[0],
                }
            )
            count += 1
            if count >= num_samples:
                break

    # Print the results in JSON format.
    print(json.dumps(results, indent=2))


# https://github.com/Lightning-AI/pytorch-lightning/issues/19731
# https://github.com/Lightning-AI/pytorch-lightning/discussions/17878
def remove_hook_from_module(
    module: torch.nn.Module, recurse=False, hook_cls=AlignDevicesHook
):

    if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, hook_cls):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

        if hasattr(module, "_old_forward"):
            module.forward = module._old_forward
            delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module


def load_llm_params(file_path: str) -> dict:
    """
    Loads the LLM parameters mapping from a JSON file.

    The JSON file should contain a dictionary mapping LLM model names to their
    respective number of parameters, e.g.:

        {
            "microsoft/Phi-3.5-mini-instruct": 2,
            "another/model-name": 4
        }

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Mapping from LLM model names to number of parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is not a valid dictionary.
        Exception: For other I/O related errors.
    """
    if not os.path.exists(file_path):
        logging.error("LLM parameters file not found: %s", file_path)
        raise FileNotFoundError(f"LLM parameters file not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            llm_params = json.load(f)
    except Exception as e:
        logging.error("Error reading LLM parameters file %s: %s", file_path, e)
        raise

    if not isinstance(llm_params, dict):
        logging.error("Invalid LLM parameters format in file: %s", file_path)
        raise ValueError("Invalid format for LLM parameters; expected a dictionary.")

    logging.info("Loaded LLM parameters from %s", file_path)
    return llm_params

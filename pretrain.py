#!/usr/bin/env python
import torch
from torch_geometric.data import DataLoader

from absl import app, flags, logging

# Import the model from a dummy path (change the path accordingly)
from src.pretrain import get_trainer
from src.data import get_dataset

# ABSL flags for hyperparameters and configuration
flags.DEFINE_string("method", "glm-lite", "Method to use for training")
flags.DEFINE_string(
    "dataset",
    "cora",
    "Dataset name. Use 'cora' for node classification or e.g., 'mutag' for graph classification",
)
flags.DEFINE_string(
    "llm_model_name", "microsoft/Phi-3.5-mini-instruct", "Name of the LLM model to use."
)
flags.DEFINE_string("save_dir", "models", "Directory to save the model checkpoints")
flags.DEFINE_integer("num_layers", 3, "Number of layers in the GNN")
flags.DEFINE_integer("hidden_dim", 500, "Hidden dimension for GAT layers")
flags.DEFINE_integer(
    "output_dim", 500, "Output dimension from the GAT before the classification head"
)
flags.DEFINE_integer("num_heads", 4, "Number of attention heads for GAT layers")
flags.DEFINE_float("lr", 0.005, "Learning rate")
flags.DEFINE_float("weight_decay", 5e-4, "Weight decay")
flags.DEFINE_integer("epochs", 10, "Number of training epochs")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_bool("inference_only", False, "Whether to only run inference")
flags.DEFINE_string("root", "datasets", "Path to dataset root.")

FLAGS = flags.FLAGS


def main(_argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # Load dataset
    # TODO: add a loading mechanism for different datasets
    dataset = get_dataset(dataset_name=FLAGS.dataset)(root=FLAGS.root)

    split = dataset.split(seed=FLAGS.seed)

    # For node classification, use the indices to create masks, while for graph classification, use the indices to create different datasets
    if dataset.type == "node":
        # initialise masks
        train_mask = torch.zeros((dataset.graph_data.num_nodes,), dtype=torch.bool)
        test_mask = torch.zeros((dataset.graph_data.num_nodes,), dtype=torch.bool)
        val_mask = torch.zeros((dataset.graph_data.num_nodes,), dtype=torch.bool)

        # set the masks
        train_mask[split["train"]] = True
        test_mask[split["test"]] = True
        val_mask[split["val"]] = True

        dataset.train_mask = train_mask
        dataset.test_mask = test_mask
        dataset.val_mask = val_mask
        dataset.graph_data.train_mask = train_mask
        dataset.graph_data.test_mask = test_mask
        dataset.graph_data.val_mask = val_mask

        train_dataset = [dataset.graph_data]
        test_dataset = [dataset.graph_data]
        val_dataset = [dataset.graph_data]

    else:
        # graph classification
        train_dataset = dataset[split["train"]]
        test_dataset = dataset[split["test"]]
        val_dataset = dataset[split["val"]]

    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    # Define the trainer
    # TODO: add more trainers

    trainer = get_trainer(FLAGS.method)(
        dataset=dataset,
        device=device,
        task_level=dataset.type,
        FLAGS=FLAGS,
    )

    # Train the model
    if not FLAGS.inference_only:
        trainer.train(train_loader, val_loader)
    trainer.load()

    # Evaluate the model
    test_acc = trainer.evaluate(test_loader)
    logging.info(f"Test accuracy: {test_acc}")


if __name__ == "__main__":
    app.run(main)

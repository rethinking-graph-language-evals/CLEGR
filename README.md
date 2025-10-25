# CLEGR

![clegr_github](https://github.com/user-attachments/assets/c973467e-b7dd-4579-961c-22b2d4524aa2)


### Official code repository for A Graph Talks, But Who’s Listening? Rethinking Evaluations for Graph-Language Models

> **Accepted at NeurIPS 2025 New Perspectives on Graph Machine Learning (NPGML) Workshop**

This repository contains the code for CLEGR, a framework for generating graph-based reasoning tasks. It includes scripts for generating graph-based question-answering datasets, pre-training and training GLMs.

## Dataset Generation

The core of this repository is the ability to generate complex graph-language based question answering (GQA) datasets. The generation process is highly customizable.
Much of the code is taken from the [CLEVR Graph Repo](https://github.com/Octavian-ai/clevr-graph) with much appreciation.

Our generated dataset is available on the following link - [HF CLEGR Dataset](https://huggingface.co/datasets/tenseisoham/CLEGR/tree/main)

### Generating the CLEGR Dataset

The main script for dataset generation is `src/data/gqa/generate.py`. You can run it from the root of the repository.

```bash
python -m src.data.gqa.generate [arguments]
```

**Key Arguments:**

*   `--name`: Name for the generated dataset (e.g., `my_clegr_dataset`).
*   `--count`: The number of unique graphs to generate.
*   `--questions-per-graph`: The number of question-answer pairs to generate for each graph.
*   `--group`: Filter questions by a specific group (e.g., `query`, `count`, `compare`).
*   `--type-prefix`: Filter questions by a type prefix (e.g., `queryAttr`, `countNodes`).
*   `--seed`: Random seed for reproducibility.
*   `--draw`: If set, saves visualizations of a few generated graphs as PNG files.

**Example:**

To generate a small dataset with 100 graphs and 5 questions per graph, you can run:

```bash
python -m src.data.gqa.generate --name clegr_small --count 100 --questions-per-graph 5
```

The generated dataset will be saved as a `.pt` file in the `data_pyg/` directory, along with a `_mappers.pkl` file containing feature mappings.

## Training

The `training.py` script is used to train GLMs on Node Classification and CLEGR (GQA).

### Running the Training Script

```bash
python training.py [arguments]
```

**Key Arguments:**

*   `--method`: The GLM variant to use (e.g., `g-retriever`, `tea-glm`).
*   `--dataset`: The name of the dataset to use (should match the name provided during generation).
*   `--llm_model_name`: The name of the Large Language Model to use (e.g., `phi-3.5`).
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Training batch size.
*   `--lr`: Learning rate.
*   `--num_gpus`: Number of GPUs to use for training.
*   `--use_lora`: Whether to use LoRA for efficient LLM fine-tuning.
*   `--use_wandb`: Enable logging with Weights & Biases.

**Example:**

To train a `tea-glm` model on the `clegr-reasoning` dataset using LoRA:

```bash
python training.py --method tea-glm --dataset clegr-reasoning --llm_model_name phi-3.5 --use_lora --epochs 10 --batch_size 16
```

Checkpoints and LoRA adapters will be saved in the `models/` directory.

## Pre-training

The `pretrain.py` script is used for pre-training GLMs, specifically TEA-GLM.

### Running the Pre-training Script

```bash
python pretrain.py [arguments]
```

**Key Arguments:**

*   `--method`: The pre-training method (e.g., `tea-glm`).
*   `--dataset`: The dataset to use (e.g., `cora`, `computers`, `photo`).
*   `--llm_model_name`: The LLM to use for the pre-training method.
*   `--epochs`: Number of training epochs.
*   `--num_layers`: Number of layers in the GNN.
*   `--hidden_dim`: Hidden dimension of the GNN layers.

**Example:**

To pre-train a GNN on the Cora dataset:

```bash
python pretrain.py --dataset cora --epochs 50 --num_layers 3 --hidden_dim 256
```

The trained model will be saved in the `models/` directory.

**Important Note**

The g-retriever model is only to be trained with the dataset generated using `clegr_retrieve.py` file, which performs the PCST based retrieval and creates the dataset. Using the normal/full `clegr-facts` or `clegr-reasoning` on g-retriever will be the Graph-Token GLM behaviour.

import json
import os
from typing import Optional
import torch
import lightning as pl

from torch_geometric.data import Batch, Data

from src.models import GAT, GraphSAGE
from src.modifiers import Modifier
from src.models.glm.base import BaseGLM
from src.models import GraphSAGE
from src.models.llm.llm_wrapper import LLM


class GRetriever(BaseGLM):
    r"""The G-Retriever model from the `"G-Retriever: Retrieval-Augmented
    Generation for Textual Graph Understanding and Question Answering"
    <https://arxiv.org/abs/2402.07630>`_ paper.

    Args:
        llm (LLM): The LLM to use.
        gnn (torch.nn.Module): The GNN to use.
        use_lora (bool, optional): If set to :obj:`True`, will use LORA from
            :obj:`peft` for training the LLM, see
            `here <https://huggingface.co/docs/peft/en/index>`_ for details.
            (default: :obj:`False`)
        mlp_out_channels (int, optional): The size of each graph embedding
            after projection. (default: :obj:`4096`)
        mlp_out_tokens (int, optional): Number of LLM prefix tokens to
            reserve for GNN output. (default: :obj:`1`)

    .. warning::
        This module has been tested with the following HuggingFace models

        * :obj:`llm_to_use="meta-llama/Llama-2-7b-chat-hf"`
        * :obj:`llm_to_use="google/gemma-7b"`

        and may not work with other models. See other models at `HuggingFace
        Models <https://huggingface.co/models>`_ and let us know if you
        encounter any issues.

    .. note::
        For an example of using :class:`GRetriever`, see
        `examples/llm/g_retriever.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/llm/g_retriever.py>`_.
    """

    def __init__(
        self,
        llm: LLM,
        projector_out_channels: int,
        use_lora: bool = False,
        n_gnn_toks: int = 1,
        pretrain_path: str = None,
    ) -> None:
        dataset_name = pretrain_path.split("pretrain_")[1].split("_seed_")[0]
        with open(f"src/g_ret_params.json", "r") as f:
            try:
                param_dict = json.load(f)[dataset_name]
            except KeyError:
                raise KeyError(f"Dataset {dataset_name} not found in g_ret_params.json")

        # gnn = GAT(**param_dict).to(llm.device)
        gnn = GraphSAGE(**param_dict).to(llm.device)

        super().__init__(llm, gnn, use_lora, projector_out_channels, n_gnn_toks)

        self.mlp_hidden_channels = self.gnn.n_classes
        self.mlp_out_channels = projector_out_channels
        self.mlp_out_tokens = n_gnn_toks
        self.prepare_projector()

    def prepare_projector(self):
        self.projector = torch.nn.Sequential(
            # torch.nn.Linear(self.mlp_hidden_channels, self.mlp_hidden_channels),
            # torch.nn.Sigmoid(),
            torch.nn.Linear(
                self.mlp_hidden_channels, self.mlp_out_channels * self.mlp_out_tokens
            ),
            # torch.nn.ReLU(),
            torch.nn.Unflatten(-1, (self.mlp_out_tokens, self.mlp_out_channels)),
        ).to(self.llm.device)

    def encode_graph(self, data):
        return self.gnn(
            data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr
        )

    def forward(
        self,
        data: Batch,
        graph_data: Optional[Data] = None,
    ):
        self.llm.eval()

        data = data.to(self.llm.device)

        node_id = data.node_id if hasattr(data, "node_id") else None
        context = data.context if hasattr(data, "context") else None

        if graph_data is not None:
            x = self.encode_graph(graph_data)
        else:
            x = self.encode_graph(data)

        # filter based on node_id
        if node_id is not None:
            x = x[node_id]
        x = self.projector(x)
        # print(x.shape)
        xs = [hi.squeeze(0) for hi in x.split(1, dim=0)]
        # print(xs)

        # convert context to text from dict
        con = self.get_context(data)

        # Handle questions without node features:
        batch_unique = data.batch.unique()
        batch_size = len(data.question)
        if len(batch_unique) < batch_size:
            xs = [xs[i] if i in batch_unique else None for i in range(batch_size)]

        # print(xs.shape)
        (
            inputs_embeds,
            attention_mask,
            label_input_ids,
        ) = self.llm._get_embeds(data.question, con, xs, data.label)

        with self.llm.autocast_context:
            outputs = self.llm_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    @torch.no_grad()
    def inference(
        self,
        data,
        max_out_tokens: int = 32,
        graph_data=None,
        modifier: Optional[Modifier] = None,
    ):
        self.llm.eval()

        data = data.to(self.llm.device)

        # Extract optional fields if they exist.
        node_id = data.node_id if hasattr(data, "node_id") else None
        context = data.context if hasattr(data, "context") else None
        question = data.question

        # Encode the graph using the provided GNN.
        if graph_data is not None:
            # Encode the graph using the provided GNN.
            x = self.encode_graph(graph_data)
        else:
            # Encode the graph using the GNN from the data.
            x = self.encode_graph(data)

        if node_id is not None:
            x = x[node_id]
        x = self.projector(x)
        xs = [hi.squeeze(0) for hi in x.split(1, dim=0)]

        # Convert context dict to a string, if provided.
        # convert context to text from dict
        con = self.get_context(data)
        # Ensure that the number of graph tokens aligns with the batch.
        batch_unique = data.batch.unique()
        batch_size = len(question)
        if len(batch_unique) < batch_size:
            xs = [xs[i] if i in batch_unique else None for i in range(batch_size)]

        # Get the inputs for the language model.
        inputs_embeds, attention_mask, _ = self.llm._get_embeds(
            question, con, xs, modifier=modifier
        )
        with self.llm.autocast_context:
            outputs = self.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=max_out_tokens,
                return_dict_in_generate=True,
            )
        return self.llm.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class GRetrieverLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        scheduler_step_size=10,
        scheduler_gamma=0.1,
        graph_data=None,
    ):
        super().__init__()
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters(ignore=["model", "graph_data"])
        self.model = model
        self.strict_loading = (
            False  # Set to False because we only care about the projector
        )
        # Graph data moved to device if provided
        self.graph_data = (
            graph_data.to(self.model.llm.device) if graph_data is not None else None
        )

    def forward(self, batch):
        return self.model(batch, self.graph_data)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("val_loss", loss, batch_size=1, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Optimizer on trainable parameters
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
        )
        # StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # call every epoch
                "monitor": "val_loss",  # for ReduceLROnPlateau (not used here)
            },
        }

    def state_dict(self):
        # Don't save the llm, it is not being trained
        return {
            k: v
            for k, v in super().state_dict().items()
            if "projector" in k or "gnn" in k
        }

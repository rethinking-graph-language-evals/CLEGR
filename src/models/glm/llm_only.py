import json
from absl import logging
from typing import Optional
import torch
import lightning as pl

from torch_geometric.data import Batch

from src.modifiers import Modifier
from src.models.glm.base import BaseGLM
from src.models import GAT
from src.models.llm.llm_wrapper import LLM


class LLMOnly(BaseGLM):
    r"""
    A simple Graph Language Model (GLM) that uses a GNN to encode the graph.

    Args:
        llm (LLM): The language model to use.
        gnn (torch.nn.Module): The graph neural network to use.
        use_lora (bool): Whether to use LORA or not.
        projector_out_channels (int): The number of output channels of the projector.
        n_gnn_toks (int): The number of GNN tokens to use in the LLM prompt.
    """

    def __init__(
        self,
        llm: LLM,
        projector_out_channels: int,
        use_lora: bool = False,
        n_gnn_toks: int = 1,
        pretrain_path: str = None,
    ) -> None:

        # with open(f"{pretrain_path}_model_params.json", "r") as f:
        #     param_dict = json.load(f)

        # gnn = GAT(**param_dict).to(llm.device)

        # # load state dict
        # state_dict = torch.load(f"{pretrain_path}_model.pth", weights_only=False)
        # gnn.load_state_dict(state_dict)
        super().__init__(
            llm, use_lora=use_lora, projector_out_channels=0, n_gnn_toks=0, gnn=None
        )
        # super().__init__(llm, gnn, use_lora, 0, 0)

        self.prepare_projector()

    def prepare_projector(self):
        if self.gnn is None:
            self.projector = None
            return

        projector_in_channels = self.gnn.layers[-1].out_channels
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(
                projector_in_channels, self.projector_out_channels * self.n_gnn_toks
            ),
            torch.nn.Unflatten(-1, (self.n_gnn_toks, self.projector_out_channels)),
        ).to(self.llm.device)

    def encode_graph(self, data):
        return self.gnn(data.x, data.edge_index, batch=data.batch)

    def forward(
        self,
        data: Batch,
        graph_data=None,
    ):
        self.llm.eval()

        data = data.to(self.llm.device)

        node_id = data.node_id if hasattr(data, "node_id") else None
        context = data.context if hasattr(data, "context") else None

        # convert context to text from dict
        con = self.get_context(data)

        # Handle questions without node features:
        batch_unique = data.batch.unique()
        batch_size = len(data.question)
        # if len(batch_unique) < batch_size:
        #     xs = [xs[i] if i in batch_unique else None for i in range(batch_size)]

        # print(xs.shape)
        (
            inputs_embeds,
            attention_mask,
            label_input_ids,
        ) = self.llm._get_embeds(data.question, con, None, data.label)

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
        max_out_tokens: int,
        graph_data=None,
        modifier: Optional[Modifier] = None,
    ):
        self.llm.eval()

        # Extract optional fields if they exist.
        node_id = data.node_id if hasattr(data, "node_id") else None
        context = data.context if hasattr(data, "context") else None
        question = data.question

        # Encode the graph using the provided GNN.

        # Convert context dict to a string, if provided.
        con = self.get_context(data)
        # Get the inputs for the language model.
        inputs_embeds, attention_mask, _ = self.llm._get_embeds(
            question, con, None, modifier=modifier
        )
        with self.llm.autocast_context:
            outputs = self.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=max_out_tokens,
                return_dict_in_generate=True,
            )
        return self.llm.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class LLMOnlyLightningModule(pl.LightningModule):
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
        return self.model(batch)

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
        # Don't save the llm//gnn, it is not being trained
        return {k: v for k, v in super().state_dict().items() if "projector" in k}

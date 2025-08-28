import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from typing import Literal
from torch_geometric.data import Data

from src.data.base import BaseDataset
from src.pretrain.base_trainer import GNNTrainer
from src.models import GraphSAGE

# Import TEA-GLM modules
from ._teaglm.dataloader import NodeNegativeLoader
from ._teaglm.loss.contrastive_loss import ContrastiveLoss, GraceLoss
from ._teaglm.get_pc import get_pc

llm_name_map = json.load(open("src/llm_name_map.json", "r"))

class TEAGLMTrainer(GNNTrainer):
    def __init__(
        self,
        dataset: BaseDataset,
        device: torch.device,
        task_level: Literal["node", "graph"],
        FLAGS,
    ):
        
        # Hardcoded TEA-GLM hyperparameters (defaults)
        self.num_epochs = 70 if task_level == "node" else 1
        self.num_layers = 2
        self.num_negs = 0
        self.patience = 10
        self.fan_out = [25, 10]  # list corresponding to each layer
        self.batch_size = 512
        self.log_every = 20
        self.eval_every = 50
        self.lr = 0.002
        self.dropout = 0.5
        self.drop_edge_rate_1 = 0.3
        self.drop_edge_rate_2 = 0.4
        self.drop_feature_rate_1 = 0.0
        self.drop_feature_rate_2 = 0.1
        self.tau = 0.4
        self.use_tp = True
        self.self_tp = False
        self.gnn_type = 'sage'
        self.datasets_root = "datasets/teaglm/"

        self.FLAGS = FLAGS

        self.seed_everything(self.FLAGS.seed)
        
        if task_level == "node":
            self.dataset = dataset.graph_data
        else:
            self.dataset = dataset

        if task_level == "graph":
            for i in range(len(self.dataset)):
                self.dataset[i].x = self.dataset[i].x.type(torch.float)
            self.train_id = torch.cat([g.x for g in self.dataset], dim=0)
        else:
            self.dataset.x = self.dataset.x.type(torch.float)
            self.train_id = self.dataset.x
            self.dataset = [self.dataset]

        self.dataset = [
            Data(x=g.x.to(device), edge_index=g.edge_index.to(device)) for g in self.dataset
        ]

        llm_name = llm_name_map[self.FLAGS.llm_model_name]
        llm_str, all_principal_component = get_pc(llm_name, datasets_root=self.datasets_root)
        self.llm_str = llm_str
        self.all_principal_component = all_principal_component.to(device, dtype=torch.float)

        num_node_features = self.train_id.shape[1]
        param_dict = {
            "in_channels": num_node_features,
            "hidden_channels": self.all_principal_component.size(1) // 2,
            "out_channels": self.all_principal_component.size(1),
            "n_layers": self.num_layers,
            "num_proj_hidden": self.all_principal_component.size(1),
            "dropout": self.dropout,
            "edge_dim": None,
            "gnn_type": self.gnn_type
        }

        model = GraphSAGE(**param_dict).to(device, dtype=torch.float)
        super().__init__(model, self.dataset, device, task_level, FLAGS, param_dict)

        self.params_path = f"{self.save_name}_{self.llm_str}_model_params.json"
        self.model_path = f"{self.save_name}_{self.llm_str}_model.pth"

        if self.use_tp:
            self.criterion = ContrastiveLoss(self.tau, self_tp=self.self_tp).to(device)
        else:
            self.criterion = GraceLoss(self.tau).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.best_loss = float('inf')
        self.no_increase = 0

    def seed_everything(self, seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def train(self, loader=None, optimizer=None):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")
            total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss = self.train_epoch()
            if total_mean_loss < self.best_loss:
                self.best_loss = total_mean_loss
                self.no_increase = 0
                self.save()
            else:
                self.no_increase += 1
                if self.no_increase > self.patience:
                    print("Early stopping triggered.")
                    break

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_ins_loss = 0
        total_con_loss = 0
    
        pbar = tqdm(total=len(self.dataset))
        
        for data in self.dataset:
            train_loader = NodeNegativeLoader(
                data,
                batch_size=self.batch_size,
                shuffle=True,
                neg_ratio=self.num_negs,
                num_neighbors=self.fan_out,
                mask_feat_ratio_1=self.drop_feature_rate_1,
                mask_feat_ratio_2=self.drop_feature_rate_2,
                drop_edge_ratio_1=self.drop_edge_rate_1,
                drop_edge_ratio_2=self.drop_edge_rate_2,
            )
            
            for step, (ori_graph, view_1, view_2) in enumerate(train_loader):
                ori_graph, view_1, view_2 = (
                    ori_graph.to(self.device),
                    view_1.to(self.device),
                    view_2.to(self.device),
                )
                self.optimizer.zero_grad()

                z1 = self.model(view_1.x, view_1.edge_index)[view_1.node_label_index]
                z2 = self.model(view_2.x, view_2.edge_index)[view_2.node_label_index]
                proj_z1 = self.model.projection(z1)
                proj_z2 = self.model.projection(z2)

                principal_component = self.all_principal_component[ori_graph.raw_nodes] if self.self_tp else self.all_principal_component

                if self.use_tp:
                    loss, ins_loss, contrast_loss = self.criterion(z1, z2, proj_z1, proj_z2, principal_component)
                    total_ins_loss += ins_loss * proj_z1.shape[0]
                    total_con_loss += contrast_loss * proj_z1.shape[0]
                    total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]
                else:
                    loss = self.criterion(proj_z1, proj_z2)
                    total_loss += loss.data.item() * proj_z1.shape[0]

                loss.backward()
                self.optimizer.step()

                if step % self.log_every == 0:
                    if self.use_tp:
                        print("Step {:05d} | Loss {:.4f} | Instance Loss {:.4f} | Contrastive Loss {:.4f}".format(
                            step, loss.item(), ins_loss, contrast_loss))
                    else:
                        print("Step {:05d} | Loss {:.4f}".format(step, loss.item()))
            pbar.update()
        pbar.close()

        total_mean_loss = total_loss / self.train_id.shape[0]
        total_mean_instance_loss = total_ins_loss / self.train_id.shape[0]
        total_mean_contrastive_loss = total_con_loss / self.train_id.shape[0]

        return total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss

    def evaluate(self, loader=None):
        self.model.eval()
        total_loss = 0
        total_ins_loss = 0
        total_con_loss = 0

        for data in self.dataset:
            test_loader = NodeNegativeLoader(
                data,
                batch_size=512,
                shuffle=False,
                neg_ratio=0,
                num_neighbors=[-1],
                mask_feat_ratio_1=self.drop_feature_rate_1,
                mask_feat_ratio_2=self.drop_feature_rate_2,
                drop_edge_ratio_1=self.drop_edge_rate_1,
                drop_edge_ratio_2=self.drop_edge_rate_2,
            )
            pbar = tqdm(total=len(test_loader))
            for ori_graph, view_1, view_2 in test_loader:
                ori_graph, view_1, view_2 = (
                    ori_graph.to(self.device),
                    view_1.to(self.device),
                    view_2.to(self.device),
                )
                z1 = self.model(view_1.x, view_1.edge_index)[view_1.node_label_index]
                z2 = self.model(view_2.x, view_2.edge_index)[view_2.node_label_index]
                proj_z1 = self.model.projection(z1)
                proj_z2 = self.model.projection(z2)

                principal_component = self.all_principal_component[ori_graph.raw_nodes] if self.self_tp else self.all_principal_component

                if self.use_tp:
                    loss, ins_loss, contrast_loss = self.criterion(z1, z2, proj_z1, proj_z2, principal_component)
                    total_ins_loss += ins_loss * proj_z1.shape[0]
                    total_con_loss += contrast_loss * proj_z1.shape[0]
                    total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]
                else:
                    loss = self.criterion(proj_z1, proj_z2)
                    total_loss += loss.data.item() * proj_z1.shape[0]
                pbar.update()
            pbar.close()

        num_nodes = self.train_id.shape[0]
        total_mean_loss = total_loss / num_nodes
        total_mean_instance_loss = total_ins_loss / num_nodes
        total_mean_contrastive_loss = total_con_loss / num_nodes

        print(f"Mean Test Loss: {total_mean_loss}")
        print(f"Mean Test Instance Loss: {total_mean_instance_loss}")
        print(f"Mean Test Contrastive Loss: {total_mean_contrastive_loss}")

        return total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss

    def evaluate_transfer(self, loader=None):
        self.model.eval()
        self.classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                batch = data.batch if self.task_level == "graph_classification" and hasattr(data, "batch") else None
                x = self.model(data.x, data.edge_index, batch=batch, edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None)
                logits = self.classifier(x)
                logits = F.softmax(logits, dim=-1)
                pred = logits.argmax(dim=1)
                correct += pred.eq(data.y).sum().item()
                total += data.num_graphs if self.task_level == "graph_classification" else data.num_nodes
        return correct / total

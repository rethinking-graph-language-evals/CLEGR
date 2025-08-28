import json
import os
import torch
import torch.nn as nn
from typing import Literal


class GNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        device: torch.device,
        task_level: Literal["node", "graph"],
        FLAGS,
        param_dict,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.task_level = task_level
        self.FLAGS = FLAGS
        self.param_dict = param_dict

        self.save_name = f"{self.FLAGS.save_dir}/{self.FLAGS.method}/pretrain_{self.FLAGS.dataset}_seed_{self.FLAGS.seed}"
        self.params_path = f"{self.save_name}_model_params.json"
        self.model_path = f"{self.save_name}_model.pth"

        os.makedirs(f"{self.FLAGS.save_dir}/{self.FLAGS.method}", exist_ok=True)
        
        self.classifier = None

    def train(self, train_loader=None, eval_loader=None):
        raise NotImplementedError

    def evaluate(self, loader=None):
        raise NotImplementedError
    
    def evaluate_transfer(self, loader=None):
        raise NotImplementedError

    def save(self):
        # save all model params to a json
        with open(self.params_path, "w") as f:
            json.dump(self.param_dict, f, indent=4)

        # save the model
        torch.save(self.model.state_dict(), self.model_path)
        
        # save the classifier
        if self.classifier is not None:
            torch.save(self.classifier.state_dict(), self.model_path.replace("_model.pth", "_classifier.pth"))

    def load(self):
        # load the model
        self.model.load_state_dict(torch.load(self.model_path, weights_only=False))
        

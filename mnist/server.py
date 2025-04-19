from collections import OrderedDict

import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from model import Net, test


def get_on_fit_config(config: DictConfig):
    """Return a function to configure the client's fit."""

    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "local_epochs": config.local_epochs,
            "proximal_mu": config.proximal_mu,
            "server_round": server_round,
        }

    return fit_config_fn


def get_evalulate_fn(model_cfg: int, testloader):
    """Return a function to evaluate the global model."""

    def evaluate_fn(server_round: int, parameters, config):
        model = instantiate(model_cfg)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        loss, accuracy = test(model, testloader, device)

        return loss, {"loss": float(loss), "accuracy": accuracy}

    return evaluate_fn


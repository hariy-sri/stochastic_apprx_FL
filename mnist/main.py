import pickle
import os
import torch
import numpy as np
from pathlib import Path

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from client import generate_client_fn
from dataset import prepare_dataset
from server import get_evalulate_fn, get_on_fit_config

os.environ["HYDRA_FULL_ERROR"] = "1"

def seed_everything(seed: int):
    import random, os
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(42)
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    ## Prepare your dataset
    trainloaders, validationloaders, testloader, dataset_sizes = prepare_dataset(cfg.dataset.name,
        cfg.dataset.path, cfg.num_clients, cfg.num_classes, cfg.seed, cfg.partitioner, cfg.batch_size, cfg.alpha
    )

    ## Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg, save_path)

    ## Define your strategy
    strategy = instantiate(
            cfg.strategy, evaluate_fn=get_evalulate_fn(cfg.model, testloader),
        )
    ##Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 1},
    )

    print(history)

if __name__ == "__main__":
    main()
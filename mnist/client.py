import pickle
import os
import torch
import flwr as fl
import numpy as np
from pathlib import Path
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from collections import OrderedDict
from torch.utils.data import DataLoader
from typing import Dict


from model import test, train
from copy import deepcopy

#create a dictionary to store client losses and accuracies with client id as key and empty list as value
client_train_losses = {i: [] for i in range(10)}
client_train_accuracies = {i: [] for i in range(10)}
client_test_losses = {i: [] for i in range(10)}
client_test_accuracies = {i: [] for i in range(10)}
round_gradient_norms = {i: [] for i in range(10)}

class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient."""

    def __init__(
        self,
        save_path: str,
        client_id: int,
        model: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        stochastic: str, 
        method: str,
        partitioner: str,
    ) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.partitioner = partitioner

        self.model = model

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.stochastic = stochastic

        self.method = method

        self.save_path = save_path
        self.client_id = client_id

        lrs = {i: 0.001 for i in range(10)}  # Initialize all clients with 0.0001
        if self.method == "single_dominant":
            lrs[0] = 0.1  # Keep original value for client 0
        elif self.method == "double_dominant":
            lrs[0] = 0.1  # Keep original value for client 0
            lrs[1] = 0.01  # Keep original value for client 1
        
        self.lrs = lrs

    def set_parameters(self, parameters):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config["lr"] * self.lrs[self.client_id[0]]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        server_round = config["server_round"]
        proximal_mu = config["proximal_mu"]

        if self.stochastic == "per_round":
            if self.client_id[0] == 0:
                lr = lr / server_round ** (3 / 4)
            else:
                lr = lr / server_round

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Store gradient norms for clients

        batch_gradients = []
        batch_gradients_new = []
        global_model = deepcopy(self.model)
        params_dict = zip(global_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        global_model.load_state_dict(state_dict, strict=True)
        global_model.to(self.device)

        train(self.model, self.trainloader, optimizer, epochs, proximal_mu, server_round,
            self.stochastic, self.method, self.device, self.client_id[0], batch_gradients,batch_gradients_new)
        # Average the batch-level gradient norms per round
        avg_grad_norm = sum(batch_gradients) / len(batch_gradients)
        round_gradient_norms[self.client_id[0]].append(avg_grad_norm)

        os.makedirs(f"{self.save_path}/gradient_norms/{self.method}", exist_ok=True)
        with open(f"{self.save_path}/gradient_norms/{self.method}/{self.partitioner}.pkl", "wb") as f:
            pickle.dump(round_gradient_norms, f)

    # Save averaged round gradient
        if batch_gradients_new:
            avg_grad = torch.stack(batch_gradients_new).mean(dim=0)
            save_path = f"{self.save_path}/client_gradients/{self.method}/client_{self.client_id[0]}_round_{server_round}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(avg_grad, f)

        train_loss, train_acc = test(self.model, self.trainloader, self.device)
        print(f"Client {self.client_id[0]} train_loss: {train_loss}, train_accuracy: {train_acc}")
        client_train_losses[self.client_id[0]].append(train_loss)
        client_train_accuracies[self.client_id[0]].append(train_acc)

        os.makedirs(f"{self.save_path}/client_train_losses/{self.method}", exist_ok=True)
        os.makedirs(f"{self.save_path}/client_train_accuracies/{self.method}", exist_ok=True)
        with open(f"{self.save_path}/client_train_losses/{self.method}/{self.partitioner}.pkl", "wb") as f:
            pickle.dump(client_train_losses, f)
        with open(f"{self.save_path}/client_train_accuracies/{self.method}/{self.partitioner}.pkl", "wb") as f:
            pickle.dump(client_train_accuracies, f)

        return self.get_parameters({}), len(self.trainloader), {}


    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)
        print(f"Client {self.client_id[0]} test_loss: {loss}, test_accuracy: {accuracy}")
        client_test_losses[self.client_id[0]].append(loss)
        client_test_accuracies[self.client_id[0]].append(accuracy)
        #create all folders if they don't exist
        os.makedirs(f"{self.save_path}/client_test_losses/{self.method}", exist_ok=True)
        os.makedirs(f"{self.save_path}/client_test_accuracies/{self.method}", exist_ok=True)
        with open(f"{self.save_path}/client_test_losses/{self.method}/{self.partitioner}.pkl", "wb") as f:
            pickle.dump(client_test_losses, f)
        with open(f"{self.save_path}/client_test_accuracies/{self.method}/{self.partitioner}.pkl", "wb") as f:
            pickle.dump(client_test_accuracies, f)
        return float(loss), len(self.valloader), {"loss": float(loss), "accuracy": accuracy}


def generate_client_fn(trainloaders, valloaders, cfg, save_path) :
    """Return a function that will be called to instantiate the cid-th client."""

    def client_fn(context) -> fl.client.Client:
        """Create a Flower client representing a single organization."""
        # Instantiate model
        net = instantiate(cfg.model)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        return instantiate(
            cfg.client,
            save_path=save_path,
            client_id=[int(context.node_config["partition-id"])],
            model = net,
            trainloader=trainloaders[int(context.node_config["partition-id"])],
            valloader=valloaders[int(context.node_config["partition-id"])],
            stochastic = cfg.stochastic,
            method = cfg.method,
            partitioner = cfg.partitioner,
        ).to_client()

    return client_fn
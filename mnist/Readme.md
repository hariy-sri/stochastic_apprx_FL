# Federated Learning with MNIST Dataset

This repository contains a federated learning simulation using the MNIST dataset and a Feedforward Neural Network. The simulation explores two scenarios: a single dominant client and a double dominant client, focusing on the convergence behavior of model parameters and the impact of different learning rates.

## Overview

### Single Dominant Client

- **Objective**: Simulate a federated learning environment with one dominant client.
- **Dominant Client**: Client 1 has a significantly higher learning rate compared to other clients.
- **Convergence**: The global model parameters are influenced heavily by Client 1, leading to faster convergence.
- **Learning Rate Decay**: For Client 1, the learning rate decays as $lr = \frac{lr}{iter^{3/4}}$. For other clients, it decays as $lr = \frac{lr}{iter}$.

### Double Dominant Client

- **Objective**: Simulate a federated learning environment with two dominant clients.
- **Dominant Clients**: Clients 1 and 2 have higher learning rates, with Client 2's learning rate being half of Client 1's.
- **Convergence**: The global model parameters converge to a point where the combined influence of Client 1 and half of Client 2's influence is balanced..
- **Learning Rate Decay**: For Client 1, the learning rate decays as $lr = \frac{lr}{iter^{3/4}}$ . For Client 2, it decays as $lr = \frac{lr}{2*iter^{3/4}}$. For other clients, it decays as $lr = \frac{lr}{iter}$.

## Usage

1. **Dependencies**: Ensure you have the required Python packages installed. You can install them using:
   ```bash
   pip install numpy matplotlib torch torchvision flwr hydra-core
   ```

2. **Run Simulations**:
   - To run the simulation with a single dominant client, configure the `method` parameter in the configuration file to `single_dominant` and execute:
     ```bash
     python main.py
     ```
   - To run the simulation with double dominant clients, set the `method` parameter to `double_dominant` and execute:
     ```bash
     python main.py
     ```

3. **Results**: The results, including gradient norms and test loss plots, are saved in the `outputs` directory.

## File Descriptions

- **client.py**: Defines the Flower client class and client-specific operations.
- **dataset.py**: Handles the downloading and partitioning of the MNIST dataset.
- **main.py**: The main script to configure and start the federated learning simulation.
- **model.py**: Contains the definition of the Feedforward Neural Network and training/testing functions.
- **server.py**: Configures the server-side operations and evaluation functions.
- **plot_pickle.py**: Processes and plots the gradient norms and test loss from the simulation results.

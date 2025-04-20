
# Federated Learning Simulation on Synthetic data

This repository contains two Python scripts, `CODE_1.py` and `CODE_2.py`, which simulate federated learning scenarios with different configurations of dominant clients. The simulations aim to perform least squares regression using gradient descent, with varying learning rates among clients to observe the convergence behavior of model parameters.

## Overview

### CODE_1.py

- **Objective**: Simulate a federated learning environment with one dominant client.
- **Dominant Client**: Client 1 has a higher learning rate compared to other clients.
- **Learning Rate Decay**: For Client 1, the learning rate decays as $lr = \frac{lr}{iter^{3/4}}$. For other clients, it decays as $lr = \frac{lr}{iter}$.
- **Convergence**: The global model parameters (weights) converge to the values of Client 1, and the gradient of Client 1 approaches zero.

### CODE_2.py

- **Objective**: Simulate a federated learning environment with two dominant clients.
- **Dominant Clients**: Clients 1 and 2 have higher learning rates, with Client 2's learning rate being half of Client 1's.
- **Learning Rate Decay**: For Client 1, the learning rate decays as $lr = \frac{lr}{iter^{3/4}}$ . For Client 2, it decays as $lr = \frac{lr}{2*iter^{3/4}}$. For other clients, it decays as $lr = \frac{lr}{iter}$.
- **Convergence**: The global model parameters converge to a point where the combined gradient of Client 1 and half of Client 2's gradient is zero.


## Simulation Details

- **Data Generation**: Synthetic data is generated for each client using the `generate_and_save_client_data` function in `utils.py`.
- **Gradient Descent**: The scripts use mini-batch stochastic gradient descent (SGD) to update model parameters.
- **Global Aggregation**: After local updates, the global model parameters are aggregated using a weighted sum based on the data size of each client.
- **Metrics and Plots**: The scripts save the trajectories of model parameters and gradients, and generate plots to visualize convergence.

## Usage

1. **Dependencies**: Ensure you have the required Python packages installed. You can install them using:
   ```bash
   pip install numpy matplotlib joblib
   ```

2. **Run Simulations**:
   - To run the simulation with one dominant client, execute:
     ```bash
     python CODE_1.py
     ```
   - To run the simulation with two dominant clients, execute:
     ```bash
     python CODE_2.py
     ```

3. **Results**: The results, including model parameter trajectories and gradient plots, are saved in the respective directories (`single_dominant` and `double_dominant`).

## File Descriptions

- **CODE_1.py**: Script for simulating federated learning with one dominant client.
- **CODE_2.py**: Script for simulating federated learning with two dominant clients.
- **utils.py**: Contains utility functions for data generation, SGD updates, and plotting.
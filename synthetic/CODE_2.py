import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import json
import os
import pickle
from utils import *

# Set seed for reproducibility
np.random.seed(42)

# Parameters
num_clients = 10
datapoints_per_client = 5000
rounds = 5000
N_values = [1, 5, 10]
batch_size = 50
save_dir = "double_dominant"

# Load and preprocess data
with open('clients_data.json', 'r') as f:
    raw_data = json.load(f)

# Vectorized data loading
clients_data = {
    int(client_id): (
        np.array(client_data['theta'], dtype=np.float32),
        np.array(client_data['X'], dtype=np.float32),
        np.array(client_data['Y'], dtype=np.float32),
        client_data['length']
    )
    for client_id, client_data in raw_data.items()
}

# Pre-extract frequently used data
X0, Y0 = clients_data[0][1], clients_data[0][2]
X1, Y1 = clients_data[1][1], clients_data[1][2]
data_sizes = np.array([data[3] for data in clients_data.values()])
total_data_size = np.sum(data_sizes)
data_weights = data_sizes / total_data_size

# Initialize storage structures
gradient_means = {}
global_theta_sgd_trajectories = {}
local_thetas_sgd_trajectories_N = {}
local_gradient_sgd_trajectories_N = {}

# Initialize client thetas (random 3D vectors)
initial_client_thetas = np.random.normal(-10, 10, (num_clients, 3)).astype(np.float32)
print(f"Initial Client Thetas:\n{initial_client_thetas}")

def process_client_epoch(client_id, X, Y, theta, base_lr, batch_counter, N):
    """Process one epoch for a client"""
    gradients = []
    data_size = len(X)
    num_batches = data_size // batch_size
    
    # Pre-compute learning rate based on client ID
    if client_id == 0:
        lr_denominator = (batch_counter + 10) ** (3/4)
    elif client_id == 1:
        lr_denominator = 2 * (batch_counter + 10) ** (3/4)
    else:
        lr_denominator = batch_counter
    current_lr = base_lr / lr_denominator
    
    # Create batches
    indices = np.random.permutation(data_size)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]
        
        theta, gradient = sgd_update(theta, X_batch, Y_batch, current_lr)
        gradients.append(gradient)
    
    return theta, np.mean(gradients, axis=0), num_batches

# Main training loop
for N in N_values:
    client_thetas = initial_client_thetas.copy()
    client_batch_counters = np.ones(num_clients, dtype=np.int32)
    local_thetas_trajectories = {i: [] for i in range(num_clients)}
    local_gradients_trajectories = {i: [] for i in range(num_clients)}
    global_theta_estimates = []

    for r in range(rounds):
        for client_id in range(num_clients):
            _, X, Y, _ = clients_data[client_id]
            base_lr = 1.0 if client_id in [0, 1] else 0.01
            
            # Process N local epochs
            client_gradients = []
            for _ in range(N):
                theta, gradient, num_batches = process_client_epoch(
                    client_id, X, Y, client_thetas[client_id],
                    base_lr, client_batch_counters[client_id], N
                )
                client_thetas[client_id] = theta
                client_batch_counters[client_id] += num_batches
                client_gradients.append(gradient)
            
            local_thetas_trajectories[client_id].append(client_thetas[client_id].copy())
            local_gradients_trajectories[client_id].append(np.mean(client_gradients, axis=0))

        # Global aggregation
        global_theta = np.sum(client_thetas * data_weights[:, np.newaxis], axis=0)
        global_theta_estimates.append(global_theta)
        client_thetas = np.tile(global_theta, (num_clients, 1))

    # Store results
    global_theta_sgd_trajectories[N] = global_theta_estimates
    local_thetas_sgd_trajectories_N[N] = local_thetas_trajectories
    local_gradient_sgd_trajectories_N[N] = local_gradients_trajectories
    
    # Compute final metrics
    final_global_theta = np.mean(global_theta_estimates[-100:], axis=0)
    gradient_client0 = compute_gradient(X0, Y0, final_global_theta)
    gradient_client1 = compute_gradient(X1, Y1, final_global_theta)
    
    gradient_means[N] = {
        "client0": gradient_client0,
        "client1": gradient_client1,
        "combined": gradient_client0 + 0.5 * gradient_client1
    }
    
    print(f"\nN={N}:")
    print(f"Global Theta: {global_theta}")
    print(f"Gradient Client 0: {gradient_client0}")
    print(f"Gradient Client 1: {gradient_client1}")
    print(f"Combined Gradient: {gradient_means[N]['combined']}")

# Save results 
os.makedirs(save_dir, exist_ok=True)
save_data = {
    'gradient_means': gradient_means,
    'global_theta_sgd_trajectories': global_theta_sgd_trajectories,
    'local_gradient_sgd_trajectories_N': local_gradient_sgd_trajectories_N,
    'local_thetas_sgd_trajectories_N': local_thetas_sgd_trajectories_N
}

for name, data in save_data.items():
    with open(f"{save_dir}/{name}.pkl", "wb") as f:
        pickle.dump(data, f)

# Generate plots
plot_theta_convergence(global_theta_sgd_trajectories, rounds, reference=None, 
                      output_path=f"{save_dir}/2_vector_sgd_convergence_components.png")
plot_local_gradients_multiple_N(local_gradient_sgd_trajectories_N, 
                              output_path=f"{save_dir}/2_local_gradients_multiple_N.png")
plot_local_thetas_multiple_N(local_thetas_sgd_trajectories_N, 
                            output_path=f"{save_dir}/2_local_thetas_multiple_N.png")
plot_combined_gradients_multiple_N(local_gradient_sgd_trajectories_N, 
                                 output_path=f"{save_dir}/2_combined_gradients_multiple_N.png")
plot_combined_thetas_multiple_N(local_thetas_sgd_trajectories_N, 
                              output_path=f"{save_dir}/2_combined_thetas_multiple_N.png")
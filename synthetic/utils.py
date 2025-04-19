import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
import json

def generate_and_save_client_data(num_clients, datapoints_per_client, output_file, fixed_theta_client0=None, snr_db=10):
    """
    Generate client-specific theta and data, and save to a file.

    Parameters:
        num_clients (int): Number of clients.
        datapoints_per_client (int): Number of data points per client.
        output_file (str): File path to save the serialized client data.
        fixed_theta_client0 (array-like, optional): Fixed theta for client 0. Defaults to None.
        snr_db (float): Signal-to-noise ratio in decibels. Defaults to 10.

    """
    def generate_client_data(client_id):
        theta_client = np.random.uniform(-25, 25, size=3)
        if client_id == 0 and fixed_theta_client0 is not None:
            theta_client = np.array(fixed_theta_client0)
        X_variance = np.random.uniform(0.5, 5.0, size=3)
        X = np.random.normal(0, np.sqrt(X_variance), (datapoints_per_client, 3))

        # Calculate signal power
        signal = X @ theta_client
        signal_power = np.mean(signal ** 2)

        # Calculate required noise variance for the target SNR
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        epsilon_variance = target_noise_power

        epsilon = np.random.normal(0, np.sqrt(epsilon_variance), datapoints_per_client)
        Y = signal + epsilon

        return client_id, (theta_client, X, Y, len(X))

    # Generate data using parallel processing
    clients_data = dict(Parallel(n_jobs=-1)(delayed(generate_client_data)(client_id) for client_id in range(num_clients)))

    # Serialize the data for each client and save it to a file
    serializable_data = {}
    for client_id, (theta, X, Y, length) in clients_data.items():
        serializable_data[str(client_id)] = {
            'theta': theta.tolist(),
            'X': X.tolist(),
            'Y': Y.tolist(),
            'length': length
        }

    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=4)



# Function to perform mini-batch SGD update
def sgd_update(theta, X_batch, Y_batch, learning_rate):
    prediction_error = Y_batch - X_batch @ theta
    gradient = -2 * X_batch.T @ prediction_error / len(X_batch)
    return theta - learning_rate * gradient, gradient

#compute the gradient of client 0 and client 1 at the final global theta
def compute_gradient(X, Y, theta):
    prediction_error = Y - X @ theta
    return -2 * X.T @ prediction_error / len(X)

def plot_gradient_means_client0(gradient_means, output_path="CODE_2/gradient_means_plot.png"):
    """
    Plots the mean gradients for each component across different values of N.

    Parameters:
        gradient_means (dict): Dictionary containing gradient means for each N.
                              Format: {
                                  N: {
                                      "client0": np.array,
                                  },
                                  ...
                              }
        output_path (str): Path to save the resulting plot.
    """
    # components = ['Gradient[0]', 'Gradient[1]', 'Gradient[2]']
    components_ = [r'Local Epochs ($N$) vs Mean $h^{(1)}[0]$', r'Local Epochs ($N$) vs Mean $h^{(1)}[1]$', r'Local Epochs ($N$) vs Mean $h^{(1)}[2]$']
    components = [r'Mean $h^{(1)}[0]$', r'Mean $h^{(1)}[1]$', r'Mean $h^{(1)}[2]$']
    N_values = gradient_means.keys()

    plt.figure(figsize=(18, 6))

    for i in range(3):  # Gradient components
        plt.subplot(1, 3, i+1)
        for N in N_values:
            combined_gradient = gradient_means[N]["client0"]
            plt.bar(str(N), combined_gradient[i], label=f"N={N}")
        plt.title(f"{components_[i]}")
        plt.xlabel("Local Epochs (N)")
        plt.ylabel(components[i])
        plt.legend(title="Local Epochs")

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()

def plot_gradient_means(gradient_means, output_path="CODE_2/gradient_means_plot.png"):
    """
    Plots the mean gradients for each component across different values of N.

    Parameters:
        gradient_means (dict): Dictionary containing gradient means for each N.
                              Format: {
                                  N: {
                                      "client0": np.array,
                                      "client1": np.array,
                                      "combined": np.array
                                  },
                                  ...
                              }
        output_path (str): Path to save the resulting plot.
    """
    # components = ['Gradient[0]', 'Gradient[1]', 'Gradient[2]']
    components_ = [r'Local Epochs ($N$) vs Mean $h^{(1)}[0]$+$\frac{1}{2}h^{(2)}[0]$', r'Local Epochs ($N$) vs Mean $h^{(1)}[1]$+$\frac{1}{2}h^{(2)}[1]$', r'Local Epochs ($N$) vs Mean $h^{(1)}[2]$+$\frac{1}{2}h^{(2)}[2]$']
    components = [r'$h^{(1)}[0]$+$\frac{1}{2}h^{(2)}[0]$', r'$h^{(1)}[1]$+$\frac{1}{2}h^{(2)}[1]$', r'$h^{(1)}[2]$+$\frac{1}{2}h^{(2)}[2]$']
    N_values = gradient_means.keys()

    plt.figure(figsize=(18, 6))

    for i in range(3):  # Gradient components
        plt.subplot(1, 3, i+1)
        for N in N_values:
            combined_gradient = gradient_means[N]["combined"]
            plt.bar(str(N), combined_gradient[i], label=f"N={N}")
        plt.title(f"{components_[i]}")
        plt.xlabel("Local Epochs (N)")
        plt.ylabel(components[i])
        plt.legend(title="Local Epochs")

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def plot_theta_convergence(global_theta_sgd_trajectories, rounds, reference= None, output_path="CODE_2/vector_sgd_convergence_components.png"):
    """
    Plots the global theta estimates for each component across federated learning rounds.

    Parameters:
        global_theta_sgd_trajectories (dict): Dictionary containing global theta estimates for each N.
                                             Format: {
                                                 N: [
                                                     theta_estimates_per_round
                                                 ],
                                                 ...
                                             }
        rounds (int): Number of federated learning rounds.
        output_path (str): Path to save the resulting plot.
    """
    # theta_components = ['Theta[0]', 'Theta[1]', 'Theta[2]']
    theta_components = [r'$w*[0]$', r'$w*[1]$', r'$w*[2]$']

    plt.figure(figsize=(18, 6))
    # rounds = 2000
    for i in range(3):  # Theta components
        plt.subplot(1, 3, i+1)
        for N, global_theta_estimates in global_theta_sgd_trajectories.items():
            estimates = np.array(global_theta_estimates)
            # estimates = estimates[:2000]
            plt.plot(
                np.arange(1, rounds + 1), 
                estimates[:, i], 
                label=f"Local Epochs N={N}", 
                marker='o'
            )
        if reference is not None:
            plt.axhline(y=reference[i], color='r', linestyle='-', label=f"True {theta_components[i]}")
        plt.xlabel("Federated Learning Rounds")
        plt.ylabel(f"{theta_components[i]} Estimate")
        plt.title(f"{theta_components[i]} Convergence")
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def plot_local_gradients_multiple_N(local_gradient_sgd_trajectories_N, output_path="CODE_2/local_gradients_multiple_N.png"):
    """
    Plot the local gradients for each client over time for multiple N values, with each N as a column and each gradient component as a row.

    Parameters:
        local_gradient_sgd_trajectories_N (dict): Gradient trajectories for each N.
                                                Format: {
                                                    N: {
                                                        client_id: [gradients]
                                                    }
                                                }
        output_path (str): Path to save the plot.
    """
    num_N = len(local_gradient_sgd_trajectories_N)
    num_components = 3  # Gradient components
    components__ = [r'$h^{(i)}[0]$', r'$h^{(i)}[1]$', r'$h^{(i)}[2]$']
    components_ = [r'$\mathcal{{C}}_{10}$',r'$\mathcal{{C}}_{9}$',r'$\mathcal{{C}}_{8}$',r'$\mathcal{{C}}_{7}$',r'$\mathcal{{C}}_{6}$',r'$\mathcal{{C}}_{5}$',r'$\mathcal{{C}}_{4}$',r'$\mathcal{{C}}_{3}$',r'$\mathcal{{C}}_{2}$', r'$\mathcal{{C}}_{1}$']
    fig, axes = plt.subplots(num_components, num_N, figsize=(6 * num_N, 4 * num_components), sharey=True)

    for col_idx, (N, client_gradients) in enumerate(local_gradient_sgd_trajectories_N.items()):
        for i in range(num_components):  # Gradient components
            ax = axes[i, col_idx] if num_components > 1 else axes[col_idx]
            # for client_id, gradients in client_gradients.items():
            #     gradients_array = np.array(gradients)
            #     ax.plot(gradients_array[:, i], label=f'Client {client_id}')
            for l,client_id in enumerate(reversed(client_gradients.keys())):
                gradients = client_gradients[client_id]
                gradients_array = np.array(gradients)
                # gradients_array = gradients_array[:2000]
                ax.plot(gradients_array[:, i], label=f'{components_[l]}')
            ax.set_title(rf'{components__[i]} for $N={N}$ ')
            # ax.set_title(rf'N={N} Gradient Component {i}')
            ax.set_xlabel('Iteration')
            if col_idx == 0:
                # ax.set_ylabel(f'Gradient {i}')
                ax.set_ylabel(f'{components__[i]}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()

def plot_local_thetas_multiple_N(local_thetas_sgd_trajectories_N, output_path="CODE_2/local_thetas_multiple_N.png"):
    """
    Plot the local theta trajectories for each client over time for multiple N values, with each N as a column and each theta component as a row.

    Parameters:
        local_thetas_sgd_trajectories_N (dict): Theta trajectories for each N.
                                              Format: {
                                                  N: {
                                                      client_id: [thetas]
                                                  }
                                              }
        output_path (str): Path to save the plot.
    """
    num_N = len(local_thetas_sgd_trajectories_N)
    num_components = 3  # Theta components
    components_ = [r'$\mathcal{{C}}_{10}$',r'$\mathcal{{C}}_{9}$',r'$\mathcal{{C}}_{8}$',r'$\mathcal{{C}}_{7}$',r'$\mathcal{{C}}_{6}$',r'$\mathcal{{C}}_{5}$',r'$\mathcal{{C}}_{4}$',r'$\mathcal{{C}}_{3}$',r'$\mathcal{{C}}_{2}$', r'$\mathcal{{C}}_{1}$']
    fig, axes = plt.subplots(num_components, num_N, figsize=(6 * num_N, 4 * num_components), sharey=True)

    for col_idx, (N, client_thetas) in enumerate(local_thetas_sgd_trajectories_N.items()):
        for i in range(num_components):  # Theta components
            ax = axes[i, col_idx] if num_components > 1 else axes[col_idx]
            
            for l,client_id in enumerate(reversed(client_thetas.keys())):
                thetas = client_thetas[client_id]
                thetas_array = np.array(thetas)
                # thetas_array = thetas_array[:2000]
                ax.plot(thetas_array[:, i], label=f'{components_[l]}')
            ax.set_title(rf'$w^{{(i)}}[{i}]$ for $N={N}$')
            ax.set_xlabel('Iteration')
            if col_idx == 0:
                ax.set_ylabel(rf'$w^{{(i}}[{i}]$')
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def plot_gradients_multiple_N(local_gradient_sgd_trajectories_N, output_path="CODE_2/combined_gradients_multiple_N.png"):
    """
    Plot the client 0 gradient over time for multiple N values, with each N as a column and each gradient component as a row.

    Parameters:
        local_gradient_sgd_trajectories_N (dict): Gradient trajectories for each N.
        output_path (str): Path to save the plot.
    """
    num_N = len(local_gradient_sgd_trajectories_N)
    num_components = 3  # Gradient components

    components__ = [r'$h^{(1)}[0]$', r'$h^{(1)}[1]$', r'$h^{(1)}[2]$']

    fig, axes = plt.subplots(num_components, num_N, figsize=(6 * num_N, 4 * num_components), sharey=True)

    for col_idx, (N, client_gradients) in enumerate(local_gradient_sgd_trajectories_N.items()):
        gradient_client0 = np.array(client_gradients[0])
        # gradient_combined = np.clip(gradient_combined, -10.0, 10.0)
        gradient_combined = 10 * (2 / (1 + np.exp(-gradient_client0 / 10)) - 1) 
        mean_gradient_combined = np.mean(gradient_combined, axis=0)
        # mean_gradient_combined = mean_gradient_combined[:2000]

        for i in range(num_components):  # Gradient components
            ax = axes[i, col_idx] if num_components > 1 else axes[col_idx]
            ax.plot(gradient_combined[:, i], label=rf'$h[{i}]$')
            ax.axhline(y=mean_gradient_combined[i], color=f'C{i + 1}', linestyle='--',
                       label=f'Mean {components__[i]}: {mean_gradient_combined[i]:.2f}')
            ax.set_title(f'{components__[i]} for $N={N}$')
            ax.set_xlabel('Iteration')
            if col_idx == 0:
                ax.set_ylabel(f'{components__[i]}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()

def plot_combined_gradients_multiple_N(local_gradient_sgd_trajectories_N, output_path="CODE_2/combined_gradients_multiple_N.png"):
    """
    Plot the combined gradients (client 0 + 0.5 * client 1) over time for multiple N values, with each N as a column and each gradient component as a row.

    Parameters:
        local_gradient_sgd_trajectories_N (dict): Gradient trajectories for each N.
        output_path (str): Path to save the plot.
    """
    num_N = len(local_gradient_sgd_trajectories_N)
    num_components = 3  # Gradient components

    fig, axes = plt.subplots(num_components, num_N, figsize=(6 * num_N, 4 * num_components), sharey=True)
    components_ = [r'Mean $h^{(1)}[0]$+$\frac{1}{2}h^{(2)}[0]$', r'Mean $h^{(1)}[1]$+$\frac{1}{2}h^{(2)}[1]$', r'Mean $h^{(1)}[2]$+$\frac{1}{2}h^{(2)}[2]$']
    components__ = [r'$h^{(1)}[0]$+$\frac{1}{2}h^{(2)}[0]$', r'$h^{(1)}[1]$+$\frac{1}{2}h^{(2)}[1]$', r'$h^{(1)}[2]$+$\frac{1}{2}h^{(2)}[2]$']
    for col_idx, (N, client_gradients) in enumerate(local_gradient_sgd_trajectories_N.items()):
        gradient_client0 = np.array(client_gradients[0])
        gradient_client1 = np.array(client_gradients[1])
        gradient_combined = gradient_client0 + 0.5 * gradient_client1
        # gradient_combined = np.clip(gradient_combined, -10.0, 10.0)
        gradient_combined = 10 * (2 / (1 + np.exp(-gradient_combined / 10)) - 1) 
        mean_gradient_combined = np.mean(gradient_combined, axis=0)
        # mean_gradient_combined = mean_gradient_combined[:2000]

        for i in range(num_components):  # Gradient components
            ax = axes[i, col_idx] if num_components > 1 else axes[col_idx]
            ax.plot(gradient_combined[:, i], label=components__[i])
            ax.axhline(y=mean_gradient_combined[i], color=f'C{i + 1}', linestyle='--',
                       label=f"{components_[i]}: {mean_gradient_combined[i]:.2f}")
            ax.set_title(rf'{components__[i]} for $N={N}$')
            ax.set_xlabel('Iteration')
            if col_idx == 0:
                ax.set_ylabel(components__[i])
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def plot_all_combined_gradients_multiple_N(local_gradient_sgd_trajectories_N, output_path="CODE_2/combined_gradients_multiple_N.png"):
    """
    Plot the combined gradients (client 0 +  client 1 + ... client 9) over time for multiple N values, with each N as a column and each gradient component as a row.

    Parameters:
        local_gradient_sgd_trajectories_N (dict): Gradient trajectories for each N.
        output_path (str): Path to save the plot.
    """
    num_N = len(local_gradient_sgd_trajectories_N)
    num_components = 3  # Gradient components

    fig, axes = plt.subplots(num_components, num_N, figsize=(6 * num_N, 4 * num_components), sharey=True)

    for col_idx, (N, client_gradients) in enumerate(local_gradient_sgd_trajectories_N.items()):
        #add all the gradient_client
        gradient_combined = np.sum([np.array(client_gradients[i]) for i in range(10)], axis=0)
        # gradient_combined = np.sum(client_gradients, axis=0)
        # print(gradient_combined)
        # gradient_combined = np.clip(gradient_combined, -10.0, 10.0)
        gradient_combined = 10 * (2 / (1 + np.exp(-gradient_combined / 10)) - 1) 
        mean_gradient_combined = np.mean(gradient_combined, axis=0)
        # mean_gradient_combined = mean_gradient_combined[:2000]

        for i in range(num_components):  # Gradient components
            ax = axes[i, col_idx] if num_components > 1 else axes[col_idx]
            ax.plot(gradient_combined[:, i], label=rf'$h[{i}]$')
            ax.axhline(y=mean_gradient_combined[i], color=f'C{i + 1}', linestyle='--',
                       label=rf'Mean $h[{i}]$: {mean_gradient_combined[i]:.2f}')
            ax.set_title(rf'Combined $h[{i}]$ for $N={N}$')
            ax.set_xlabel('Iteration')
            if col_idx == 0:
                ax.set_ylabel(rf'$h[{i}]$')
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def plot_combined_thetas_multiple_N(local_thetas_sgd_trajectories_N, output_path="CODE_2/combined_thetas_multiple_N.png"):
    """
    Plot the combined thetas (client 0 + 0.5 * client 1) over time for multiple N values, with each N as a subplot.

    Parameters:
        local_thetas_sgd_trajectories_N (dict): Theta trajectories for each N.
        output_path (str): Path to save the plot.
    """
    num_N = len(local_thetas_sgd_trajectories_N)
    fig, axes = plt.subplots(1, num_N, figsize=(6 * num_N, 6), sharey=True)
    components__ = [r'$w^{(1)}[0]$+$\frac{1}{2}w^{(2)}[0]$', r'$w^{(1)}[1]$+$\frac{1}{2}w^{(2)}[1]$', r'$w^{(1)}[2]$+$\frac{1}{2}w^{(2)}[2]$']
    a = r'$w^{(1)}$+$\frac{1}{2}w^{(2)}$'
    for idx, (N, client_thetas) in enumerate(local_thetas_sgd_trajectories_N.items()):
        ax = axes[idx] if num_N > 1 else axes
        theta_client0 = np.array(client_thetas[0])
        theta_client1 = np.array(client_thetas[1])
        theta_combined = theta_client0 + 0.5 * theta_client1
        # theta_combined = theta_combined[:2000]

        for i in range(theta_combined.shape[1]):
            ax.plot(theta_combined[:, i], label=components__[i])
        ax.set_title(f'{a} for $N={N}$')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(a)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def plot_all_combined_thetas_multiple_N(local_thetas_sgd_trajectories_N, output_path="CODE_2/combined_thetas_multiple_N.png"):
    """
    Plot the combined thetas (client 0 + client 1+ .. client 9) over time for multiple N values, with each N as a subplot.

    Parameters:
        local_thetas_sgd_trajectories_N (dict): Theta trajectories for each N.
        output_path (str): Path to save the plot.
    """
    num_N = len(local_thetas_sgd_trajectories_N)
    fig, axes = plt.subplots(1, num_N, figsize=(6 * num_N, 6), sharey=True)

    for idx, (N, client_thetas) in enumerate(local_thetas_sgd_trajectories_N.items()):
        ax = axes[idx] if num_N > 1 else axes
        theta_combined = np.sum([np.array(client_thetas[i]) for i in range(10)], axis=0)
        # theta_combined = theta_combined[:2000]

        for i in range(theta_combined.shape[1]):
            ax.plot(theta_combined[:, i], label=rf'$w[{i}]$')
        ax.set_title(rf'Combined $w$ for $N={N}$')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$w$')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def plot_thetas_multiple_N(local_thetas_sgd_trajectories_N, output_path="CODE_2/combined_thetas_multiple_N.png"):
    """
    Plot the client 0  over time for multiple N values, with each N as a subplot.

    Parameters:
        local_thetas_sgd_trajectories_N (dict): Theta trajectories for each N.
        output_path (str): Path to save the plot.
    """
    num_N = len(local_thetas_sgd_trajectories_N)
    fig, axes = plt.subplots(1, num_N, figsize=(6 * num_N, 6), sharey=True)

    for idx, (N, client_thetas) in enumerate(local_thetas_sgd_trajectories_N.items()):
        ax = axes[idx] if num_N > 1 else axes
        theta_client0 = np.array(client_thetas[0])
        # theta_client0 = theta_client0[:2000]

        for i in range(theta_client0.shape[1]):
            ax.plot(theta_client0[:, i], label=rf'$w^{(1)}[{i}]$')
        ax.set_title(rf'$w^{(1)}$ for $N={N}$')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(rf'$w^{(1)}$')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()
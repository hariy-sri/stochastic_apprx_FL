import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from pathlib import Path
import logging
from typing import Dict, List, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAIN_PATH = "outputs/2025-04-19/16-44-43"
METHOD = "double_dominant"
COMPONENTS = [
    r'$\mathcal{{C}}_{10}$', r'$\mathcal{{C}}_{9}$', r'$\mathcal{{C}}_{8}$',
    r'$\mathcal{{C}}_{7}$', r'$\mathcal{{C}}_{6}$', r'$\mathcal{{C}}_{5}$',
    r'$\mathcal{{C}}_{4}$', r'$\mathcal{{C}}_{3}$', r'$\mathcal{{C}}_{2}$',
    r'$\mathcal{{C}}_{1}$'
]
FOLDERS = [
    "client_train_accuracies", "client_test_losses",
    "client_test_accuracies", "client_train_losses"
]

def load_pickle(file_path: str) -> Dict:
    """Load data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}")
        return {}

def save_plot(plt, file_path: str) -> None:
    """Save plot to file."""
    try:
        plt.savefig(file_path)
        logger.info(f"Plot saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {file_path}: {e}")

def plot_metrics(data: Dict, file_path: str, metric_type: str) -> None:
    """Plot metrics (accuracy/loss) for each client."""
    is_loss = "loss" in metric_type.lower()
    is_test = "test" in metric_type.lower()
    
    title = f"{'Test' if is_test else 'Train'} {'Loss' if is_loss else 'Accuracy'} per Client"
    ylabel = "Loss" if is_loss else "Accuracy"
    
    plt.figure(figsize=(10, 6))
    for i, client in enumerate(reversed(data.keys())):
        plt.plot(data[client], label=COMPONENTS[i])
    
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    plot_file_path = file_path.replace('.pkl', '.png')
    save_plot(plt, plot_file_path)
    plt.close()

def process_gradient_norms(folder_path: str) -> None:
    """Process and plot gradient norms for single dominant method."""
    for file in os.listdir(folder_path):
        if not file.endswith('.pkl'):
            logger.info(f"Skipping non-pickle file: {file}")
            continue
            
        file_path = os.path.join(folder_path, file)
        round_gradients = load_pickle(file_path)
        
        if not round_gradients:
            continue
            
        plt.figure(figsize=(10, 6))
        for i, client in enumerate(reversed(round_gradients.keys())):
            plt.plot(round_gradients[client], label=COMPONENTS[i])
            
        plt.title("Average Gradient Norm Per Round")
        plt.xlabel("Round")
        plt.ylabel("Avg Gradient Norm (L2)")
        plt.grid(True)
        plt.legend()
        
        plot_file_path = file_path.replace('.pkl', '.png')
        save_plot(plt, plot_file_path)
        plt.close()

def process_double_dominant(main_path: str, method: str, num_clients: int = 10, num_rounds: int = 110) -> None:
    """Process and plot gradients for double dominant method."""
    client_gradients = defaultdict(list)
    combined_norms = []

    # Load all client gradients per round
    for rnd in range(1, num_rounds + 1):
        round_grads = {}
        for cid in range(num_clients):
            path = Path(main_path) /  "client_gradients" / method /f"client_{cid}_round_{rnd}.pkl"
            if path.exists():
                grad = load_pickle(str(path))
                round_grads[cid] = grad
                client_gradients[cid].append(torch.norm(grad).item())
            else:
                logger.warning(f"Missing gradient: client {cid} round {rnd}")

        # Compute combined gradient
        if 0 in round_grads and 1 in round_grads:
            combined = round_grads[0] + 0.5 * round_grads[1]
            combined_norms.append(torch.norm(combined).item())
        else:
            combined_norms.append(None)

    # Plot gradient norms
    plt.figure(figsize=(10, 6))
    for i, cid in enumerate(reversed(client_gradients.keys())):
        plt.plot(client_gradients[cid], label=COMPONENTS[i])

    plt.plot(combined_norms, label=r"$\mathcal{C}_{1}$ + $\frac{1}{2}\mathcal{C}_{2}$",
             linewidth=2, linestyle="--", color="black")

    plt.title("Gradient Norms per Round")
    plt.xlabel("Round")
    plt.ylabel("L2 Norm of Gradient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = Path(main_path) / "gradient_norms" /method / "gradient_norms_all_clients.png"
    save_plot(plt, str(plot_path))
    plt.close()

def main():
    """Main execution function."""
    # Process metrics for each folder
    for folder in FOLDERS:
        folder_path = os.path.join(MAIN_PATH, folder, METHOD)
        for file in os.listdir(folder_path):
            if file.endswith('.pkl'):
                file_path = os.path.join(folder_path, file)
                data = load_pickle(file_path)
                if data:
                    plot_metrics(data, file_path, folder)

    # Process based on method
    if METHOD == "single_dominant":
        gradient_folder = os.path.join(MAIN_PATH, "gradient_norms", METHOD)
        process_gradient_norms(gradient_folder)
    elif METHOD == "double_dominant":
        process_double_dominant(MAIN_PATH, METHOD)

if __name__ == "__main__":
    main()
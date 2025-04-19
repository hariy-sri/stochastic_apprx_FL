import os
import pickle
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

num_clients = 10
num_rounds = 110  # Adjust to your actual value

client_gradients = defaultdict(list)
combined_norms = []

# Load all client gradients per round
for rnd in range(1, num_rounds + 1):
    round_grads = {}
    for cid in range(num_clients):
        path = f"pickles2/double/client_gradients/client_{cid}_round_{rnd}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                grad = pickle.load(f)
                round_grads[cid] = grad
                client_gradients[cid].append(torch.norm(grad).item())
        else:
            print(f"Missing gradient: client {cid} round {rnd}")

    # Compute combined gradient (client 1 + 0.5 * client 2)
    if 0 in round_grads and 1 in round_grads:
        combined = round_grads[0] + 0.5 * round_grads[1]
        combined_norms.append(torch.norm(combined).item())
    else:
        combined_norms.append(None)

# Plot gradient norms
plt.figure(figsize=(10, 6))
cids = reversed(client_gradients.keys())
components_ = [r'$\mathcal{{C}}_{10}$',r'$\mathcal{{C}}_{9}$',r'$\mathcal{{C}}_{8}$',r'$\mathcal{{C}}_{7}$',r'$\mathcal{{C}}_{6}$',r'$\mathcal{{C}}_{5}$',r'$\mathcal{{C}}_{4}$',r'$\mathcal{{C}}_{3}$',r'$\mathcal{{C}}_{2}$', r'$\mathcal{{C}}_{1}$']
for i,cid in enumerate(cids):
    norms = client_gradients[cid]
    plt.plot(norms, label=f"{components_[i]}")
# for cid, norms in client_gradients.items():
#     plt.plot(norms, label=f"Client {cid+1}")

plt.plot(combined_norms, label=r"$\mathcal{C}_{1}$ + $\frac{1}{2}\mathcal{C}_{2}$", linewidth=2, linestyle="--", color="black")

plt.title("Gradient Norms per Round")
plt.xlabel("Round")
plt.ylabel("L2 Norm of Gradient")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pickles2/double/gradient_norms_all_clients.png")
plt.show()

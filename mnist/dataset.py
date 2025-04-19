import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST,FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from scipy.stats import dirichlet
import numpy as np

def get_mnist(data_path: str = "./data"):
    """Downlaod MNIST and apply a simple transform."""

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def get_fashion_mnist(data_path: str = "./data"):
    """Downlaod MNIST and apply a simple transform."""

    tr = Compose([ToTensor()])

    trainset = FashionMNIST(data_path, train=True, download=True, transform=tr)
    testset = FashionMNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def prepare_dataset(name: str, path: str, num_partitions: int, num_classes: int, seed: int,  partitioner: str, batch_size: int, alpha: float, val_ratio: float = 0.1):
    """Download and partition the MNIST dataset."""
    if name == "mnist":
    
        trainset, testset = get_mnist(path)

    if name == "fmnist":
    
        trainset, testset = get_fashion_mnist(path)
 
    if partitioner == "IID":

        # split trainset into `num_partitions` trainsets
        num_images = len(trainset) // num_partitions

        partition_len = [num_images] * num_partitions

        trainsets = random_split(
            trainset, partition_len, torch.Generator().manual_seed(seed)
        )

        # create dataloaders with train+val support
        trainloaders = []
        valloaders = []
        dataset_sizes = []
        for trainset_ in trainsets:
            num_total = len(trainset_)
            num_val = int(val_ratio * num_total)
            num_train = num_total - num_val
            dataset_sizes.append(num_train)

            for_train, for_val = random_split(
                trainset_, [num_train, num_val], torch.Generator().manual_seed(seed)
            )

            trainloaders.append(
                DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
            )
            valloaders.append(
                DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
            )     

    elif partitioner == "DIR":
        trainloaders, valloaders, dataset_sizes = create_dir_non_iid_data(trainset, num_partitions, num_classes, alpha, batch_size, seed)
    elif partitioner == "N-IID":
        trainloaders, valloaders, dataset_sizes = create_pathological_dataloaders(trainset, batch_size, seed, num_classes, num_partitions)

    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader, dataset_sizes



def dirichlet_allocation(dataset, num_clients, num_classes, alpha):
    """
    Allocate indices of dataset among clients based on Dirichlet distribution.
    :param labels: Array of dataset labels
    :param num_clients: Number of clients
    :param alpha: Concentration parameter for Dirichlet distribution
    :return: List of indices for each client
    """
 
    labels = np.array(dataset.targets)
    # Dirichlet distribution
    distribution = dirichlet.rvs([alpha] * num_clients, size=num_classes)

    # Indices allocation for each client
    client_indices = {i: np.array([], dtype='int') for i in range(num_clients)}
    for k in range(num_classes):
        # Indices of class k
        class_k_indices = np.where(labels == k)[0]

        # Multinomial distribution to split indices among clients
        indices_split = np.random.multinomial(len(class_k_indices), distribution[k])

        # Distribute indices of class k to clients
        start = 0
        for i in range(num_clients):
            end = start + indices_split[i]
            client_indices[i] = np.concatenate((client_indices[i], class_k_indices[start:end]))
            start = end
    return client_indices


def create_dir_non_iid_data(dataset,num_clients, num_classes, alpha, batch_size, seed):
    # Apply Dirichlet allocation
    client_data_indices = dirichlet_allocation(dataset, num_clients, num_classes, alpha)
    # print(client_data_indices)
    clients_data = {i: Subset(dataset, client_data_indices[i]) for i in range(num_clients)}
        
    trainloaders = []
    valloaders = []
    dataset_sizes = []
    for i in range(num_clients):
        ds = clients_data[i]
        print(ds)
        print(len(ds))
        len_val = len(ds)//10
        if len_val == 0:
            len_val = 1
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        dataset_sizes.append(len_train)
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size, shuffle=True))

    return trainloaders, valloaders, dataset_sizes


def create_dataloaders(datasets,batch_size,seed):
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    client_samples = []
    for i in range(len(datasets)):
        ds = datasets[i]
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
        client_samples.append(len(ds_train))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders,valloaders, client_samples


def create_pathological_dataloaders(datasets, batch_size, seed, num_classes, num_clients):
    trainloaders = []
    valloaders = []
    client_samples = []
    
    # Determine the number of classes per client
    if num_classes == 10:  # 1 class per client
        classes_per_client = 1
        clients_per_class = num_clients // num_classes
    else:  # 10 unique classes per client
        classes_per_client = 10
        clients_per_class = num_clients

    # Create a mapping from class indices to dataset indices
    class_to_indices = {cls: [] for cls in range(num_classes)}
    for idx, (data, label) in enumerate(datasets):
        class_to_indices[label].append(idx)

    client_id = 0
    
    # Assign classes to clients in a pathological way
    for cls_start in range(0, num_classes, classes_per_client):
        class_subset = list(range(cls_start, min(cls_start + classes_per_client, num_classes)))
        
        for _ in range(clients_per_class):
            client_indices = []
            
            # Collect data from assigned classes
            for cls in class_subset:
                client_indices.extend(class_to_indices[cls])
            
            # Create a subset for this client
            client_ds = Subset(datasets, client_indices)
            len_val = len(client_ds) // 10  # 10% for validation
            len_train = len(client_ds) - len_val
            lengths = [len_train, len_val]
            
            ds_train, ds_val = random_split(client_ds, lengths, torch.Generator().manual_seed(seed))
            client_samples.append(len(ds_train))
            
            trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=batch_size))
            
            client_id += 1
    
    return trainloaders, valloaders, client_samples

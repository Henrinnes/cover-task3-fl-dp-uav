"""
Non-IID data partitioning for FL clients using the Dirichlet distribution.
Ref: Hsieh et al. (2020), "Quagmire of momentum: understanding the effects of
     learning rate in FL" — established benchmark for non-IID FL experiments.

Supported datasets
------------------
cifar10   — auto-downloads via torchvision; standard FL benchmark
eurosat   — auto-downloads via torchvision; 10-class satellite imagery (64×64)
            Most relevant to COVER: land-cover classification from aerial sensors.
            Classes include River, Forest, Highway, Industrial, Residential, etc.
aider     — 5-class real UAV disaster imagery (fire, flooded_areas,
            collapsed_building, traffic_incident, normal); download from
            https://zenodo.org/record/3888300, extract AIDER.zip into
            data/aider/ (produces data/aider/AIDER/<class>/*.jpg).
custom    — point load_dataset() at any folder with ImageFolder-compatible structure
            (pre-split into train/ and test/ subfolders; use for FloodNet or
            any other dataset once manually downloaded and split)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Tuple


# ------------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------------

def load_cifar10(data_dir: str = "./data/cifar10") -> Tuple[Dataset, Dataset]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    return train, test


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_samples: int = 10,
    min_per_class: int = 0,
) -> List[Subset]:
    """
    Partition dataset into `num_clients` non-IID subsets.
    Each class is split according to a Dirichlet(alpha) draw.
    Lower alpha → fewer classes per client (more heterogeneous).

    `min_per_class` (stratified floor): guarantees every client receives at
    least this many examples of *every* class before the Dirichlet-skewed
    remainder is distributed, capped at what the class can evenly supply.
    Without this, a low alpha combined with a small and/or class-imbalanced
    dataset (e.g. AIDER: 5 classes, ~400-500 images each bar the majority
    class, over 20 clients) routinely truncates a client's share of a class
    to exactly zero -- 9/20 clients had zero examples of at least one class
    at alpha=0.5 on AIDER before this floor was added, which is enough on
    its own to prevent convergence regardless of DP noise or model choice.
    Default 0 preserves the original (non-stratified) behaviour exactly.
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = int(labels.max()) + 1

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)

        floor_n = min(min_per_class, len(c_idx) // num_clients)
        if floor_n > 0:
            for cid in range(num_clients):
                client_indices[cid].extend(
                    c_idx[cid * floor_n : (cid + 1) * floor_n].tolist()
                )
            c_idx = c_idx[floor_n * num_clients :]

        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = (proportions * len(c_idx)).astype(int)
        # fix rounding so all samples are assigned
        counts[-1] = len(c_idx) - counts[:-1].sum()

        start = 0
        for cid, n in enumerate(counts):
            client_indices[cid].extend(c_idx[start : start + n].tolist())
            start += n

    # Ensure every client has at least min_samples (redistribute from largest)
    for cid in range(num_clients):
        if len(client_indices[cid]) < min_samples:
            donor = max(range(num_clients), key=lambda x: len(client_indices[x]))
            needed = min_samples - len(client_indices[cid])
            client_indices[cid].extend(client_indices[donor][:needed])
            client_indices[donor] = client_indices[donor][needed:]

    return [Subset(dataset, idxs) for idxs in client_indices]


def load_eurosat(data_dir: str = "./data/eurosat") -> Tuple[Dataset, Dataset]:
    """
    EuroSAT: 10-class satellite/aerial land-cover dataset (64×64 RGB).
    Auto-downloads ~89 MB. Classes directly relevant to emergency response:
    Forest, River, Highway, Industrial, Residential, AnnualCrop, etc.

    Ref: Helber et al. (2019), "EuroSAT: A Novel Dataset and Deep Learning
         Benchmark for Land Use and Land Cover Classification." IEEE J-STARS.
    """
    transform_train = transforms.Compose([
        transforms.Resize(32),           # downscale to match SimpleCNN input
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.3444, 0.3803, 0.4078), (0.2034, 0.1366, 0.1148)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.3444, 0.3803, 0.4078), (0.2034, 0.1366, 0.1148)),
    ])
    full = datasets.EuroSAT(data_dir, download=True, transform=transform_train)

    # 80/20 train-test split (EuroSAT has no official split)
    n = len(full)
    n_train = int(0.8 * n)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    train_set = Subset(full, train_idx)
    # Apply test transforms to the test split via a wrapper
    test_full = datasets.EuroSAT(data_dir, download=False, transform=transform_test)
    test_set = Subset(test_full, test_idx)
    return train_set, test_set


def load_custom(data_dir: str, img_size: int = 32) -> Tuple[Dataset, Dataset]:
    """
    Generic ImageFolder loader for custom aerial datasets (FloodNet, AIDER, etc.).
    Expects directory layout:
        data_dir/
          train/class_a/img1.jpg ...
          test/class_a/img1.jpg  ...

    Download links:
      FloodNet : https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021
      AIDER    : https://github.com/ckyrkou/AIDER
      HERIDAL  : https://ipsar.fesb.unist.hr/HERIDAL%20database.html
    """
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    train_set = datasets.ImageFolder(f"{data_dir}/train", transform=transform_train)
    test_set = datasets.ImageFolder(f"{data_dir}/test", transform=transform_test)
    return train_set, test_set


def load_aider(data_dir: str = "./data/aider", img_size: int = 32) -> Tuple[Dataset, Dataset]:
    """
    AIDER: Aerial Image dataset for Emergency Response applications.
    5 classes: fire, flooded_areas, collapsed_building, traffic_incident, normal
    (heavily imbalanced -- 'normal' is ~68% of the 6433 images). Directly
    on-task for COVER: real UAV-captured disaster imagery, not a proxy dataset.

    Expects `data_dir` to contain the extracted `AIDER/<class>/*.jpg` folders
    (download: https://zenodo.org/record/3888300, AIDER.zip, 263MB). No
    official train/test split is shipped, so we take an 80/20 split here,
    the same approach as load_eurosat().

    Ref: Kyrkou & Theocharides (2020), "EmergencyNet: Efficient Aerial Image
         Classification for Drone-Based Emergency Monitoring Using Atrous
         Convolutional Feature Fusion." IEEE JSTARS.
    """
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    root = f"{data_dir}/AIDER"
    full = datasets.ImageFolder(root, transform=transform_train)

    n = len(full)
    n_train = int(0.8 * n)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    train_set = Subset(full, train_idx)
    test_full = datasets.ImageFolder(root, transform=transform_test)
    test_set = Subset(test_full, test_idx)
    return train_set, test_set


def load_dataset(name: str, data_dir: str) -> Tuple[Dataset, Dataset]:
    """Dispatcher. name: 'cifar10' | 'eurosat' | 'aider' | 'custom'"""
    if name == "cifar10":
        return load_cifar10(data_dir)
    elif name == "eurosat":
        return load_eurosat(data_dir)
    elif name == "aider":
        return load_aider(data_dir)
    elif name == "custom":
        return load_custom(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose cifar10 | eurosat | aider | custom")

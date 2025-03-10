import torch
from torch.utils.data import Dataset, TensorDataset


def load_in_memory_dataset(dataset: Dataset) -> TensorDataset:
    samples, labels = [], []
    for sample, label in dataset:
        samples.append(sample)
        labels.append(label)

    stacked_samples = torch.stack(samples)
    stacked_labels = torch.stack(labels)

    return TensorDataset(stacked_samples, stacked_labels)

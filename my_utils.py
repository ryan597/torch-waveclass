"""Modules to help with loading data from HDF5 format and viewing samples"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class H5Dataset(Dataset):
    """Dataset of image and label data in HDF5 format.

    Args:
        path (string) : path to the HDF5 file for train or test
        transforms (callable, optional): A transforms.Compose([...]) callable
            to transform or augment the images with before the appropriate
            transforms for the pretrained base model
        pretrain_transforms ()
    """
    def __init__(self, path, transform=None):
        self.file_path = path
        self.images = None
        self.labels = None
        self.transform = transform

        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file['images'])

    def __getitem__(self, index):
        if self.images is None:
            self.images = h5py.File(self.file_path, 'r')['images']
            self.labels = h5py.File(self.file_path, 'r')['labels']

        image = self.transform(self.images[index])
        # loss expects type long
        class_label = np.long(self.labels[index])
        return (image, class_label)

    def __len__(self):
        return self.dataset_len

def h5_dataloader(h5_filepath, transform, **kwargs):
    """Creates a torch.utils.data.DataLoader when supplied with a HDF5 filepath.
    It is up to user to supply the correct transform for the dataset and model.
    kwargs includes batch_size, shuffle and num_workers"""
    dataset = H5Dataset(h5_filepath, transform)
    dataloader = DataLoader(dataset, **kwargs)
    return dataloader


def view_batch(dataloader, num_samples=16, shape=(4, 4)):
    """View sample images with provided labels from the dataloader"""
    if shape[0]*shape[1] != num_samples:
        raise Exception("incorrect subplot shape, num_samples must equal shape[0]*shape[1]")

    classes = np.array(['plunge', 'spill', 'nonbreaking'])
    plt.figure(figsize=(10, 10))
    dataset = dataloader.dataset
    rand_indx = np.random.random_integers(0, len(dataset), num_samples)
    for i, _  in enumerate(dataset):
        plt.subplot(shape[0], shape[1], i+1)
        sample_image = dataset.images[rand_indx[i]]
        sample_label = dataset.labels[rand_indx[i]]

        plt.imshow(sample_image, cmap='gray')
        plt.title(f"{classes[sample_label]}")
        plt.axis("off")
        if i == (num_samples-1):
            break

def class_weight(dataset):
    """Calculates and returns class weights as a torch.Tensor"""
    classes = np.array([])
    class_totals = np.array([])
    i = 0
    while i < len(dataset):
        label = dataset[i][1]
        
        if label not in classes:
            classes = np.append(classes, label)
            class_totals = np.append(class_totals, 1)
        else:
            class_totals[int(label)] += 1
        i += 1

    weights = torch.FloatTensor(np.ones_like(classes) / class_totals)

    return weights

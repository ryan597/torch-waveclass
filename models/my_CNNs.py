"""
Class to give an instance of a pretrained CNN with a custom fully connected
layer on top.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    """Convolutional Neural Network model for the classification of input images
    Args:
        x (tensor): inputs into the CNN, must be a torch.Tensor with shape
            [B, 3, ...] with B the batch size, with 3 channels and arbitray image
            input size ...
        PLACEHOLDER
    Returns:
        x (tensor): outputs from the CNN forward pass as a torch.Tensor with
            dimension [B, C], where B is the batch size and C is the number of
            classes (3).
    """
    def __init__(self, base=None):
        super().__init__()

        model_dict = {
            "resnet18" : models.resnet18,
            "alexnet" : models.alexnet,
            "vgg16" : models.vgg16,
            "squeezenet" : models.squeezenet1_0,
            "densenet" : models.densenet161,
            "inception" : models.inception_v3,
            "googlenet" : models.googlenet,
            "mobilenet" : models.mobilenet_v2,
        }
        if base is None:
            base = 'resnet18'
        self.base = base

        # load the pretrained model
        base_model = model_dict[base]
        base_model = base_model(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False
        self.pretrained = nn.Sequential(*list(base_model.children())[:-1])

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

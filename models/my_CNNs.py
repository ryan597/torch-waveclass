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
        super(Net, self).__init__()

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
        pretrained = base_model(pretrained=True)
        for param in pretrained.parameters():
            param.requires_grad = False
        self.model = pretrained
        # remove the top from pretrained and replace with custom fully connected
        self.model.classifier = nn.Dropout(0.5)
        self.norm1 = nn.BatchNorm1d(1280)
        self.fc1 = nn.Linear(1280, 10)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu6(self.model(x))
        x = self.norm1(x)
        x = self.fc1(x)
        x = F.relu6(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x

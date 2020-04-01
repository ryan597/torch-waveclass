"""
Class to give an instance of a pretrained CNN with a custom fully connected
layer on top.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    """Convolutional Neural Network model for the classification of input images
    Args:
        x (tensor): inputs into the CNN, must be a torch.Tensor with shape
            [B, 3, ...] with B the batch size, with 3 channels and arbitray image
            input size.
    Returns:
        x (tensor): outputs from the CNN forward pass as a torch.Tensor with
            dimension [B, C], where B is the batch size and C is the number of
            classes (3).
    """
    def __init__(self, config):
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
            "resnet50"  : models.resnet50
        }

        self.base = config['base_model']
        in_channels = config['fc_in']
        out_channels = config['fc_out']

        # load the pretrained model
        base_model = model_dict[self.base]
        base_model = base_model(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False
        self.pretrained = nn.Sequential(*list(base_model.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, out_channels)
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, *args, **kwargs),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, *args, **kwargs),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layer_sizes, *args, **kwargs):
        super().__init__()
        down_sizes = [in_channels, *layer_sizes]
        up_sizes = [*layer_sizes[::-1] , out_channels]

        self.conv_down = nn.ModuleList([
            DoubleConv(in_ch, out_ch) for in_ch, out_ch in zip(down_sizes, down_sizes[1:])
        ])

        self.down_sample = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for i in range(len(layer_sizes)-1)
        ])

        self.up_sample = nn.ModuleList([
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2) for in_ch, out_ch in zip(up_sizes, up_sizes[1:-1])
        ])

        self.conv_up = nn.ModuleList([
            DoubleConv(in_ch, out_ch) for in_ch, out_ch in zip(up_sizes, up_sizes[1:])
        ])

    def forward(self, x):
        copies = []
        for i, conv in enumerate(self.conv_down[:-1]):
            x = conv(x)
            copies.append(x)
            x = self.down_sample[i](x)
        
        x = self.conv_down[-1](x)
        for i, conv in enumerate(self.conv_up[:-1]):
            x = self.up_sample[i](x)
            x = conv(torch.cat((x, copies[-(i+1)]), dim=1))

        x = self.conv_up[-1](x)
        return x

import torch.nn as nn
from torchvision import models

class ClassifyNet(nn.Module):

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

# model_definition.py
import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=120):
    model = models.resnet50()

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.6), 
        nn.Linear(512, num_classes)
    )

    return model

import torch
import torch.nn as nn
import torchvision.models.video as models

class ShopliftingDetectionModel(nn.Module):
    def __init__(self):
        super(ShopliftingDetectionModel, self).__init__()
        self.base_model = models.r3d_18(weights=None)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)

    def forward(self, x):
        return self.base_model(x)

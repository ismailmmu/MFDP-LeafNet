import torch
from torch import nn
from torchvision.models import resnet152, densenet121, mobilenet_v2,efficientnet_b0


class FeatureExtraction(nn.Module):
    def __init__(self, device):
        super(FeatureExtraction, self).__init__()
        self.resnet = resnet152(pretrained=True).to(device)
        self.resnet.fc = nn.Identity()
        #self.mobile = mobilenet_v2(pretrained=True).to(device)
        #self.mobile.classifier = nn.Identity()
        self.efficient = efficientnet_b0(pretrained=True).to(device)
        self.efficient.classifier = nn.Identity()

        self.dense = densenet121(pretrained=True).to(device)
        self.dense.classifier = nn.Identity()

    def forward(self, x):
        res_features = torch.flatten(self.resnet(x), start_dim=1)
        mobile_features = torch.flatten(self.mobile(x), start_dim=1)
        efficient_features = torch.flatten(self.efficient(x), start_dim=1)
        dense_features = torch.flatten(self.dense(x), start_dim=1)
        concatenated_features = torch.cat((res_features, mobile_features, dense_features), dim=1)
        return concatenated_features
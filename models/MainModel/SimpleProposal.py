import torch
import numpy as np
import torch.nn as nn
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.resnet import _resnet, resnet18, BasicBlock

class SimpleProposal(nn.Module):
    def __init__(self, num_proposal=10, length=256) -> None:
        super().__init__()
        self.length = length
        self.num_proposal = num_proposal
        self.neck = nn.Sequential(
            nn.Linear(self.length, self.length),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length, self.length),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length, self.length),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length, self.length),
            torch.nn.ReLU(inplace=True),
        )
        self.head_p = nn.Sequential(
            nn.Linear(self.length, self.length // 4),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.length // 4),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.length // 4),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.num_proposal),
            torch.nn.Sigmoid(),
        )
        self.head_c = nn.Sequential(
            nn.Linear(self.length, self.length // 4),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.length // 4),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.length // 4),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.num_proposal * 2),
            torch.nn.ReLU6(inplace=True),
        )

        self.norm = nn.BatchNorm2d(3)

    def forward(self, x):
        # x: x of fft
        x = self.norm(x)

        x_triu = torch.triu(x)
        x_triu_squeeze = torch.sum(x_triu, dim=-1)

        x_v = self.neck(x_triu_squeeze)

        p = self.head_p(x_v)
        c = self.head_c(x_v) / 6
        c_1 = c[:, :, :self.num_proposal]
        c_2 = c[:, :, self.num_proposal:]

        return {
            "x_proposal": {"p": p, "c_1": c_1, "c_2": c_2}, 
            "y_proposal": {"p": p, "c_1": c_1, "c_2": c_2}
        }
    



import torch
import numpy as np
import torch.nn as nn
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.resnet import _resnet, resnet18, BasicBlock


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResConvBlock(nn.Module):
    def __init__(
        self,
        inplanes: int = 3,
        planes: int = 3,
        stride: int = 1,
        groups: int = 3,
        dilation: int = 1,
        expanision: int = 1,
    ) -> None:
        super().__init__()
        self.expanision = expanision
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes * self.expanision, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(planes * self.expanision)
        self.conv3 = conv1x1(planes * self.expanision, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class FreqConvolution(nn.Module):
    def __init__(self, proposal_dim, width, inplanes=3, outplanes=3) -> None:
        super().__init__()
        self.width = width
        self.proposal_dim = proposal_dim
        if proposal_dim == -1:
            last_kernel_size = (self.width, 3)
            last_padding_size = (0, 1)
        elif proposal_dim == -2:
            last_kernel_size = (3, self.width)
            last_padding_size = (1, 0)
        else:
            raise NotImplementedError
        self.conv_body = nn.Sequential(
            ResConvBlock(inplanes=inplanes, planes=inplanes),
            ResConvBlock(inplanes=inplanes, planes=inplanes),
            ResConvBlock(inplanes=inplanes, planes=inplanes),
        )
        self.last_conv = nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=last_kernel_size,
            padding=last_padding_size,
            stride=1,
            groups=3,
            bias=True,
            dilation=1,
        )
    
    def forward(self, x):
        x = self.conv_body(x)
        x = self.last_conv(x)
        if self.proposal_dim == -1:
            x = x.squeeze(-2)
        else:
            x = x.squeeze(-1)
        return x



class FreqConvProposalSubProposal(nn.Module):
    def __init__(self, num_proposal=10, length=256, proposal_dim=-1, img_size=(256, 256), inplanes=3) -> None:
        super().__init__()
        assert proposal_dim == -1 or proposal_dim == -2
        self.proposal_dim = proposal_dim
        self.length = img_size[-1] if proposal_dim == -1 else img_size[-2]
        self.num_proposal = num_proposal
        self.width = img_size[0] if proposal_dim == -1 else img_size[1]
        self.freq_conv = FreqConvolution(self.proposal_dim, self.width, inplanes=inplanes)
        self.neck = nn.Sequential(
            nn.Linear(self.length, self.length),
            nn.ReLU(inplace=True),
            nn.Linear(self.length, self.length),
            nn.ReLU(inplace=True),
            nn.Linear(self.length, self.length),
            nn.ReLU(inplace=True),
            nn.Linear(self.length, self.length),
            nn.ReLU(inplace=True),
        )
        self.head_p = nn.Sequential(
            nn.Linear(self.length, self.length // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.length // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.length // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.num_proposal),
            nn.Sigmoid(),
        )
        self.head_c = nn.Sequential(
            nn.Linear(self.length, self.length // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.length // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.length // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.length // 4, self.num_proposal * 2),
            nn.Sigmoid(),
            # nn.ReLU6(inplace=True),
        )

        # self.prior = torch.linspace(0, 0, self.num_proposal) # 无先验
        self.prior = torch.linspace(0, 0.9, self.num_proposal) # 给定先验
        self.prior = torch.nn.Parameter(self.prior, requires_grad=False)
    
    def forward(self, x):
        x_squeeze = self.freq_conv(x)

        x_v = self.neck(x_squeeze)

        p = self.head_p(x_v)
        c = self.head_c(x_v)
        c_1 = c[:, :, :self.num_proposal]
        c_2 = c[:, :, self.num_proposal:]

        # prior
        B, C = c_1.shape[0], c_1.shape[1]
        prior = self.prior.repeat([B, C, 1])
        c_1 = c_1 + prior
        c_2 = c_2 + prior

        return {"p": p, "c_1": c_1, "c_2": c_2}



class FreqConvProposal(nn.Module):
    def __init__(self, num_proposal=10, length=256, img_size=(256, 129)) -> None:
        super().__init__()
        self.length = length
        self.num_proposal = num_proposal
        self.norm = nn.BatchNorm2d(3)
        self.x_proposaler = FreqConvProposalSubProposal(self.num_proposal, self.length, -2, img_size)
        self.y_proposaler = FreqConvProposalSubProposal(self.num_proposal, self.length, -1, img_size)
    
    def forward(self, x):
        proposal = {}

        x = self.norm(x)
        
        x_proposal = self.x_proposaler(x)
        y_proposal = self.y_proposaler(x)

        proposal['x_proposal'] = x_proposal
        proposal['y_proposal'] = y_proposal
        
        return proposal



class FreqConvProposalSingleSide(nn.Module):
    def __init__(self, num_proposal=10, length=256, img_size=(256, 129)) -> None:
        super().__init__()
        self.length = length
        self.num_proposal = num_proposal
        self.norm = nn.BatchNorm2d(3)
        self.proposaler = FreqConvProposalSubProposal(self.num_proposal, self.length, -2, img_size=img_size)
    
    def forward(self, x):
        proposal = {}

        x = self.norm(x)
        
        single_proposal = self.proposaler(x)

        proposal['x_proposal'] = single_proposal
        proposal['y_proposal'] = single_proposal
        
        return proposal


class FreqConvProposalNEW(nn.Module):
    def __init__(self, num_proposal=10, length=256, img_size=(256, 129)) -> None:
        super().__init__()
        self.length = length
        self.num_proposal = num_proposal
        self.norm_1 = nn.BatchNorm2d(3)
        self.norm_2 = nn.BatchNorm2d(3)
        self.x_proposaler = FreqConvProposalSubProposal(self.num_proposal, self.length, -2, img_size, inplanes=6)
        self.y_proposaler = FreqConvProposalSubProposal(self.num_proposal, self.length, -1, img_size, inplanes=6)
    
    def forward(self, x_1, x_2):
        proposal = {}

        x_1 = self.norm_1(x_1)
        x_2 = self.norm_2(x_2)

        B, C, H, W = x_1.shape

        x = torch.zeros_like(x_1)
        x = torch.cat([x, x.clone()], dim=1)  # B * 6 * H * W
        for i in range(C):
            x[:, 2 * i, :, :] = x_1[:, i, :, :]
            x[:, 2 * i + 1, :, :] = x_2[:, i, :, :]
        
        x_proposal = self.x_proposaler(x) # B * 3
        y_proposal = self.y_proposaler(x)

        proposal['x_proposal'] = x_proposal
        proposal['y_proposal'] = y_proposal
        
        return proposal


if __name__ == '__main__':
    x = torch.rand(size=(1, 3, 256, 256))
    m = FreqConvProposal()
    y = m(x)

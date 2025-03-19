import torch
import cv2
from math import ceil
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights, convnext_small, ConvNeXt_Small_Weights
from torchvision.models.resnet import _resnet, resnet18, resnet34, resnet50, BasicBlock, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from .swin_transformers import SwinTransformer
from .SimpleProposal import SimpleProposal
from .FreqConvProposal import FreqConvProposal, FreqConvProposalSingleSide, FreqConvProposalNEW

def random_crop(img, target_size):
    x, y = img.shape[:2]
    edge_min = min(x, y)
    if edge_min < target_size:
        x_new = ceil((x / edge_min * target_size))
        y_new = ceil((y / edge_min * target_size))
        img = cv2.copyMakeBorder(img, 0, x_new - x, 0, y_new - y, cv2.BORDER_REFLECT)
        x = x_new
        y = y_new
    c_x = np.random.randint(0, x - target_size) if x > target_size else 0
    c_y = np.random.randint(0, y - target_size) if y > target_size else 0
    img = img[c_x: c_x + target_size, c_y: c_y + target_size, :]
    return img

class SpaceBase(nn.Module):
    def __init__(self, r, dim, ema=False) -> None:
        super().__init__()
        assert r * 2 == dim
        I = torch.eye(r, r, dtype=torch.float)
        Z = torch.zeros_like(I)
        self.base = nn.ParameterDict({
            "semantic_base": nn.Parameter(torch.cat([I, Z], dim=1)),  # rank * dim (f_c // 2, f_c)
            "generative_base": nn.Parameter(torch.cat([Z, I], dim=1)),  # rank * dim (f_c // 2, f_c)
        })
        self.ema = ema
        if self.ema:
            self.beta = 0.95
            self.old = self.base
    
    def forward(self, feature):
        if self.ema:
            self.base['semantic_base'].data = self.beta * self.old['semantic_base'] + (1 - self.beta) * self.base['semantic_base']
            self.base['generative_base'].data = self.beta * self.old['generative_base'] + (1 - self.beta) * self.base['generative_base']
            self.old = self.base

        w_s, w_g = feature.mm(self.base['semantic_base'].transpose(0, 1)), feature.mm(self.base['generative_base'].transpose(0, 1))
        semantic_feature, generative_feature = w_s.mm(self.base['semantic_base']), w_g.mm(self.base['generative_base'])
        return semantic_feature, generative_feature


class FrequencyBranch(nn.Module):
    def __init__(self, backbone, thr=0.5, num_proposal=10, f_c=256, img_size=256, is_seperate=True, att=True) -> None:
        super().__init__()
        freq_size = (img_size, img_size // 2 + 1)
        self.is_seperate = is_seperate
        if self.is_seperate:
            self.proposaler_1 = FreqConvProposal(num_proposal=num_proposal, length=f_c, img_size=freq_size)
            self.proposaler_2 = FreqConvProposal(num_proposal=num_proposal, length=f_c, img_size=freq_size)
        else:
            self.proposaler = FreqConvProposalNEW(num_proposal=num_proposal, length=f_c, img_size=freq_size)
        self.iou_thr = thr
        self.num_proposal = num_proposal
        self.att = att

        if backbone == "resnet18":
            self.semantic_head = nn.ModuleList([
                resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(512, f_c)
            ])

            self.semantic_head[0].avgpool = nn.Identity()
            self.semantic_head[0].fc = nn.Identity()

            self.generative_head = nn.ModuleList([
                resnet18(weights=None),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(512, f_c)
            ])

            self.generative_head[0].avgpool = nn.Identity()
            self.generative_head[0].fc = nn.Identity()
        elif backbone == "resnet50":
            self.semantic_head = nn.ModuleList([
                resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(512, f_c)
            ])

            self.semantic_head[0].avgpool = nn.Identity()
            self.semantic_head[0].fc = nn.Identity()

            self.generative_head = nn.ModuleList([
                resnet50(weights=None),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(512, f_c)
            ])

            self.generative_head[0].avgpool = nn.Identity()
            self.generative_head[0].fc = nn.Identity()
        elif backbone == "resnet34":
            self.semantic_head = nn.ModuleList([
                resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(512, f_c)
            ])

            self.semantic_head[0].avgpool = nn.Identity()
            self.semantic_head[0].fc = nn.Identity()

            self.generative_head = nn.ModuleList([
                resnet34(weights=None),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(512, f_c)
            ])

            self.generative_head[0].avgpool = nn.Identity()
            self.generative_head[0].fc = nn.Identity()
        
        self.p_pred = nn.ModuleDict({
            "atten": nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True),
            "head": nn.Sequential(
                nn.Linear(512, 512 // 4),
                nn.ReLU(inplace=True),
                nn.Linear(512 // 4, 512 // 16),
                nn.ReLU(inplace=True),
                nn.Linear(512 // 16, 1),
                nn.Sigmoid(),
            )
        })
    
    @staticmethod
    def forward_head(head, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = head.conv1(x)
        x = head.bn1(x)
        x = head.relu(x)
        x = head.maxpool(x)

        x = head.layer1(x)
        x = head.layer2(x)
        x = head.layer3(x)
        x = head.layer4(x)
        return x
    
    def uncover_band(self, both_proposal: dict, nms=False):
        ret = {}

        for k, proposal in both_proposal.items():
            B, C, N = proposal["p"].shape
            p, c_1, c_2 = proposal["p"], proposal["c_1"], proposal["c_2"]
            
            # band width >= 0
            c_delta = torch.clamp(c_2 - c_1, 0)
            indiactor = torch.zeros_like(p).detach()
            indiactor[c_delta.nonzero(as_tuple=True)] = 1

            # nms
            if nms:
                for b in range(B):
                    for c in range(C):
                        _, indices = torch.sort(p[b, c, :], dim=-1, descending=True)
                        c_b1 = c_1[b, c, :]
                        c_b2 = c_2[b, c, :]
                        c_length = c_b2 - c_b1 + 0.00001
                
                        for i in indices:
                            if not indiactor[b, c, i]:
                                continue

                            c_x, c_y = c_1[b, c, i], c_2[b, c, i]
                            c_b1_max, c_b2_min = torch.maximum(c_b1, c_x), torch.minimum(c_b2, c_y)
                            c_b_delta = (c_b2_min - c_b1_max).clamp(min=0)

                            IoU = c_b_delta / c_length
                            single_indices = torch.nonzero(IoU>self.iou_thr)
                            for j in single_indices:
                                if i != j:
                                    indiactor[b, c, j] = 0
        
            p, c_1, c_2 = p * indiactor, c_1 * indiactor, c_2 * indiactor

            ret[k] = {"p": p, "c_1": c_1, "c_2": c_2}

        return ret
    
    def quantize(self, x_fft, both_proposal):
        assert x_fft.shape[1] == both_proposal['x_proposal']["p"].shape[1]
        B, C, W, H = x_fft.shape

        X, Y = np.ogrid[:W, :H]
        X, Y = torch.from_numpy(X).to(device=x_fft.device), torch.from_numpy(Y).to(device=x_fft.device)
        X, Y = X.repeat(B, C, self.num_proposal, 1, 1), Y.repeat(B, C, self.num_proposal, 1, 1)

        x_ffts = x_fft.repeat(self.num_proposal, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)  # B, C, N, W, H

        # mask
        px, py = both_proposal['x_proposal']["p"], both_proposal['y_proposal']["p"]
        ## inner mask
        cx_1, cy_1 = torch.floor(both_proposal['x_proposal']["c_1"] * W).clamp(min=0, max=W-1), \
            torch.floor(both_proposal['y_proposal']["c_1"] * H).clamp(min=0, max=H-1)
        ## outer mask
        cx_2, cy_2 = torch.floor(both_proposal['x_proposal']["c_2"] * W).clamp(min=0, max=W-1), \
            torch.floor(both_proposal['y_proposal']["c_2"] * H).clamp(min=0, max=H-1)

        px, py = px.unsqueeze(-1).unsqueeze(-1), py.unsqueeze(-1).unsqueeze(-1)
        cx_1, cx_2 = cx_1.unsqueeze(-1).unsqueeze(-1), cx_2.unsqueeze(-1).unsqueeze(-1)
        cy_1, cy_2 = cy_1.unsqueeze(-1).unsqueeze(-1), cy_2.unsqueeze(-1).unsqueeze(-1)
        
        # union p
        p = torch.sqrt(px * py + 1e-10)

        # cross band
        mask_x1, mask_x2 = torch.zeros_like(X), torch.zeros_like(X)
        mask_x1[X >= cx_1] = 1
        mask_x2[X <= cx_2] = 1
        mask_x = mask_x1 * mask_x2
 
        mask_y1, mask_y2 = torch.zeros_like(Y), torch.zeros_like(Y)
        mask_y1[Y >= cy_1] = 1
        mask_y2[Y <= cy_2] = 1
        mask_y = mask_y1 * mask_y2

        mask = mask_x + mask_y
        mask[mask > 1] = 1

        x_ffts_dirty = x_ffts * mask
        x_ffts_clean = x_ffts - x_ffts_dirty

        return x_ffts_clean, x_ffts_dirty, p
    
    def forward_respective(self, x, is_real_imag=False):
        B, C, W, H = x.shape
        x_fft = torch.fft.rfft2(x)
        if is_real_imag:
            comp1, comp2 = torch.real(x_fft), torch.imag(x_fft)
        else:
            comp1, comp2 = torch.abs(x_fft), torch.angle(x_fft)
        
        # comp1
        proposal_comp1_before = self.proposaler_1(comp1)
        proposal_comp1_after = self.uncover_band(proposal_comp1_before)
        x_comp1_clean, x_comp1_dirty, p_comp1 = self.quantize(comp1, proposal_comp1_after)

        # comp2
        proposal_comp2_before = self.proposaler_2(comp2)
        proposal_comp2_after = self.uncover_band(proposal_comp2_before)
        x_comp2_clean, x_comp2_dirty, p_comp2 = self.quantize(comp2, proposal_comp2_after)

        # post-process
        if is_real_imag:
            x_ffts_dirty, x_ffts_clean = torch.complex(x_comp1_dirty, x_comp2_dirty), torch.complex(x_comp1_clean, x_comp2_clean)
        else:
            x_ffts_dirty, x_ffts_clean = torch.complex(x_comp1_dirty * torch.cos(x_comp2_dirty), x_comp1_dirty * torch.sin(x_comp2_dirty)), \
                                         torch.complex(x_comp1_clean * torch.cos(x_comp2_clean), x_comp1_clean * torch.sin(x_comp2_clean))
        x_dirty, x_clean = torch.fft.irfft2(x_ffts_dirty), torch.fft.irfft2(x_ffts_clean)  # B, C, N, W, H

        separate_imgs_dirty = x_dirty.transpose(1, 2)
        separate_imgs_clean = x_clean.transpose(1, 2)

        x_dirty, x_clean = x_dirty.transpose(1, 2).reshape(B * self.num_proposal, -1, W, H), x_clean.transpose(1, 2).reshape(B * self.num_proposal, -1, W, H) # (B, N), C, W, H

        generative_feature_per_channel, semantic_feature_per_channel = \
                                self.forward_head(self.generative_head[0], x_dirty), self.forward_head(self.semantic_head[0], x_clean) # (B, N), C, 7, 7
        generative_feature_per_channel, semantic_feature_per_channel = \
                                self.generative_head[1](generative_feature_per_channel), self.semantic_head[1](semantic_feature_per_channel)
        generative_feature_per_channel, semantic_feature_per_channel = torch.flatten(generative_feature_per_channel, 1), torch.flatten(semantic_feature_per_channel, 1) # (B, N), C
        
        # weights
        _, C = generative_feature_per_channel.shape
        generative_feature_per_channel, semantic_feature_per_channel = \
                                generative_feature_per_channel.reshape(B, self.num_proposal, C), semantic_feature_per_channel.reshape(B, self.num_proposal, C) # B, N, C
        f = self.p_pred['atten'](query=generative_feature_per_channel, key=generative_feature_per_channel, value=generative_feature_per_channel, need_weights=False)
        p = self.p_pred['head'](f[0]) # # B, N, 1

        if self.num_proposal > 15:
            t, _ = torch.kthvalue(p.squeeze(-1), 15)
            p[p < t.unsqueeze(-1).unsqueeze(-1)] = 0

        generative_feature, semantic_feature = self.generative_head[2](generative_feature_per_channel), self.semantic_head[2](semantic_feature_per_channel) # B, N, f
        if self.att:
            generative_feature, semantic_feature = (generative_feature * p).sum(dim=1), semantic_feature.mean(1)
        else:
            generative_feature, semantic_feature = generative_feature.mean(1), semantic_feature.mean(1)

        return {
            "imgs_dirty": separate_imgs_dirty,
            "imgs_clean": separate_imgs_clean,
            "generative_feature_frequency": generative_feature,
            "semantic_feature_frequency": semantic_feature,
            "proposal_before": [proposal_comp1_before, proposal_comp2_before],
            "proposal_after": [proposal_comp1_after, proposal_comp2_after],
            "proposal_p": p
            }
    
    def forward_together(self, x, is_real_imag=False):
        B, C, W, H = x.shape
        x_fft = torch.fft.rfft2(x)
        if is_real_imag:
            comp1, comp2 = torch.real(x_fft), torch.imag(x_fft)
        else:
            comp1, comp2 = torch.abs(x_fft), torch.angle(x_fft)
        
        proposal_before = self.proposaler(comp1, comp2)
        proposal_after = self.uncover_band(proposal_before)
        x_clean, x_dirty, p_comp1 = self.quantize(x_fft, proposal_after)

        x_clean, x_dirty = torch.fft.irfft2(x_clean), torch.fft.irfft2(x_dirty)  # B, C, N, W, H

        x_dirty, x_clean = x_dirty.transpose(1, 2).reshape(B * self.num_proposal, -1, W, H), x_clean.transpose(1, 2).reshape(B * self.num_proposal, -1, W, H) # (B, N), C, W, H

        generative_feature_per_channel, semantic_feature_per_channel = \
                                self.forward_head(self.generative_head[0], x_dirty), self.forward_head(self.semantic_head[0], x_clean) # (B, N), C, 7, 7
        generative_feature_per_channel, semantic_feature_per_channel = \
                                self.generative_head[1](generative_feature_per_channel), self.semantic_head[1](semantic_feature_per_channel)
        generative_feature_per_channel, semantic_feature_per_channel = torch.flatten(generative_feature_per_channel, 1), torch.flatten(semantic_feature_per_channel, 1) # (B, N), C
        
        # weights
        _, C = generative_feature_per_channel.shape
        generative_feature_per_channel, semantic_feature_per_channel = \
                                generative_feature_per_channel.reshape(B, self.num_proposal, C), semantic_feature_per_channel.reshape(B, self.num_proposal, C) # B, N, C
        f = self.p_pred['atten'](query=generative_feature_per_channel, key=generative_feature_per_channel, value=generative_feature_per_channel, need_weights=False)
        p = self.p_pred['head'](f[0]) # # B, N, 1

        generative_feature, semantic_feature = self.generative_head[2](generative_feature_per_channel), self.semantic_head[2](semantic_feature_per_channel) # B, N, f

        generative_feature, semantic_feature = (generative_feature * p).sum(dim=1), semantic_feature.mean(1)

        return {
            "generative_feature_frequency": generative_feature,
            "semantic_feature_frequency": semantic_feature,
            "proposal_before": [proposal_before],
            "proposal_after": [proposal_after],
            "proposal_p": p
        }

    def forward(self, x):
        if self.is_seperate:
            return self.forward_respective(x)
        else:
            return self.forward_together(x)


class SpitalBranch(nn.Module):
    def __init__(self, backbone="convnext_base", f_c=256, base_ema=True, img_size=200, base=True) -> None:
        super().__init__()
        self.base = base

        assert f_c == 256
        if self.base:
            self.f_c = f_c
            self.space_base = SpaceBase(f_c // 2, f_c, ema=base_ema)
        else:
            self.f_c = f_c * 2
            self.space_base = None

        if backbone == "convnext_base":
            self.backbone = convnext_base(weights=ConvNeXt_Base_Weights)
            c = self.backbone.classifier[-1].in_features
            self.backbone.classifier.pop(-1)
            self.backbone.classifier.append(torch.nn.Linear(c, self.f_c))
            self.backbone.classifier.append(torch.nn.ReLU(inplace=True))
            self.backbone.classifier.append(torch.nn.Linear(self.f_c, self.f_c))
        elif backbone == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            c = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(c, self.f_c),
                nn.ReLU(inplace=True),
                nn.Linear(self.f_c, self.f_c)
            )
        elif backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            c = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(c, self.f_c),
                nn.ReLU(inplace=True),
                nn.Linear(self.f_c, self.f_c)
            )
        else:
            self.backbone = SwinTransformer(num_classes=self.f_c, img_size=img_size, window_size=7)
            
    def forward(self, x):
        feature = self.backbone(x)
        if self.base:
            semantic_feature, generative_feature = self.space_base(feature)
        else:
            semantic_feature, generative_feature = torch.split(feature, self.f_c // 2, dim=-1)

        return {
            "semantic_feature_spital": semantic_feature, 
            "generative_feature_spital": generative_feature, 
            "semantic_base": self.space_base.base['semantic_base'] if self.base else None,
            "generative_base": self.space_base.base['generative_base'] if self.base else None
        }


class MainModel(nn.Module):
    def __init__(self, backbone_s, backbone_f, feature_dim, num_proposal, base_ema, img_size, is_seperate=True, base=True, att=True) -> None:
        super().__init__()
        self.f_c = feature_dim
        self.spital_branch = SpitalBranch(
            backbone_s, 
            f_c=self.f_c, 
            base_ema=base_ema, 
            img_size=img_size, 
            base=base
        )
        self.frequency_branch = FrequencyBranch(
            backbone_f, 
            f_c=self.f_c, 
            img_size=img_size, 
            num_proposal=num_proposal, 
            is_seperate=is_seperate, 
            att=att
        )

        self.classificator_generative = nn.Sequential(
            nn.Linear(self.f_c, self.f_c // 2),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.f_c // 2, self.f_c // 8),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.f_c // 8, self.f_c // 24),
            torch.nn.ReLU(inplace=True),
            nn.Linear(self.f_c // 24, 2)
        )

    def forward(self, x, other_model=None):
        ret_dict = {}
        spital_feature_keys = self.spital_branch(x['img'])
        Frequency_feature_keys = self.frequency_branch(x['img'])
        ret_dict.update(spital_feature_keys)
        ret_dict.update(Frequency_feature_keys)

        ret_dict['logits_f'] = self.classificator_generative(ret_dict['generative_feature_frequency'])
        ret_dict['logits_s'] = self.classificator_generative(ret_dict['generative_feature_spital'])
        
        ret_dict['logits'] = self.classificator_generative(
            (ret_dict['generative_feature_spital'] + ret_dict['generative_feature_frequency']) / 2
        )

        return ret_dict



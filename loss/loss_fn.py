import torch
from PIL import Image
import torch.nn.functional as F

def CELoss(x, other_model=None):
    return F.cross_entropy(x['model_out']['logits'], x['label'], reduction='none').mean() + \
                F.cross_entropy(x['model_out']['logits_f'], x['label'], reduction='none').mean() + \
                F.cross_entropy(x['model_out']['logits_s'], x['label'], reduction='none').mean()

def GenLoss(x, other_model=None):
    f1 = x['model_out']['generative_feature_spital']
    f2 = x['model_out']['generative_feature_frequency']
    # loss = f1.shape[0] - F.cosine_similarity(f1, f2, dim=-1).sum(dim=-1).mean()
    loss = F.mse_loss(f1, f2, reduction='none').sum(dim=-1).mean()
    return loss

def SELoss(x, other_model=None):
    f1 = x['model_out']['semantic_feature_spital'].unsqueeze(1)
    f2 = x['model_out']['semantic_feature_frequency'].unsqueeze(1)
    loss1 = F.mse_loss(f1, f2, reduction='none').sum(dim=-1).mean()
    if 'features_blip' in x['model_out'].keys():
        loss2 = F.mse_loss(f1, x['model_out']['features_blip'], reduction='none').sum(dim=-1).mean()
        loss3 = F.mse_loss(f2, x['model_out']['features_blip'], reduction='none').sum(dim=-1).mean()
    return loss1 + loss2 + loss3

def SELossSingle1(x, other_model=None):
    f1 = x['model_out']['semantic_feature_spital'].unsqueeze(1)
    f2 = x['model_out']['semantic_feature_frequency'].unsqueeze(1)
    loss1 = F.mse_loss(f1, f2, reduction='none').sum(dim=-1).mean()
    return loss1

def SELossSingle2(x, other_model=None):
    f1 = x['model_out']['semantic_feature_spital'].unsqueeze(1)
    f2 = x['model_out']['semantic_feature_frequency'].unsqueeze(1)
    if 'features_blip' in x['model_out'].keys():
        loss2 = F.mse_loss(f1, x['model_out']['features_blip'], reduction='none').sum(dim=-1).mean()
    else:
        raise RuntimeError
    return loss2

def SELossSingle3(x, other_model=None):
    f1 = x['model_out']['semantic_feature_spital'].unsqueeze(1)
    f2 = x['model_out']['semantic_feature_frequency'].unsqueeze(1)
    if 'features_blip' in x['model_out'].keys():
        loss3 = F.mse_loss(f2, x['model_out']['features_blip'], reduction='none').sum(dim=-1).mean()
    else:
        RuntimeError
    return loss3

def BandLoss(x, other_model=None, eps=0.01):
    ret = 0
    # c1 <= c2
    for prp in x['model_out']['proposal_before']:
        for _, v in prp.items():
            c1 = v['c_1']
            c2 = v['c_2']
            c_delta = (- (c2 - c1 - eps)).clamp(0)
            ret += c_delta.sum(dim=-1).sum(dim=-1).mean()

    # c1 < 1, c2 < 1
    for prp in x['model_out']['proposal_before']:
        for _, v in prp.items():
            c1 = v['c_1']
            c2 = v['c_2']
            ret += (c1 - 1).clamp(0).sum(dim=-1).sum(dim=-1).mean() + (c2 - 1).clamp(0).sum(dim=-1).sum(dim=-1).mean()
    return ret

def PullLoss(x, other_model=None):
    semantic_base, generative_base = x['model_out']["semantic_base"], x['model_out']["generative_base"]   # rank * dim
    semantic_base, generative_base = \
        semantic_base / semantic_base.norm(dim=-1, keepdim=True), generative_base / generative_base.norm(dim=-1, keepdim=True)
    loss = torch.abs(semantic_base.mm(generative_base.transpose(0, 1)))
    return loss.sum()

def OrthLoss(x, other_model=None):
    semantic_base, generative_base = x['model_out']["semantic_base"], x['model_out']["generative_base"]   # rank * dim
    semantic_base_orth, generative_base_orth = \
        semantic_base.mm(semantic_base.transpose(0, 1)), generative_base.mm(generative_base.transpose(0, 1))
    rank = semantic_base.shape[0]
    I = torch.eye(rank, device=semantic_base.device)
    loss = (semantic_base_orth - I).abs().sum() + (generative_base_orth - I).abs().sum()
    return loss 

def MILoss(x, other_model=None):
    return other_model['CLUB'].eval(x)['mi_loss']

def MILearningLoss(x, other_model=None):
    return x['model_out']['learning_loss']


def IdentityLoss(x, other_model=None):
    return x['model_out']["loss_D"]

def PullLoss2(x, other_model=None):
    semantic_base, generative_base = x['model_out']["semantic_base1"], x['model_out']["generative_base1"]   # rank * dim
    semantic_base, generative_base = \
        semantic_base / semantic_base.norm(dim=-1, keepdim=True), generative_base / generative_base.norm(dim=-1, keepdim=True)
    loss = torch.abs(semantic_base.mm(generative_base.transpose(0, 1))).sum()

    semantic_base, generative_base = x['model_out']["semantic_base2"], x['model_out']["generative_base2"]   # rank * dim
    semantic_base, generative_base = \
        semantic_base / semantic_base.norm(dim=-1, keepdim=True), generative_base / generative_base.norm(dim=-1, keepdim=True)
    loss += torch.abs(semantic_base.mm(generative_base.transpose(0, 1))).sum()
    return loss
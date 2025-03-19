import torch.nn as nn
from . import loss_fn as L

class LossWarper(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
    
    def forward(self, x, other_model=None):
        loss_info = {}
        loss = 0
        for loss_name, loss_kwargs in self.cfg.items():
            loss_info[loss_name] = loss_kwargs['scale'] * getattr(L, loss_name)(x, other_model=other_model, **loss_kwargs['kwargs'])
            mi, ma = loss_kwargs['min'] if "min" in loss_kwargs.keys() else None, loss_kwargs['max'] if "max" in loss_kwargs.keys() else None
            if mi and loss_info[loss_name] < mi:
                    loss_info[loss_name] = loss_info[loss_name] / loss_info[loss_name].norm() * (- mi)
            if ma and loss_info[loss_name] > ma:
                 loss_info[loss_name] = loss_info[loss_name] / loss_info[loss_name].norm() * ma
            loss += loss_info[loss_name]
        loss_info["all"] = loss
        return loss_info
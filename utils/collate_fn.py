import torch
import numpy as np

def collate_fn(batch):
    B = len(batch)

    assert B > 0, "Empty batch!"

    keys = batch[0].keys()
    ret = {}

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            ret[key] = [batch[0][key].unsqueeze(0)]

            for i in range(1, B):
                ret[key].append(batch[i][key].unsqueeze(0))
            ret[key] = torch.cat(ret[key], dim=0)
        else:
            ret[key] = [batch[0][key]]

            for i in range(1, B):
                ret[key].append(batch[i][key])
    return ret

def collate_fn_pairs(batch):
    B = len(batch)

    assert B > 0, "Empty batch!"

    keys = batch[0].keys()
    ret = {}

    for key in keys:
        if isinstance(batch[0][key], list) and isinstance(batch[0][key][0], torch.Tensor):
            ret[key] = [batch[0][key][0].unsqueeze(0), batch[0][key][1].unsqueeze(0)]

            for i in range(1, B):
                ret[key].append(batch[i][key][0].unsqueeze(0))
                ret[key].append(batch[i][key][1].unsqueeze(0))
            ret[key] = torch.cat(ret[key], dim=0)
        else:
            ret[key] = [batch[0][key][0], batch[0][key][1]]

            for i in range(1, B):
                ret[key].append(batch[i][key][0])
                ret[key].append(batch[i][key][1])
    return ret
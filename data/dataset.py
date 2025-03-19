import torch
import copy
import json
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from . import tools as Tls
from . import transformers as Tfs

class ImgDataset(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.data_list = open(cfg['data_json'], 'r').readlines()
        self.preprocess = cfg['preprocess']
        self.data_aug = cfg['data_aug']

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        info = json.loads(self.data_list[index])

        img = Image.open(info['img_path']).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        assert img is not None, "Img read error at {}".format(info['img_path'])

        # if "png" in info['img_path'].lower():
        #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        #     result, encimg = cv2.imencode(".jpg", img, encode_param)
        #     img = cv2.imdecode(encimg, 1)

        for name, kwarg in self.data_aug.items():
            img = getattr(Tfs, name)(img, **kwarg)

        img_BGR_np = copy.deepcopy(img)

        for name, kwarg in self.preprocess.items():
            img = getattr(Tls, name)(img, **kwarg)
            if "norm" not in name:
                img_BGR_np = getattr(Tls, name)(img_BGR_np, **kwarg)

        info["img_BGR_np"] = img_BGR_np

        info['img'] = img
        Tls.cvtTensor(info)
        return info


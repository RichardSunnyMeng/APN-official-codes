import os
import datetime
import sys
import json
import torch
import logging
import argparse

from torch.utils.tensorboard import SummaryWriter   
from models import ModelWarper
from utils import log_creator, print_args

def train_pipeline(cfg):
    logger = log_creator(
        os.path.join(
            cfg['training_cfg']['log_path'], "train." + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ".log"
        )
    )
    writter = SummaryWriter(cfg['training_cfg']['log_path'])
    print_args(cfg, logger)

    model_dict = {}
    for name, model_cfg in cfg['models'].items():
        model_dict[name] = ModelWarper(name, model_cfg, logger, writter)
    
    epoch_num = cfg['training_cfg']['running_epoch']

    iter_per_epoch = 0
    for _, model in model_dict.items():
        if model.running_iter == 0:
            continue
        iter_per_epoch = max(iter_per_epoch, len(model.dataloader) // model.running_iter)

    for epoch in range(epoch_num):
        # train
        for iter in range(iter_per_epoch):
            for name, model in model_dict.items():
                if model.running_iter == 0:
                    continue
                model.train(epoch, iter, iter_per_epoch, model_dict)

        # save
        for name, model in model_dict.items():
            model.save_checkpoint(epoch)

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--train_cfg",
        default="/home/mengzheling/research_framework/cfg/train.json",
        type=str,
    )
    arg = arg.parse_args()

    f = open(arg.train_cfg, "r")
    cfg = json.load(f)

    train_pipeline(cfg)

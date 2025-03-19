import os
import datetime
import sys
import json
import torch
import logging
import argparse

from torch.utils.tensorboard import SummaryWriter   
from torch.utils.data import DataLoader

import utils
from models import ModelWarperEval
from utils import log_creator, print_args
from data import build_dataset

def eval_pipeline(cfg):
    logger = log_creator(
            os.path.join(cfg['log_path'], "eval." + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ".log"
        )
    )
    print_args(cfg, logger)

    model_dict = {}
    for name, model_cfg in cfg['models'].items():
        model_dict[name] = ModelWarperEval(name, model_cfg, logger)
    
    eval_lists = cfg['eval_dataset']
    dataset_cfg = cfg['data_cfg']['dataset']
    dataloader_cfg = cfg['data_cfg']['data_loader']

    outs = {}
    for model_name, model_cfg in cfg['models'].items():
        model = model_dict[model_name]
        outs_per_model = {}
        for eval_name, eval_list in eval_lists.items():
            out = {}
            for idx, (eval_dataset_name, eval_dataset_path) in enumerate(eval_list.items()):
                dataset_cfg['data_json'] = eval_dataset_path
                eval_dataset = build_dataset(dataset_cfg)
                eval_dataloader = DataLoader(eval_dataset, 
                                     batch_size=dataloader_cfg['batch_size'], 
                                     shuffle=dataloader_cfg['shuffle'],
                                     num_workers=8,
                                     collate_fn=getattr(utils, dataloader_cfg['collate_fn']))

                logger.info("Model {} start to eval dataset {} ({}/{})".format(model_name, eval_dataset_name, idx, len(eval_list.items())))

                results = {"logits": [], "label": [], "p": []}

                model.start_eval()
                for idx, x in enumerate(eval_dataloader):
                    with torch.no_grad():
                        x['model_out'] = model.eval(x)
                    print("Testing batch {}/{}".format(idx, len(eval_dataloader))) 

                metrics_test = model.cal_metrics()

                logger.info("Test dataset {}: auc {:.3f} acc {:.3f} recall {:.3f} precision {:.3f} recall_r {:.3f} precision_r {:.3f} ".format(
                                eval_dataset_name, metrics_test['auc'], metrics_test['acc'], 
                                metrics_test['pos_recall'], metrics_test['pos_precision'], 
                                metrics_test['neg_recall'], metrics_test['neg_precision'], 
                    ))
                
                out[eval_dataset_name] = metrics_test['acc']
            outs_per_model[eval_name] = out
        outs[model_name] = outs_per_model
    
    # print
    for model_name, outs_per_model in outs.items():
        logger.info(model_name)
        for domain_name, out in outs_per_model.items():
            logger.info(domain_name)
            for data_name, value_ in out.items():
                logger.info("{:<30s}: {:1.3f}".format(data_name, value_))
                logger.info("")

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--eval_cfg",
        default="/home/mengzheling/research_framework/cfg/eval.json",
        type=str,
    )
    arg = arg.parse_args()

    f = open(arg.eval_cfg, "r")
    cfg = json.load(f)

    eval_pipeline(cfg)

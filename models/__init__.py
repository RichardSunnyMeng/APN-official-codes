from .MainModel import MainModel


import os
import torch
import numpy as np
from torchsummary import summary_logger
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import utils
import time
from data import build_dataset
from loss import LossWarper

model_list = {"MainModel": MainModel}

class ModelWarper():
    def __init__(self, name:str, cfg:dict, logger=None, SummaryWriter=None) -> None:
        self.name = name
        self.summary_writer = SummaryWriter
        self.logger = logger
        self.device = cfg['device']
        self.model = model_list[name](**cfg['model_cfg']).to(self.device)
        self.save = cfg['save']
        self.dataset = build_dataset(cfg['data']) if "data" in cfg.keys() else None
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=cfg['data_loader']['batch_size'], 
                                     shuffle=cfg['data_loader']['shuffle'],
                                     num_workers=cfg['data_loader']['num_workers'],
                                     collate_fn=getattr(utils, cfg['data_loader']['collate_fn'])) if "data_loader" in cfg.keys() else None
        if self.dataloader:
            self.data_iter = iter(self.dataloader)
            self.loader_info = {"all_iter": len(self.dataloader), "cur_iter": 0}
        self.running_iter = cfg['data_loader']['running_iter'] if "data_loader" in cfg.keys() else None

        if "optimizer" in cfg.keys():
            if "groups" in cfg['optimizer'].keys():
                params = []
                base = {"name": [], "params": [], "lr": cfg['optimizer']["groups"]["others"]}
                for set_name, lr in cfg['optimizer']["groups"].items():
                    if set_name == "others":
                        continue
                    tmp = {"name": [], "params": [], "lr": lr}
                    for m_name, param in self.model.named_parameters():
                        if set_name in m_name and param.requires_grad:
                            tmp["params"].append(param)
                            tmp["name"].append(m_name)
                    params.append(tmp)
                for m_name, param in self.model.named_parameters():
                    flag = True
                    for set_name, lr in cfg['optimizer']["groups"].items():
                        if set_name in m_name:
                            flag = False
                            break
                    if flag:
                        base["params"].append(param)
                        base["name"].append(m_name)
                params.append(base)
                for param in params:
                    self.logger.info("Parameters: {}, learning rate {}".format(param["name"], param["lr"]))
                self.optimizer = getattr(optim, cfg['optimizer']['type'])(params, **cfg['optimizer']['kwargs'])
            else:
                self.optimizer = getattr(optim, cfg['optimizer']['type'])(self.model.parameters(), **cfg['optimizer']['kwargs'])
        self.optim_warm_up = getattr(lr_scheduler, cfg['scheduler_warmup']['type'])(self.optimizer, **cfg['scheduler_warmup']['kwargs']) if "scheduler_warmup" in cfg.keys() else None
        self.optim_decay = getattr(lr_scheduler, cfg['scheduler_decay']['type'])(self.optimizer, **cfg['scheduler_decay']['kwargs']) if "scheduler_decay" in cfg.keys() else None

        self.loss_warper = LossWarper(cfg['loss']) if "loss" in cfg.keys() else None

        self.init(cfg)
        summary_logger(self.model)
    
    def init(self, cfg):
        if "checkpoint" in cfg.keys():
            if os.path.exists(cfg['checkpoint']) and str.endswith(cfg['checkpoint'], ".pth"):
                state_dict = torch.load(cfg['checkpoint'], map_location=torch.device('cpu'))
                missing, unexpected = self.model.load_state_dict(state_dict=state_dict['model'], strict=False)
                # if optimizer:
                #     optimizer.load_state_dict(state_dict['optimizer'])
                # if scheduler:
                #     scheduler.load_state_dict(state_dict['scheduler'])
                self.logger.info("Load model {} checkpoint success!".format(self.name))
                self.logger.info("Missing Module: {}".format(missing))
                self.logger.info("Unexpected Module: {}".format(unexpected))
    
    def save_checkpoint(self, epoch):
        if epoch < self.save['save_start_epoch'] and epoch % self.save['save_interval'] != 0:
            return
        if not os.path.exists(self.save['ckpt_path']):
            os.makedirs(self.save['ckpt_path'])
        state = {}
        state['model'] = self.model.state_dict()
        if self.optimizer:
            state['optimizer'] = self.optimizer.state_dict()
        if self.optim_decay:
            state['scheduler'] = self.optim_decay.state_dict()
        state = torch.save(state, os.path.join(self.save['ckpt_path'], self.name + "_epoch_{}.pth".format(epoch)))
    
    def train(self, epoch, cur_iter, iter_per_epoch, other_models=None):
        self.model.train()
        for idx in range(self.running_iter):
            try:
                x = next(self.data_iter)
                self.loader_info['cur_iter'] += 1
            except:
                self.data_iter = iter(self.dataloader)
                self.loader_info['cur_iter'] = 1
                x = next(self.data_iter)
            utils.toDevice(self.device, x)
            
            x['model_out'] = self.model(x, other_models)
            loss = self.loss_warper(x, other_models)
            self.write_log(self.summary_writer, loss, epoch * iter_per_epoch + cur_iter)
            lr_dict = {}
            for i in range(len(self.optimizer.state_dict()['param_groups'][0])):
                lr_dict["lr_{}".format(i)] = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.write_log(self.summary_writer, lr_dict, epoch * iter_per_epoch + cur_iter)

            self.optimizer.zero_grad()
            loss['all'].backward()

            # for n, p in self.model.named_parameters():
            #     print(n)
            #     print(p.grad)
            # time.sleep(10)

            self.optimizer.step()

            if self.optim_warm_up is not None:
                self.optim_warm_up.step()
            if self.optim_decay is not None:
                self.optim_decay.step()

            lr_log = ""
            for i in range(len(self.optimizer.param_groups)):
                lr_log += format(self.optimizer.param_groups[i]['lr'], ".5f") + " "
            
            self.logger.info("Epoch {}: Model {} Train_loss {:.4f} batch {}/{} running {}/{} ".format(
                    epoch, self.name, 
                    loss['all'].detach().cpu().item(), 
                    self.loader_info['cur_iter'], self.loader_info['all_iter'],
                    cur_iter + 1, iter_per_epoch) + lr_log
                )
            self.log_print_loss(loss, "         ")
    
    def eval(self, x):
        self.model.eval()
        utils.toDevice(self.device, x)
        model_out = self.model(x)
        return model_out

    def write_log(self, writter, info: dict, epoch):
        if writter:
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                writter.add_scalar(self.name + "_" + k, v, epoch)

    def log_print_loss(self, loss, prefix=""):
        for k, v in loss.items():
            if "Loss" in k:
                prefix += "{} {:.4f} ".format(k, v.detach().cpu().item())
        self.logger.info(prefix)


class ModelWarperEval():
    def __init__(self, name:str, cfg:dict, logger=None, SummaryWriter=None) -> None:
        self.name = name
        self.summary_writer = SummaryWriter
        self.device = cfg['device']
        self.model = model_list[name](**cfg['model_cfg']).to(self.device)
        self.model.eval()
        self.metrics_cfg = cfg['metrics'] if "metrics" in cfg.keys() else None

        self.logger = logger

        self.init(cfg)
        summary_logger(self.model)

        self.results = {'logits': [], 'label': []}
    
    def init(self, cfg):
        if "checkpoint" in cfg.keys():
            if os.path.exists(cfg['checkpoint']) and str.endswith(cfg['checkpoint'], ".pth"):
                state_dict = torch.load(cfg['checkpoint'], map_location=torch.device('cpu'))
                missing, unexpected = self.model.load_state_dict(state_dict=state_dict['model'], strict=False)
                # if optimizer:
                #     optimizer.load_state_dict(state_dict['optimizer'])
                # if scheduler:
                #     scheduler.load_state_dict(state_dict['scheduler'])
                self.logger.info("Load model {} checkpoint success!".format(self.name))
                self.logger.info("Missing Module: {}".format(missing))
                self.logger.info("Unexpected Module: {}".format(unexpected))
            else:
                self.logger.info("Load model {} fail: {}".format(self.name, cfg['checkpoint']))
        else:
            self.logger.info("No KEY checkpoint exists in configs.")
    
    def eval(self, x):
        utils.toDevice(self.device, x)
        model_out = self.model(x)

        self.results['logits'].append(model_out['logits'].detach().cpu().numpy())
        self.results["label"].append(x['label'].detach().cpu().numpy())
        return model_out
    
    def start_eval(self):
        self.results = {'logits': [], 'label': []}

    def cal_metrics(self):
        if self.metrics_cfg is not None:
            return getattr(utils, self.metrics_cfg['method'])(self.results, **self.metrics_cfg['kwargs'])

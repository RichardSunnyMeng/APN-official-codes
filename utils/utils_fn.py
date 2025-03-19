import sys
import torch
import json
import logging

def toDevice(device, x):
    keys = x.keys()
    for key in keys:
        if isinstance(x[key], torch.Tensor):
            x[key] = x[key].to(device)


def log_creator(path):
 
    # 创建一个日志器
    logger = logging.getLogger("logger")
    logger.propagate = False
 
    # 设置日志输出的最低等级,低于当前等级则会被忽略
    logger.setLevel(logging.INFO)
 
    # 创建处理器：sh为控制台处理器，fh为文件处理器
    sh = logging.StreamHandler()
 
    # 创建处理器：sh为控制台处理器，fh为文件处理器,log_file为日志存放的文件夹
    fh = logging.FileHandler(path,encoding="UTF-8")
 
    # 创建格式器,并将sh，fh设置对应的格式
    formator = logging.Formatter(fmt = "%(asctime)s %(levelname)s %(message)s",
                                         datefmt="%Y/%m/%d %X")
    sh.setFormatter(formator)
    fh.setFormatter(formator)
 
    # 将处理器，添加至日志器中
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def print_args(arg, logger):
    try:
        for k,v in sorted(vars(arg).items()):
            logger.info('{} = {}'.format(k, v))
    except:
        logger.info(json.dumps(arg, sort_keys=True, indent=4, separators=(',', ':')))


def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []
 
    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """
 
        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)
 
            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)
 
    unfoldLayer(model)
    return layers

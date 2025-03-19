from . import dataset as D

def build_dataset(cfg):
    dataset = getattr(D, cfg['dataset_type'])(cfg)
    return dataset
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_data_src')
def set_cfg_data_source(cfg):
    """Configuration block for external graph/label directories."""
    cfg.data_src = CN()
    cfg.data_src.graphml_dir = ""
    cfg.data_src.label_dir = ""
    cfg.data_src.train_ratio = 0.8
    cfg.data_src.random_seed = 42
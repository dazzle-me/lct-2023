import os
import os.path as osp
from typing import List, Dict

from dataclasses import dataclass, field

# @dataclass(frozen=False)
class CFG:
    ## dataset

    cache_dir: str = '/storage/nvme1/hacks/ozon/.cache'
    use_dynamic_padding: bool = True
    dataset: str = 'dataset_1'
    data_dir: str = '../data/'
    max_len: int = 512
    use_cache: bool = False
    special_tokens: Dict[str, List[str]] = {
        'additional_special_tokens' : ["[ATT_SEP]", "[ITEM_SEP]"]
    }
    max_num_pics: int = 32

    ## model
    head_multiplier: float = 1.0
    archead_dim: int = 128
    margin: float = 0.2
    teacher_weights: str = ''
    num_arcface_classes: int = 91615
    dropout: float = 0.0
    model: str = 'model_1'
    weights: str = ''
    model_name: str = 'microsoft/deberta-v3-small'
    pretrained: bool = True
    num_classes: int = 1
    vision_embedding: int = 128 ## is_first_sample, is_main_embedding features
    name_embedding: int = 64
    embedding_dim: int = 128
    use_residual_wrapper: bool = True
    dropout_prob: float = 0.0
    drop_path_prob: float = 0.0

    nhead: int = 8
    vision_dropout: float = 0.0
    num_encoder_layers: int = 1
    
    ## training misc
    do_eval_every: int = 1
    device: str = 'cuda:0'
    debug: bool = True
    gradient_checkpointing: bool = False
    do_train: bool = True
    seed: int = 2023
    print_graph: bool = True
    use_compile: bool = False
    use_scheduler: bool = True
    set_grad_to_none: bool = True
    use_amp: bool = True
    start_save_epoch: int = 0

    train_folds: List[int] = (1, 2, 3, 4)
    val_folds: List[int] = (0,)

    ## training
    cos_margin: float = 0.0
    subsample_proportion: float = 0.025
    epochs: int = 1
    epochs_warmup: int = 0
    opt: str = 'adam'

    lr: float = 1e-4
    wd: float = 0.0
    
    use_t0: bool = False
    T0: int = -1
    lr_min: float = 1e-7

    criterion: str = 'BCE'

    ## dataloader
    train_bs: int = 32
    val_bs: int = 32
    prefetch_factor: int = 2
    num_workers: int = 12
    
    ## logger
    exp_dir: str = '/storage/nvme1/hacks/ozon/exps'
    log_dir: str = ''
    weights_dir: str = ''
    artifacts_dir: str = ''

    def __init__(self, path):
        ## each config has a name "config_i.py"
        self.log_dir = osp.join(self.exp_dir, f"exp_{osp.basename(path).replace('.py', '').split('/')[-1].split('_')[-1]}")
        self.weights_dir = osp.join(self.log_dir, 'weights')
        self.artifacts_dir = osp.join(self.log_dir, 'artifacts')

cfg = CFG(path=__file__)
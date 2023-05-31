from configs.config_0 import CFG

cfg = CFG(path=__file__)

cfg.dataset = 'dataset_24'
cfg.model = 'model_13'
cfg.model_name = 'microsoft/deberta-v3-large'
cfg.use_dynamic_padding = False
cfg.debug = False
cfg.subsample_proportion = 1.0

cfg.special_tokens = {
    'additional_special_tokens' : ["[SEP]"]
}
cfg.epochs = 5
cfg.do_eval_every = 1
cfg.epochs_warmup = 1
cfg.lr = 5e-6
cfg.train_bs = 16
cfg.val_bs = 16
cfg.max_len = 256

## train_tok_old
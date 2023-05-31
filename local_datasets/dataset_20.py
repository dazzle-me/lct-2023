import os
import os.path as osp
from typing import Dict
import warnings
warnings.filterwarnings('ignore')
import pickle
import json

import torch
from transformers import AutoTokenizer
import pandas as pd
import polars as pl

import tqdm
from joblib import Parallel, delayed
import numpy as np

from configs.config_0 import CFG
os.environ['TOKENIZERS_PARALLELISM'] = "false"

## bi-encoder dataset...? 
## merge all information into the single prompt
## compute BCE directly on the output of the head.

class MyDataset():
    def __init__(self, cfg: CFG, tokenizer, *, mode: str):
        self.cfg = cfg
        if cfg.debug:
            cfg.subsample_proportion = 0.01
        self.tokenizer = tokenizer
        self.df = pl.read_parquet(osp.join(cfg.data_dir, 'train_df.parquet' if mode != 'test' else 'test_df.parquet')).to_pandas()
        if mode == 'train' or self.cfg.debug:
            self.df = self.df[:int(len(self.df) * cfg.subsample_proportion)].reset_index(drop=True)
        if mode in ['train', 'val']:
            self.df = self.df[self.df.fold.isin(cfg.train_folds if mode == 'train' else cfg.val_folds)].reset_index(drop=True)
        print(f"mode : {mode}, len(df) : {len(self.df)}")
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        d = self.df.iloc[index]
        name_1 = d['name1'][:512]
        name_2 = d['name2'][:512]
        cat1 = json.loads(d['categories1'])
        cat2 = json.loads(d['categories2'])

        if d['color_parsed1'] is None:
            color1 = 'unknown'
        else:
            color1 = d['color_parsed1']

        if d['color_parsed2'] is None:
            color2 = 'unknown'
        else:
            color2 = d['color_parsed2']
        
        main_1 = torch.tensor(d['main_pic_embeddings_resnet_v11'][0])
        main_2 = torch.tensor(d['main_pic_embeddings_resnet_v12'][0])
        max_num_pics = self.cfg.max_num_pics
        embedding_1 = torch.zeros(max_num_pics, self.cfg.vision_embedding)
        embedding_1[0] = main_1
        if d['pic_embeddings_resnet_v11'] is None:
            aux_1 = torch.zeros_like(main_1)
        else:
            aux_1 = [d['pic_embeddings_resnet_v11'][i] for i in range(len(d['pic_embeddings_resnet_v11']))]
            aux_1 = torch.tensor(aux_1)
        embedding_1[1:1+len(aux_1)] = aux_1

        embedding_2 = torch.zeros(max_num_pics, self.cfg.vision_embedding)
        embedding_2[0] = main_2
        if d['pic_embeddings_resnet_v12'] is None:
            aux_2 = torch.zeros_like(main_2)
        else:
            aux_2 = [d['pic_embeddings_resnet_v12'][i] for i in range(len(d['pic_embeddings_resnet_v12']))]
            aux_2 = torch.tensor(aux_2)
            # mask_aux_2[:len(aux_2)] = 1
        embedding_2[1:1+len(aux_2)] = main_2


        prompt = f"{name_1} [SEP] {name_2}, {color1} [SEP] {color2}"
        target = torch.tensor([d['target']]).float()
        ###
        text_tensor = self.tokenizer.encode_plus(
            prompt, 
            return_tensors=None, 
            add_special_tokens=True, 
            max_length=self.cfg.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        for k, v in text_tensor.items():
            text_tensor[k] = torch.tensor(v, dtype=torch.long)

        sample =  {
            'input' : text_tensor,
            'label' : target,
            'vision_embedding_1' : embedding_1,
            'vision_embedding_2' : embedding_2,
            'text' : prompt,
            'id_1' : d['variantid1'],
            'id_2' : d['variantid2'],
            'length' : text_tensor['attention_mask'].sum(),
            # 'embedding' : embeddings.astype(np.float32),
            # 'mask' : mask.reshape(-1, ).astype(bool)
        }
        return sample

    def __len__(self) -> int:
        return len(self.df)
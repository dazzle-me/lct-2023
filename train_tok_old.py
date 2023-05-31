import os
import os.path as osp
import random
import argparse
import importlib

import torch
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import tqdm

from timm.scheduler import CosineLRScheduler

from torch.utils.tensorboard import SummaryWriter
from neptune_logger import NeptuneLogger
from sklearn.metrics import average_precision_score, roc_auc_score
DEBUG = False
import warnings
warnings.filterwarnings('ignore')

from configs.config_0 import CFG
from utils import batch_to_device, describe_batch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BinaryFocalLoss():
    def __init__(self):
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = 2
    
    def __call__(self, preds, targets):
        bce_loss = self.loss(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(targets >= 0.5, (1. - probas)**self.gamma * bce_loss, probas**self.gamma * bce_loss)
        loss = loss.mean()
        return loss
from transformers import AutoTokenizer

def create_dataloaders(cfg: CFG, dataset_class):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.add_special_tokens(cfg.special_tokens)
    # tokenizer = None
    if cfg.do_train:
        train_ds = dataset_class(cfg, tokenizer, mode='train')
    val_ds = dataset_class(cfg, tokenizer, mode='val')
    test_ds = dataset_class(cfg, tokenizer, mode='test')

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    if cfg.do_train:
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.train_bs,
            num_workers=cfg.num_workers,
            shuffle=False if cfg.use_dynamic_padding else True,
            prefetch_factor=cfg.prefetch_factor,
            pin_memory=True,
            collate_fn=None,
            generator=g,
            worker_init_fn=seed_worker
        )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.val_bs,
        num_workers=cfg.num_workers,
        shuffle=False,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
        collate_fn=None,
        generator=g,
        worker_init_fn=seed_worker
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.val_bs,
        num_workers=cfg.num_workers,
        shuffle=False,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
        collate_fn=None,
        generator=g,
        worker_init_fn=seed_worker
    ) 
    return train_dl if cfg.do_train else None, val_dl, test_dl

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

import torch.nn as nn

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config')
    args = parser.parse_args()

    cfg: CFG = importlib.import_module(f'configs.{args.config}'.replace('.py', '')).cfg
    dataset_class = importlib.import_module(f"local_datasets.{cfg.dataset}".replace('.py', '')).MyDataset
    model_class = importlib.import_module(f"models.{cfg.model}".replace('.py', '')).MyNetwork
    print(cfg)
    seed_everything(cfg.seed)
    train_dl, val_dl, test_dl = create_dataloaders(cfg, dataset_class)
    
    model: nn.Module = model_class(cfg).to(cfg.device)
    if cfg.print_graph:
        print(model)
    if cfg.use_compile:
        model = torch.compile(model)
    if cfg.weights != '':
        ckpt = torch.load(cfg.weights)
        ckpt.pop('head.weight')
        ckpt.pop('head.bias')
        status = model.load_state_dict(ckpt, strict=False)
        print(f"Loaded weights from : {cfg.weights} with status : {status}")
    if cfg.opt == 'adam':
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    model.model.resize_token_embeddings(len(train_dl.dataset.tokenizer))
    num_steps_per_epoch = len(train_dl) if cfg.do_train else 0
    warmup = cfg.epochs_warmup * num_steps_per_epoch
    nsteps = cfg.epochs * num_steps_per_epoch 
    scaler = GradScaler()
    if cfg.do_train and cfg.use_scheduler:
        sched = CosineLRScheduler(
            opt, 
            warmup_t=warmup if not cfg.use_t0 else 0, 
            warmup_lr_init=0.0, 
            warmup_prefix=not cfg.use_t0,
            t_initial=(nsteps - warmup) if not cfg.use_t0 else cfg.T0,
            lr_min=cfg.lr_min,
            cycle_limit=1 if not cfg.use_t0 else 123456
        )
    else:
        sched = None

    if cfg.criterion == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg.criterion == 'Focal':
        criterion = BinaryFocalLoss()
    elif cfg.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    summary_writer = NeptuneLogger(
        debug=cfg.debug
    )
    summary_writer.run['parameters'] = vars(cfg)    
    if not osp.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    for epoch in range(1, cfg.epochs + 1):
        if cfg.do_train:
            train_pbar = tqdm.tqdm(train_dl)
            train_targets = []
            train_predictions = []
            train_losses = []
            train_ids_1 = []
            train_ids_2 = []
            for t_iter, batch in enumerate(train_pbar):
                global_iter = (epoch - 1) * num_steps_per_epoch + t_iter

                batch = batch_to_device(batch, device=cfg.device)
                batch['input'] = collate(batch['input'])

                opt.zero_grad(set_to_none=cfg.set_grad_to_none)
                with autocast(enabled=cfg.use_amp):
                    preds = model(batch)
                    loss = criterion(preds, batch['label'])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                if cfg.use_scheduler:
                    sched.step(global_iter)
                train_losses.append(loss.item())
                train_predictions.append(preds.detach().cpu().numpy())
                train_targets.append(batch['label'].detach().cpu().numpy())
                train_ids_1.extend(batch['id_1'].tolist())
                train_ids_2.extend(batch['id_2'].tolist())
                lr = opt.param_groups[0]['lr']
                train_pbar.set_description(f"loss : {loss.item():.4f}, lr : {lr:.5f}")
                summary_writer.add_scalar("lr", lr, global_iter)
            os.makedirs(cfg.artifacts_dir, exist_ok=True)
            train_targets = np.concatenate(train_targets, axis=0)
            train_predictions = np.concatenate(train_predictions, axis=0)
            df = pd.DataFrame({
                'variantid1' : train_ids_1,
                'variantid2' : train_ids_2,
                'pred' : train_predictions.reshape(-1, ),
                'target' : train_targets.reshape(-1, )
            })
            df.to_csv(osp.join(cfg.artifacts_dir, f'train_{epoch}.csv'), index=False)
            summary_writer.add_scalar('loss/train', np.mean(train_losses), epoch)
        if epoch % cfg.do_eval_every == 0:
            model.eval()
            ## val
            val_pbar = tqdm.tqdm(val_dl)
            val_losses = []
            val_targets = []
            val_predictions = []
            val_ids_1 = []
            val_ids_2 = []
            for batch in val_pbar:
                batch = batch_to_device(batch, device=cfg.device)
                batch['input'] = collate(batch['input'])
                with autocast(enabled=cfg.use_amp):
                    with torch.no_grad():
                        preds = model(batch)
                        loss = criterion(preds, batch['label'])
                        
                val_losses.append(loss.item())
                val_predictions.append(preds.detach().cpu().numpy())
                val_targets.append(batch['label'].detach().cpu().numpy())
                val_ids_1.extend(batch['id_1'].tolist())
                val_ids_2.extend(batch['id_2'].tolist())
            val_targets = np.concatenate(val_targets, axis=0)
            val_predictions = np.concatenate(val_predictions, axis=0)
            
            metric = np.mean(val_losses)
            summary_writer.add_scalar('loss/val', metric, epoch)
            
            roc_auc = roc_auc_score(val_targets, val_predictions)
            summary_writer.add_scalar('val/roc-auc', roc_auc, epoch)
            
            if not osp.isdir(cfg.weights_dir):
                os.makedirs(cfg.weights_dir)
            save_path = osp.join(cfg.weights_dir, f"model_{roc_auc:.5f}.pth")
            torch.save(model.state_dict(), save_path)
            
            os.makedirs(cfg.artifacts_dir, exist_ok=True)
            if epoch > cfg.start_save_epoch:
                np.save(osp.join(cfg.artifacts_dir, 'label.npy'), val_targets)
                np.save(osp.join(cfg.artifacts_dir, f'val_{epoch}.npy'), val_predictions)
                df = pd.DataFrame({
                    'variantid1' : val_ids_1,
                    'variantid2' : val_ids_2,
                    'pred' : val_predictions.reshape(-1, ),
                    'target' : val_targets.reshape(-1, )
                })
                df.to_csv(osp.join(cfg.artifacts_dir, f'val_{epoch}.csv'), index=False)
            print(f"Saved model to : {save_path}")
            ### test
            if epoch > cfg.start_save_epoch:
                test_pbar = tqdm.tqdm(test_dl)
                test_predictions = []
                test_ids_1 = []
                test_ids_2 = []
                for batch in test_pbar:
                    batch = batch_to_device(batch, device=cfg.device)
                    batch['input'] = collate(batch['input'])
                    with autocast(enabled=cfg.use_amp):
                        with torch.no_grad():
                            preds = model(batch)
                    test_predictions.append(preds.detach().cpu().numpy())
                    test_ids_1.extend(batch['id_1'].tolist())
                    test_ids_2.extend(batch['id_2'].tolist())
                test_predictions = np.concatenate(test_predictions, axis=0)
                df = pd.DataFrame({
                    'variantid1' : test_ids_1,
                    'variantid2' : test_ids_2,
                    'target' : test_predictions.reshape(-1, ),
                })
                df.to_csv(osp.join(cfg.artifacts_dir, f'test_{epoch}.csv'), index=False)
            model.train()

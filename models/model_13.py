import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel

from configs.config_0 import CFG

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
import numpy as np
def positional_encoding(length, embed_dim):
	dim = embed_dim//2
	position = np.arange(length)[:, np.newaxis]     # (seq, 1)
	dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)
	angle = 1 / (10000**dim)         # (1, dim)
	angle = position * angle    # (pos, dim)
	pos_embed = np.concatenate(
		[np.sin(angle), np.cos(angle)],
		axis=-1
	)
	pos_embed = torch.from_numpy(pos_embed).float()
	return pos_embed

class MyNetwork(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model_name, output_hidden_states=True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        if cfg.pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_name, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        pos_embed = positional_encoding(256, self.config.hidden_size)
        self.register_buffer('pos_embed', torch.Tensor(pos_embed))
        # self.pos_embed = nn.Parameter(torch.Tensor(pos_embed))
        self.vision_proj = nn.Sequential(
            nn.Linear(cfg.vision_embedding, self.config.hidden_size * 2),
            nn.LayerNorm(self.config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_size* 2, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size, 
            nhead=cfg.nhead, 
            # dim_feedforward=cfg.embedding_dim * 4, 
            dropout=cfg.vision_dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.num_encoder_layers,
        )
        # self.sep_token_1 = nn.Parameter()
        self.register_buffer('sep_token', torch.zeros((1, 1, self.config.hidden_size)))
        # self.sep_token_2 = nn.Parameter(torch.zeros((1, 1, self.config.hidden_size)))
        self.pool = MeanPooling()
        self.fc = nn.Linear(2 * self.config.hidden_size, self.cfg.num_classes)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, batch):
        text_feature = self.feature(batch['input'])
        vis1 = self.vision_proj(batch['vision_embedding_1'])
        vis2 = self.vision_proj(batch['vision_embedding_2'])
        
        bs = len(vis1)
        sep_token_2 = self.sep_token.repeat(bs, 1, 1)

        feature = torch.cat([vis1, sep_token_2, vis2], dim=1)
        L = feature.shape[1]
        feature = feature + self.pos_embed[:L]
        feature = self.transformer_encoder(feature)
        feature = torch.mean(feature, dim=1)
        output = self.fc(torch.cat([feature, text_feature],dim=1))
        return output
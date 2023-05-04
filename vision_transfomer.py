import torch
import timm
from utils import *


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params

def get_vit_model(model_type, img_size):
    print('Loading pretrained weights...')
    model = timm.create_model(
        model_type,
        pretrained=True,
        drop_rate = 0.1,
        img_size = img_size,
        num_classes = 0
    )
    model.to(device)
    print(f'Loaded {model_type} with {count_parameters(model)} parameters.')
    return model

def get_patches(self, x):
    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.pos_embed
    x = self.pos_drop(x)
    return x[:, 1:, :]

def prepare_tokens(self, x, add_cls=True):
    if add_cls:
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.norm_pre(x)
    x = self.blocks(x)
    x = self.norm(x)
    return x


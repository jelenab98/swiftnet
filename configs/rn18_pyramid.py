from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from pathlib import Path

import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os

from models.loss import BoundaryAwareFocalLoss
from models.resnet.resnet_pyramid import *
from data.cityscapes import Cityscapes
from models.semseg import SemsegModel
from models.util import get_n_params
from data.transform import *
# from evaluation import StorePreds

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
root = Path('/mnt/data/City')

evaluating = False
pruning = True
is_34 = False
prune_mode = "rewind"
pyramid = True

random_crop_size = 768

scale = 1
mean = [73.15, 82.90, 72.3]
std = [47.67, 48.49, 47.73]
mean_rgb = tuple(np.uint8(scale * np.array(mean)))

num_classes = Cityscapes.num_classes
ignore_id = Cityscapes.num_classes
class_info = Cityscapes.class_info
color_info = Cityscapes.color_info

num_levels = 3
ostride = 4
target_size_crops = (random_crop_size, random_crop_size)
target_size_crops_feats = (random_crop_size // ostride, random_crop_size // ostride)

eval_each = 2
dist_trans_bins = (16, 64, 128)
dist_trans_alphas = (8., 4., 2., 1.)
target_size = (2048, 1024)
target_size_feats = (2048 // ostride, 1024 // ostride)

lr = 4e-4
lr_min = 1e-6
fine_tune_factor = 4
weight_decay = 1e-4
epochs = 75
pruning_percentages = [0.05, 0.05,
                       0.05, 0.05,
                       0.05, 0.05,
                       0.05, 0.05]

trans_val = Compose(
    [Open(),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Tensor(),
     ]
)

if evaluating:
    trans_train = trans_train_val = trans_val
else:
    trans_train = Compose(
        [Open(),
         RandomFlip(),
         RandomSquareCropAndScale(random_crop_size, ignore_id=ignore_id, mean=mean_rgb),
         SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
         LabelDistanceTransform(num_classes=num_classes, reduce=True, bins=dist_trans_bins, alphas=dist_trans_alphas),
         Tensor(),
         ])

dataset_train = Cityscapes(root, transforms=trans_train, subset='train')
dataset_val = Cityscapes(root, transforms=trans_val, subset='val')

backbone = resnet18(pretrained=True,
                    pyramid_levels=num_levels,
                    k_upsample=3,
                    scale=scale,
                    mean=mean,
                    std=std,
                    k_bneck=1,
                    output_stride=ostride,
                    efficient=False)
model = SemsegModel(backbone, num_classes, k=1, bias=True)
if pruning:
    model.load_state_dict(torch.load('weights/rn18_pyramid/model_best.pt'), strict=False)

if evaluating:
    model.load_state_dict(torch.load('weights/rn18_pyramid/model_best.pt'), strict=False)
else:
    model.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=num_classes, ignore_id=ignore_id)

bn_count = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        bn_count += 1
print(f'Num BN layers: {bn_count}')

if not evaluating:
    optim_params = [
        {'params': model.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.fine_tune_params(), 'lr': lr / fine_tune_factor,
         'weight_decay': weight_decay / fine_tune_factor},
    ]

    optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)

batch_size = bs = 10
print(f'Batch size: {bs}')
nw = 2

loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate)
if evaluating:
    loader_train = DataLoader(dataset_train, batch_size=1, collate_fn=custom_collate, num_workers=nw)
else:
    loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=nw, pin_memory=True,
                              drop_last=True, collate_fn=custom_collate, shuffle=True)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')

if evaluating:
    eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
    store_dir = f'{dir_path}/out/'
    for d in ['', 'val', 'train']:
        os.makedirs(store_dir + d, exist_ok=True)
    to_color = ColorizeLabels(color_info)
    to_image = Compose([Numpy(), to_color])
    #eval_observers = [StorePreds(store_dir, to_image, to_color)]


def reset_optimizer(model, lr=4e-4):
    optim_params = [
        {'params': model.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.fine_tune_params(), 'lr': lr / fine_tune_factor,
         'weight_decay': weight_decay / fine_tune_factor},
    ]
    optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)
    return optimizer, lr_scheduler


def get_initial_model():
    backbone = resnet18(pretrained=True,
                        pyramid_levels=num_levels,
                        k_upsample=3,
                        scale=scale,
                        mean=mean,
                        std=std,
                        k_bneck=1,
                        output_stride=ostride,
                        efficient=False)
    model_tmp = SemsegModel(backbone, num_classes, k=1, bias=True)
    model_tmp.load_state_dict(torch.load('weights/rn18_pyramid/model_best.pt'), strict=False)
    model_tmp.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=num_classes, ignore_id=ignore_id)
    return model_tmp
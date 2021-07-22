from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from pathlib import Path
import torch.optim as optim

import torch.nn as nn
import numpy as np
import torch
import os

from models.loss import BoundaryAwareFocalLoss
from models.resnet.resnet_relu import *
from data.cityscapes import Cityscapes
from models.semseg import SemsegModel
from models.util import get_n_params
from evaluation import StorePreds
from data.transform import *


root = Path('/mnt/data/City')
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

evaluating = False
pruning = True
is_34 = True
pyramid = False
prune_mode = "rewind"

pruning_percentages = [0.02, 0.02, 0.02,
                       0.02, 0.02, 0.02, 0.02,
                       0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                       0.05, 0.05, 0.05]
lr = 8e-4 / 4
lr_min = 1e-6
fine_tune_factor = 4
weight_decay = 1e-4 / fine_tune_factor
epochs = 75

random_crop_size = 768

scale = 1
mean = [73.15, 82.90, 72.3]
std = [47.67, 48.49, 47.73]
mean_rgb = tuple(np.uint8(scale * np.array(mean)))

class_info = Cityscapes.class_info
color_info = Cityscapes.color_info
num_classes = ignore_id = Cityscapes.num_classes

downsample = 1
alpha = 2.0
num_levels = 1
alphas = [alpha ** i for i in range(num_levels)]
target_size_crops = (random_crop_size, random_crop_size)
target_size_crops_feats = (random_crop_size // 4, random_crop_size // 4)
target_size = (2048, 1024)
target_size_feats = (2048 // 4, 1024 // 4)

eval_each = 2
dist_trans_bins = (16, 64, 128)
dist_trans_alphas = (8., 4., 2., 1.)

trans_val = Compose(
    [Open(),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Tensor(),
     ]
)

if evaluating:
    trans_train = trans_val
else:
    trans_train = Compose(
        [Open(copy_labels=False),
         RandomFlip(),
         RandomSquareCropAndScale(random_crop_size, ignore_id=ignore_id, mean=mean_rgb),
         SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
         LabelDistanceTransform(num_classes=num_classes, reduce=True, bins=dist_trans_bins,
                                alphas=dist_trans_alphas, ignore_id=ignore_id),
         Tensor(),
         ])

dataset_train = Cityscapes(root, transforms=trans_train, subset='train', labels_dir='labels')
dataset_val = Cityscapes(root, transforms=trans_val, subset='val', labels_dir='labels')

for dset in [dataset_train, dataset_val]:
    for atter in ['class_info', 'color_info']:
        setattr(dset, atter, getattr(Cityscapes, atter))

resnet = resnet34(pretrained=True, k_up=3, scale=scale, mean=mean, std=std, output_stride=8, efficient=False)
model = SemsegModel(resnet, num_classes, k=1, bias=True)

if pruning:
    model.load_state_dict(torch.load("weights/76-66_resnet34x8/stored/model_best.pt"), strict=False)

if evaluating:
    model.load_state_dict(torch.load("weights/76-66_resnet34x8/stored/model_best.pt"), strict=False)
else:
    model.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=num_classes, ignore_id=ignore_id)


bn_count = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        bn_count += 1
print(f'Num BN layers: {bn_count}')
print(f'Upsample modules:\n{model.backbone.upsample}')

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


collate = custom_collate
loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate)

if evaluating:
    loader_train = DataLoader(dataset_train, batch_size=1, collate_fn=collate, num_workers=nw)
else:
    loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=nw, pin_memory=True,
                              drop_last=True, collate_fn=collate)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')

if evaluating:
    eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
    store_dir = f'{dir_path}/out/'
    for d in ['', 'val', 'train']:
        os.makedirs(store_dir + d, exist_ok=True)
    to_color = ColorizeLabels(Cityscapes.color_info)
    to_image = Compose([Numpy(), to_color])
    eval_observers = [StorePreds(store_dir, to_image, to_color)]


def reset_optimizer(model):
    optim_params = [
        {'params': model.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.fine_tune_params(), 'lr': lr / fine_tune_factor,
         'weight_decay': weight_decay / fine_tune_factor},
    ]

    optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)
    return optimizer, lr_scheduler


def get_initial_model():
    tmp_resnet = resnet34(pretrained=True, k_up=3, scale=scale, mean=mean, std=std, output_stride=8, efficient=False)
    tmp_model = SemsegModel(tmp_resnet, num_classes, k=1, bias=True)
    tmp_model.load_state_dict(torch.load('weights/76-66_resnet34x8/stored/model_best.pt'), strict=False)
    model.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=num_classes, ignore_id=ignore_id)
    return tmp_model

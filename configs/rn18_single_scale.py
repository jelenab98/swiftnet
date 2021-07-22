from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from pathlib import Path

import torch.optim as optim
import numpy as np
import torch
import os

from models.resnet.resnet_single_scale import *
from models.loss import SemsegCrossEntropy
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
prune_mode = "rewind"
is_34 = False
pyramid = False

random_crop_size = 768

scale = 1
mean = [73.15, 82.90, 72.3]
std = [47.67, 48.49, 47.73]
mean_rgb = tuple(np.uint8(scale * np.array(mean)))

num_classes = Cityscapes.num_classes
ignore_id = Cityscapes.num_classes
class_info = Cityscapes.class_info
color_info = Cityscapes.color_info

target_size_crops = (random_crop_size, random_crop_size)
target_size_crops_feats = (random_crop_size // 4, random_crop_size // 4)
target_size = (2048, 1024)
target_size_feats = (2048 // 4, 1024 // 4)

eval_each = 2

lr = 4e-4
lr_min = 1e-6
fine_tune_factor = 4
weight_decay = 1e-4
epochs = 50
pruning_percentages = [0.05, 0.05,
                       0.05, 0.05,
                       0, 0,
                       0, 0]

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
        [Open(),
         RandomFlip(),
         RandomSquareCropAndScale(random_crop_size, ignore_id=num_classes, mean=mean_rgb),
         SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
         Tensor(),
         ]
    )

dataset_train = Cityscapes(root, transforms=trans_train, subset='train')
dataset_val = Cityscapes(root, transforms=trans_val, subset='val')

resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
model = SemsegModel(resnet, num_classes)

if pruning:
    model.load_state_dict(torch.load('weights/rn18_single_scale/model_best.pt'))

if evaluating:
    model.load_state_dict(torch.load('weights/rn18_single_scale/model_best.pt'))
else:
    model.criterion = SemsegCrossEntropy(num_classes=num_classes, ignore_id=ignore_id)

    optim_params = [
        {'params': model.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.fine_tune_params(), 'lr': lr / fine_tune_factor,
         'weight_decay': weight_decay / fine_tune_factor},
    ]

    optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)

batch_size = 10
print(f'Batch size: {batch_size}')

if evaluating:
    loader_train = DataLoader(dataset_train, batch_size=1, collate_fn=custom_collate)
else:
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True,
                              drop_last=True, collate_fn=custom_collate)
loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
spp_params = get_n_params(model.backbone.spp.parameters())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')
print(f'SPP params: {spp_params:,}')

if evaluating:
    eval_loaders = [(loader_val, 'val')]  # , (loader_train, 'train')]
    store_dir = f'{dir_path}/out/'
    for d in ['', 'val', 'train', 'training']:
        os.makedirs(store_dir + d, exist_ok=True)
    to_color = ColorizeLabels(color_info)
    to_image = Compose([DenormalizeTh(scale, mean, std), Numpy(), to_color])
    eval_observers = [StorePreds(store_dir, to_image, to_color)]


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
    tmp_resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
    tmp_model = SemsegModel(tmp_resnet, num_classes)
    tmp_model.load_state_dict(torch.load('weights/rn18_single_scale/model_best.pt'))
    tmp_model.criterion = SemsegCrossEntropy(num_classes=num_classes, ignore_id=ignore_id)
    return tmp_model

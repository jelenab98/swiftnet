from models.resnet.resnet_single_scale import BasicBlock as BasicBlock_SN
from models.resnet.resnet_pyramid import BasicBlock as BasicBlock_Pyr
from models.resnet.resnet_relu import BasicBlock as BasicBlock34
from torch_pruning.dependency import DependencyGraph
from evaluation import evaluate_semseg
from models.util import get_n_params
from time import perf_counter
from thop import profile
from pathlib import Path
from shutil import copy

import torch_pruning as tp
import importlib.util
import argparse
import datetime
import pickle
import torch
import time
import sys
import os


def prune_model(model, percentages, BasicBlock, p=1, H=1024, W=2048):
    model.cpu()
    prune_stats = []
    indices = []

    DG = DependencyGraph().build_dependency(model, torch.randn(1, 3, H, W))

    def prune_conv(conv, amount=0.2, p=1):
        strategy = tp.prune.strategy.LNStrategy(p)
        pruning_index = strategy.apply(conv.weight, amount)

        n_to_prune = max(int(amount * len(conv.weight)), 1)

        if isinstance(pruning_index, int):
            pruning_index = [pruning_index]

        stats = f"Layer {conv}, number to prune: {n_to_prune}, indices to prune: {pruning_index}"
        prune_stats.append(stats)
        print(stats)
        indices.append(pruning_index)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    i = 0
    for m in model.modules():
        if isinstance(m, BasicBlock):
            print(m.conv1, m.conv2)
            percentage_1 = percentages[i]
            i += 1
            if percentage_1 == 0:
                continue
            prune_conv(m.conv1, percentage_1, p)
            prune_conv(m.conv2, percentage_1, p)

    return model, prune_stats, indices


def prune_lottery_ticket(model, initial_model, pruning_indices, percentages, BasicBlock, p=1, H=1024, W=2048):
    initial_model.cpu()
    model.cpu()
    prune_stats = []

    def prune_conv(conv, idx, amount=0.2, p=1):
        strategy = tp.prune.strategy.LNStrategy(p)
        pruning_index = strategy.apply(conv.weight, amount)

        n_to_prune = max(int(amount * len(conv.weight)), 1)

        if isinstance(pruning_index, int):
            pruning_index = [pruning_index]

        stats = f"Layer {conv}, number to prune: {n_to_prune}, indices to prune: {pruning_index}"
        print(stats)
        prune_stats.append(stats)
        pruning_indices[idx] = (pruning_indices[idx][0],
                                merge_pruning_indices(pruning_indices[idx][0],
                                                      pruning_indices[idx][1], pruning_index))
        #plan = DG.get_pruning_plan(conv_initial, tp.prune_conv, pruning_indices[idx][1])
        #plan.exec()

    i = 0
    j = 0
    for m in model.modules():
        if isinstance(m, BasicBlock):
            percentage = percentages[j]
            i += 2
            j += 1
            if percentage == 0:
                continue
            prune_conv(m.conv1, i - 2, percentage, p)
            prune_conv(m.conv2, i - 1, percentage, p)

    del model

    DG = tp.DependencyGraph().build_dependency(initial_model, torch.randn(1, 3, H, W))
    idx = 0
    for m in initial_model.modules():
        if isinstance(m, BasicBlock):
            plan = DG.get_pruning_plan(m.conv1, tp.prune_conv, pruning_indices[idx][1])
            plan.exec()
            plan = DG.get_pruning_plan(m.conv2, tp.prune_conv, pruning_indices[idx + 1][1])
            plan.exec()
            idx += 2
    return initial_model, prune_stats


def merge_pruning_indices(initial_length, pruning_indices1, pruning_indices2):
    initial_indices = list(range(initial_length))

    for idx in pruning_indices1:
        initial_indices.remove(idx)

    for idx in pruning_indices2:
        pruning_indices1.append(initial_indices[idx])

    final_pruning_indices = sorted(pruning_indices1)

    return final_pruning_indices


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def store(model, store_path, name):
    with open(store_path.format(name), 'wb') as f:
        torch.save(model, f)


def load(store_path, name):
    with open(store_path.format(name), 'rb') as f:
        model = torch.load(f)
    return model


class Logger(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


class PruningTrainer:
    def __init__(self, conf, args, name, prune_stats, pruning_indices):
        self.conf = conf
        self.prune_stats = prune_stats
        self.pruning_indices = pruning_indices
        using_hparams = hasattr(conf, 'hyperparams')
        print(f'Using hparams: {using_hparams}')
        self.hyperparams = self.conf
        self.args = args
        self.name = name
        self.model = self.conf.model
        self.optimizer = self.conf.optimizer

        self.dataset_train = self.conf.dataset_train
        self.dataset_val = self.conf.dataset_val
        self.loader_train = self.conf.loader_train
        self.loader_val = self.conf.loader_val

    def __enter__(self):
        self.best_iou = -1
        self.best_iou_epoch = -1
        self.model_best = None
        self.validation_ious = []
        self.experiment_start = datetime.datetime.now()

        if self.args.resume:
            self.experiment_dir = Path(self.args.resume)
            print(f'Resuming experiment from {args.resume}')
        else:
            self.experiment_dir = Path(self.args.store_dir) / (
                    self.experiment_start.strftime('%Y_%m_%d_%H_%M_%S_') + self.name)

        self.checkpoint_dir = self.experiment_dir / 'stored'
        self.store_path = str(self.checkpoint_dir / '{}.pth')

        if not self.args.dry and not self.args.resume:
            os.makedirs(str(self.experiment_dir), exist_ok=True)
            os.makedirs(str(self.checkpoint_dir), exist_ok=True)
            copy(self.args.config, str(self.experiment_dir / 'config.py'))

        if self.args.log and not self.args.dry:
            f = (self.experiment_dir / 'log.txt').open(mode='a')
            sys.stdout = Logger(sys.stdout, f)

        self.model.cuda()

        with open(self.experiment_dir / "prune_statistics.txt", mode="w") as f2:
            for line in self.prune_stats:
                f2.write(f"{line}\n")

        with open(self.experiment_dir / "prune_indices.txt", mode="w") as f3:
            for module_idx, (length, indices) in enumerate(self.pruning_indices):
                f3.write(f"{module_idx} {length} {indices}\n")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.args.dry:
            store(self.model, self.store_path, 'model')
        if not self.args.dry:
            with open(f'{self.experiment_dir}/val_ious.pkl', 'wb') as f:
                pickle.dump(self.validation_ious, f)
            # dir_iou = Path(self.args.store_dir) / (f'{self.best_iou:.2f}_'.replace('.', '-') + self.name)
            # os.rename(self.experiment_dir, dir_iou)

    def train(self):
        num_epochs = self.hyperparams.epochs
        start_epoch = self.hyperparams.start_epoch if hasattr(self.hyperparams, 'start_epoch') else 0
        for epoch in range(start_epoch, num_epochs):
            if hasattr(self.conf, 'epoch'):
                self.conf.epoch.value = epoch
                print(self.conf.epoch)
            self.model.train()
            try:
                self.conf.lr_scheduler.step()
                print(f'Elapsed time: {datetime.datetime.now() - self.experiment_start}')
                for group in self.optimizer.param_groups:
                    print('LR: {:.4e}'.format(group['lr']))
                eval_epoch = ((epoch % self.conf.eval_each == 0) or (epoch == num_epochs - 1))  # and (epoch > 0)
                self.model.criterion.step_counter = 0
                print(f'Epoch: {epoch} / {num_epochs - 1}')
                if eval_epoch and not self.args.dry:
                    print("Experiment dir: %s" % self.experiment_dir)
                batch_iterator = iter(enumerate(self.loader_train))
                start_t = perf_counter()
                for step, batch in batch_iterator:
                    self.optimizer.zero_grad()
                    loss = self.model.loss(batch)
                    loss.backward()
                    self.optimizer.step()
                    if step % 80 == 0 and step > 0:
                        curr_t = perf_counter()
                        print(f'{(step * self.conf.batch_size) / (curr_t - start_t):.2f}fps')
                if not self.args.dry:
                    store(self.model, self.store_path, 'model')
                    store(self.optimizer, self.store_path, 'optimizer')
                if eval_epoch and self.args.eval:
                    print('Evaluating model')
                    iou, per_class_iou = evaluate_semseg(self.model, self.loader_val, self.dataset_val.class_info)
                    self.validation_ious += [iou]
                    if self.args.eval_train:
                        print('Evaluating train')
                        evaluate_semseg(self.model, self.loader_train, self.dataset_train.class_info)
                    if iou > self.best_iou:
                        self.best_iou = iou
                        self.best_iou_epoch = epoch
                        if not self.args.dry:
                            copy(self.store_path.format('model'), self.store_path.format('model_best'))
                    print(f'Best mIoU: {self.best_iou:.2f}% (epoch {self.best_iou_epoch})')

            except KeyboardInterrupt:
                break
        try:
            self.model_best = load(self.store_path, "model_best")
        except FileNotFoundError:
            print("Best model was not saved!")


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('--store_dir', default='saves/', type=str, help='Path to experiments directory')
parser.add_argument('--resume', default=None, type=str, help='Path to existing experiment dir')
parser.add_argument('--no-log', dest='log', action='store_false', help='Turn off logging')
parser.add_argument('--log', dest='log', action='store_true', help='Turn on train evaluation')
parser.add_argument('--no-eval-train', dest='eval_train', action='store_false', help='Turn off train evaluation')
parser.add_argument('--eval-train', dest='eval_train', action='store_true', help='Turn on train evaluation')
parser.add_argument('--no-eval', dest='eval', action='store_false', help='Turn off evaluation')
parser.add_argument('--eval', dest='eval', action='store_true', help='Turn on evaluation')
parser.add_argument('--dry-run', dest='dry', action='store_true', help='Don\'t store')
parser.set_defaults(log=True)
parser.set_defaults(eval_train=False)
parser.set_defaults(eval=True)


def store_data(path: Path, params, miou, gmacs, fps):
    with path.open(mode="a") as d_stream:
        d_stream.write(f"{params},{miou},{gmacs},{fps}\n")


def measure_fps(model, n=1000, test_dims=(1024, 2048)):
    h, w = test_dims
    model.eval()
    model.cuda()
    with torch.no_grad():
        data = torch.randn(1, 3, h, w).cuda()
        logits = model.forward(data)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            # data = torch.randn(1, 3, h, w).to(device)
            logits = model.forward(data)
            #_, pred = logits.max(1)
            #out = pred.data.byte().cpu()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        fps = n / (t1 - t0)
    return fps


def reset_pruning_indices(model, BasicBlock):
    indices = []
    for m in model.modules():
        if isinstance(m, BasicBlock):
            indices.append((len(m.conv1.weight), []))
            indices.append((len(m.conv2.weight), []))
    return indices


def measure_gmacs_fps(H=1024, W=2048, C=3):
    input = torch.randn(1, C, H, W)
    conf.model.to("cpu")
    macs, params = profile(conf.model, inputs=(input,))
    macs = macs / 1e9
    print(f"GMACs: {macs}")
    conf.model.cuda()
    fps = measure_fps(conf.model)
    print(f"FPS: {fps}")
    return params, macs, fps


def prune_with_rewinding():
    return prune_model(conf.model.cuda(), pruning_percentages, BasicBlock)


def prune_with_resetting():
    initial_model = conf.get_initial_model()
    initial_model.to("cpu")
    return prune_lottery_ticket(conf.model, initial_model, pruning_indices, pruning_percentages, BasicBlock)


if __name__ == '__main__':
    args = parser.parse_args()
    conf_path = Path(args.config)
    conf = import_module(args.config)

    if conf.pyramid is True:
        BasicBlock = BasicBlock_Pyr
    else:
        if conf.is_34 is True:
            BasicBlock = BasicBlock34
        else:
            BasicBlock = BasicBlock_SN

    store_dir = Path(args.store_dir)
    store_stats_path = store_dir / "params_mious.txt"
    os.makedirs(str(store_dir), exist_ok=True)

    store_data(store_stats_path, "Params", "mIoU", "GMACs", "FPS")

    pruning_percentages = conf.pruning_percentages
    print("PRUNING PERCENTAGES: ", pruning_percentages)
    pruning_indices = reset_pruning_indices(conf.model, BasicBlock)
    print("INITIAL TESTING:")
    print("MODEL:\n", conf.model)
    iou, per_class_iou = evaluate_semseg(conf.model.cuda(), conf.loader_val, conf.dataset_val.class_info)
    params, macs, fps = measure_gmacs_fps()
    store_data(store_stats_path, params, iou, macs, fps)

    for i in range(10):
        if i != 0:
            iou, per_class_iou = evaluate_semseg(conf.model.cuda(), conf.loader_val, conf.dataset_val.class_info)
        conf.model.to("cpu")

        if conf.prune_mode == "rewind":
            model, prune_stats, _ = prune_with_rewinding()
        elif conf.prune_mode == "reset":
            model, prune_stats = prune_with_resetting()
        else:
            raise ValueError("Prune mode should be rewind or reset!")

        conf.model = model
        print("PRUNED MODEL:")
        print(conf.model)

        total_params = get_n_params(model.parameters())
        ft_params = get_n_params(model.fine_tune_params())
        ran_params = get_n_params(model.random_init_params())
        print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')

        params, macs, fps = measure_gmacs_fps()

        optimizer, scheduler = conf.reset_optimizer(model)
        conf.optimizer = optimizer
        conf.lr_scheduler = scheduler

        with PruningTrainer(conf, args, conf_path.stem, prune_stats, pruning_indices) as trainer:
            trainer.train()
            if trainer.model_best is not None:
                conf.model = trainer.model_best
            else:
                conf.model = trainer.model
            store_data(store_stats_path, total_params, trainer.best_iou, macs, fps)

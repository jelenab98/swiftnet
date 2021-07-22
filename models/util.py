import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from torch.utils.checkpoint import checkpoint
from collections import namedtuple
import typing as T
import itertools
import math


ForwardUtils = namedtuple('ForwardUtils', 'prepare_data do_forward loss')

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)

batchnorm_momentum = 0.01 / 2


def _extract_tensors(*args, **kwargs):
    for a in itertools.chain(args, kwargs.values()):
        if isinstance(a, torch.Tensor):
            yield a
        elif isinstance(a, T.Sequence):
            for x in _extract_tensors(*a):
                yield x
        elif isinstance(a, T.Mapping):
            for x in _extract_tensors(*a.values()):
                yield x


def check_inf_nan_forward_hook(module_name, module, input, output):
    for x in _extract_tensors(output):
        if not isinstance(x, torch.Tensor):
            continue
        if (inf := torch.isinf(x).any()) or (nan := torch.isnan(x).any()):
            print(module_name, module)
            breakpoint()
            assert not inf
            assert not nan


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def find_largest_norms(weight):
    return torch.from_numpy(np.argsort([x.norm().item() for x in weight]))


def get_ms_image_sizes(image_wh, k_list=np.linspace(1 / 2., 2., 9), longer_side_anchor=2048, topk=3, f=8):
    w, h = image_wh
    # t = longer_side_anchor / max(image_wh)
    # t = min(max(t, np.min(k_list)), np.max(k_list))
    longer_side = max(image_wh)
    candidates = longer_side_anchor * k_list
    image_sizes = []
    if (longer_side < candidates).any():
        image_sizes += [image_wh]
        topk -= 1
    scales = (candidates)[np.abs(candidates - longer_side).argsort()][:topk] / longer_side
    return image_sizes + [(int(math.ceil(w * s / float(f)) * f), int(math.ceil(h * s / float(f)) * f)) for s in scales]


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, groups=-1):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0, groups=1, separable=False, bn_class=nn.BatchNorm2d):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', bn_class(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d
        warnings.warn(f'Using conv type {k}x{k}: {conv_class}')
        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation, groups=groups))
        if drop_rate > 0:
            warnings.warn(f'Using dropout with p: {drop_rate}')
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))


class _ConvBNReLu(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_ConvBNReLu, self).__init__()
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                          dilation=dilation))
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_out, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))


class _ConvBNReLuConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_ConvBNReLuConv, self).__init__()
        padding = k // 2
        self.add_module('conv1', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation))
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_out, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_maps_out, num_maps_out, kernel_size=1, padding=0, bias=bias,
                                           dilation=dilation))


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True,
                 bn_class=nn.BatchNorm2d):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn and bneck_starts_with_bn,
                                      bn_class=bn_class)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, separable=separable,
                                      bn_class=bn_class)
        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        warnings.warn(f'\tUsing skips: {self.use_skip} (only skips: {self.only_skip})', UserWarning)
        self.fixed_size = fixed_size

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[2:4]
        if self.fixed_size is not None:
            x = F.interpolate(x, size=self.fixed_size, mode="nearest")
        else:
            x = F.interpolate(x, size=skip_size, mode="bilinear", align_corners=False)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x


class _UpsampleSE(_Upsample):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True):
        super(_UpsampleSE, self).__init__(num_maps_in, skip_maps_in, num_maps_out,
                                          use_bn=use_bn,
                                          k=k,
                                          use_skip=use_skip,
                                          only_skip=only_skip,
                                          detach_skip=detach_skip,
                                          fixed_size=fixed_size,
                                          separable=separable,
                                          bneck_starts_with_bn=bneck_starts_with_bn)
        self.bottleneck = nn.Sequential(
            SqueezeExcitation(skip_maps_in, num_maps_out),
            _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        )


class _UpsampleCBRC(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out):
        super(_UpsampleCBRC, self).__init__()
        print(f'Upsample CBCR layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _ConvBNReLuConv(skip_maps_in, num_maps_in, k=1)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=3)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x


class _UpsampleBlend(nn.Module):
    def __init__(self, num_features, use_bn=True, use_skip=True, detach_skip=False, fixed_size=None, k=3,
                 separable=False):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNReluConv(num_features, num_features, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.detach_skip = detach_skip
        warnings.warn(f'Using skip connections: {self.use_skip}', UserWarning)
        self.fixed_size = fixed_size

    def forward(self, x, skip):
        if self.detach_skip:
            warnings.warn(f'Detaching skip connection {skip.shape[2:4]}', UserWarning)
            skip = skip.detach()
        skip_size = skip.size()[-2:]
        if self.fixed_size is not None:
            x = F.interpolate(x, size=self.fixed_size, mode="nearest")
        else:
            x = F.interpolate(x, size=skip_size, mode="bilinear", align_corners=False)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x


class _UpsampleBlendBottlencek(nn.Module):
    def __init__(self, skip_feats, out_feats, use_bn=True, k=3):
        super(_UpsampleBlendBottlencek, self).__init__()
        self.stem = _BNReluConv(skip_feats, out_feats, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(out_feats, out_feats, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + self.stem(skip)
        x = self.blend_conv.forward(x)
        return x


class _UpsampleShared(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out):
        super(_UpsampleShared, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottlenecks = nn.ModuleList([_BNReluConv(skip_in, num_maps_in, k=1) for skip_in in skip_maps_in])
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=3)

    def forward(self, x, skip, i):
        skip = self.bottlenecks[i].forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x


class _UpsampleNoSKip(nn.Module):
    def __init__(self, num_maps_in, num_maps_out):
        super(_UpsampleNoSKip, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, out = {num_maps_out}')
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=3)

    def forward(self, x, target_size):
        x = upsample(x, target_size)
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True, bn_class=nn.BatchNorm2d):
        super(SpatialPyramidPooling, self).__init__()
        self.fixed_size = fixed_size
        self.grids = grids
        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum,
                                                  batch_norm=use_bn and starts_with_bn, bn_class=bn_class))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                            drop_rate=drop_rate, bn_class=bn_class))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                        bn_class=bn_class))

    def forward(self, x):
        levels = []
        target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)
            if self.fixed_size is not None:
                level = F.interpolate(level, size=self.fixed_size, mode="nearest")
            else:
                level = F.interpolate(level, size=target_size, mode="bilinear", align_corners=False)
            levels.append(level)

        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class SpatialPyramidPoolingCBRC(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=[6, 3, 2, 1], square_grid=False, bn_momentum=0.1):
        super(SpatialPyramidPoolingCBRC, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _ConvBNReLuConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i), _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum))
        self.spp.add_module('spp_fuse', _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = upsample(level, target_size)
            levels.append(level)

        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class ReceptiveFieldFunction(torch.autograd.Function):
    '''
    https://papers.nips.cc/paper/6203-understanding-the-effective-receptive-field-in-deep-convolutional-neural-networks.pdf
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_output.view(-1)[grad_output.argmax()] = 1.
        grad_output[grad_output < 1.] = 0.
        return grad_output


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishImplementation.apply(x)
        # return x * torch.sigmoid(x)


class ConvBNSwish(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, bn_class=nn.BatchNorm2d, efficient=False):
        padding = self._get_padding(kernel_size, stride)
        self.efficient = efficient
        super(ConvBNSwish, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            bn_class(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]

    def _pad_conv(self, x):
        return self[1](self[0](x))

    def _bn_swish(self, x):
        return self[3](self[2](x))

    def forward(self, x):
        if not (self.training and self.efficient):
            return super(ConvBNSwish, self).forward(x)
        warnings.warn('Checkpointing ConvBNSwish')
        x = self._pad_conv(x)
        return checkpoint(self._bn_swish, x)


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim, efficient=False):
        super(SqueezeExcitation, self).__init__()
        self.efficient = efficient
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def se_forward(self, x):
        return x * self.se(x)

    def forward(self, x):
        if self.efficient and self.training:
            warnings.warn('Checkpointing squeeze excite operation.')
            return checkpoint(self.se_forward, x)
        return self.se_forward(x)

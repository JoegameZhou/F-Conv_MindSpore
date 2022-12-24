import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as P

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), has_bias=bias, pad_mode='pad')

class PixelShuffle(nn.Cell):
    def __init__(self, upscale_factor):
        super().__init__()
        self.pixelshuffle = P.DepthToSpace(upscale_factor)
    
    def construct(self, x):
        return self.pixelshuffle(x)

class MeanShift(nn.Cell):
    def __init__(self, rgb_range=1, norm_mean=(0.4488, 0.4371, 0.4040), norm_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__()
        std = mindspore.Tensor(norm_std, mindspore.float32)
        newe = P.Eye()(3, 3, mindspore.float32).view(3, 3, 1, 1)
        new_std = std.view(3, 1, 1, 1)
        weight = mindspore.Tensor(newe, mindspore.float32) / mindspore.Tensor(new_std, mindspore.float32)
        bias = sign * rgb_range * mindspore.Tensor(norm_mean, mindspore.float32) / std
        for p in self.get_parameters():
            p.requires_grad = False
        self.meanshift = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1,
                                   has_bias=True, weight_init=weight, bias_init=bias)

    def construct(self, x):
        out = self.meanshift(x)
        return out

class BasicBlock(nn.SequentialCell):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels,momentum=0.1))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Cell):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats,momentum=0.1))
            if i == 0:
                m.append(act)

        self.body = nn.SequentialCell(*m)
        self.res_scale = res_scale

    def construct(self, x):
        res = mindspore.ops.Mul(self.body(x), self.res_scale)
        res += x

        return res

class Upsampler(nn.SequentialCell):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats,momentum=0.1))
                if act == 'relu':
                    m.append(nn.ReLU())
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats,momentum=0.1))
            if act == 'relu':
                m.append(nn.ReLU())
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


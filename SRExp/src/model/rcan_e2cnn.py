## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
from e2cnn import nn as en
from e2cnn import gspaces
from e2cnn.nn.modules.equivariant_module import EquivariantModule
import mindspore.nn as nn
import mindspore.ops as P
import mindspore

def make_model(args, parent=False):
    return RCAN_E2CNN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = P.AdaptiveAvgPool2D(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.SequentialCell(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, pad_mode='pad', has_bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, pad_mode='pad', has_bias=True),
                nn.Sigmoid()
        )

    def construct(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Cell):
    def __init__(
        self, normal_type, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(en.R2Conv(normal_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat,momentum=0.1))
            if i == 0: modules_body.append(act)
        # modules_body.append(CALayer(n_feat, reduction))
        self.body = en.SequentialModule(*modules_body)
        self.res_scale = res_scale

    def construct(self, x):
        res = self.body(x)*(self.res_scale)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Cell):
    def __init__(self, normal_type, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB( normal_type, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            #en.R2Conv(normal_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)\
            for _ in range(n_resblocks)]
        modules_body.append(en.R2Conv(normal_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True))
        self.body = nn.SequentialCell(modules_body)
        self.res_scale = res_scale

    def construct(self, x):
        res = self.body(x)*(self.res_scale)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN_E2CNN(nn.Cell):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN_E2CNN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 5
        reduction = args.reduction 
        scale = args.scale[0]
        
        tranNum = args.tranNum
        self.r2_act = gspaces.Rot2dOnR2(tranNum)

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        
        begin_type = en.FieldType(self.r2_act, args.n_colors*[self.r2_act.trivial_repr])
        normal_type = en.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr]) 
        act = en.ReLU(normal_type, inplace=True)
        modules_head = [ en.R2Conv(begin_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]

        # define body module
        modules_body = [
            ResidualGroup(normal_type, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            #en.R2Conv(normal_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True)\
            #RCAB( normal_type, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1)
            for _ in range(n_resgroups)]

        modules_body.append(en.R2Conv(normal_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats*tranNum, act=False),
            conv(n_feats*tranNum, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = en.SequentialModule(modules_head)
        self.body = nn.SequentialCell(modules_body)
        self.tail = nn.SequentialCell(modules_tail)
        self.begin_type = begin_type

    def construct(self, x):

        x = self.sub_mean(x)
        x = en.GeometricTensor(x, self.begin_type)
        x = self.head(x)

        res = self.body(x)
        x = x
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.parameters_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, mindspore.Parameter):
                    param = param.data
                try:
                    own_state[name].copy(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].shape, param.shape))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

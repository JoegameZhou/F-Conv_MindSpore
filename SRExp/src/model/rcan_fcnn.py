## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
from model import F_Conv as fn
import mindspore.nn as nn
import mindspore

def make_model(args, parent=False):
    return RCAN_plus(args)


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Cell):
    def __init__(
        self, tranNum, inP, Smooth,  n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1.0, iniScale=1.0):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(fn.Fconv_PCA(kernel_size,n_feat,n_feat,tranNum,inP=inP,padding=(kernel_size-1)//2,  Smooth = Smooth, iniScale = iniScale))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat,momentum=0.1))
            if i == 0: modules_body.append(act)
#        modules_body.append(CALayer(tranNum, n_feat, reduction))
        self.body = nn.SequentialCell(modules_body)
        self.res_scale = res_scale

    def construct(self, x):
        res = mindspore.ops.Mul()(self.body(x), self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Cell):
    def __init__(self, tranNum, inP, Smooth, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, iniScale=1.0):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(tranNum, inP, Smooth, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(), res_scale=1.0, iniScale = iniScale) \
            for _ in range(n_resblocks)]
        modules_body.append(fn.Fconv_PCA(kernel_size,n_feat,n_feat,tranNum,inP=inP,padding=(kernel_size-1)//2,  Smooth = Smooth))
        self.body = nn.SequentialCell(modules_body)
        self.res_scale = res_scale
    def construct(self, x):
        res = mindspore.ops.Mul()(self.body(x), self.res_scale)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN_plus(nn.Cell):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN_plus, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        # kernel_size = 3
        kernel_size = int(args.kernel_size)
        inP = kernel_size
        tranNum = args.tranNum
        iniScale = args.ini_scale
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU()
        Smooth = False
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [fn.Fconv_PCA(kernel_size,args.n_colors,n_feats,tranNum,inP=inP,padding=(kernel_size-1)//2, ifIni=1, Smooth = Smooth, iniScale = iniScale)]

        # define body module
        modules_body = [
            ResidualGroup(
                tranNum, inP, Smooth, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, iniScale = iniScale) \
            for _ in range(n_resgroups)]

        modules_body.append(fn.Fconv_PCA(kernel_size,n_feats,n_feats,tranNum,inP=inP,padding=(kernel_size-1)//2,  Smooth = Smooth, iniScale = iniScale))

        # define tail module
        modules_tail = [
            fn.Fconv_PCA_out(kernel_size,n_feats,n_feats*tranNum,tranNum,inP=inP,padding=(kernel_size-1)//2,  Smooth = Smooth, iniScale = iniScale),
            common.Upsampler(conv, scale, n_feats*tranNum, act=False),
            conv(n_feats*tranNum, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.SequentialCell(modules_head)
        self.body = nn.SequentialCell(modules_body)
        self.tail = nn.SequentialCell(modules_tail)

    def construct(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
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

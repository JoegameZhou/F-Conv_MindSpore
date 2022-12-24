from model import common

import mindspore.nn as nn
import mindspore
from model import PDOe as fn

def make_model(args, parent=False):
    return edsu_PDOe(args)

class edsu_PDOe(nn.Cell):
    def __init__(self, args):
        super(edsu_PDOe, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = int(args.kernel_size)
        scale = args.scale[0]
        act = nn.ReLU()
        inP = kernel_size
        tranNum = args.tranNum
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        Smooth = False
        m_head =  [fn.Fconv_PCA(kernel_size,args.n_colors,n_feats,tranNum,inP=inP,padding=(kernel_size-1)//2, ifIni=1, Smooth = Smooth)]

        # define body module
        m_body = [
            fn.ResBlock(
                fn.Fconv_PCA, n_feats, kernel_size,tranNum = tranNum, inP = inP,  act=act, res_scale=args.res_scale, Smooth = Smooth
            ) for _ in range(n_resblocks)
        ]
#        m_body.append(fn.GroupFusion(n_feats, tranNum))
        # 要加一个整合不同 tranNum 的层
        # define tail module
        conv = common.default_conv
        n_feats = n_feats*tranNum
        m_tail = [
#            fn.GroupFusion(n_feats, tranNum),    
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, 3)
        ]

        self.head = nn.SequentialCell(m_head)
        self.body = nn.SequentialCell(m_body)
        self.tail = nn.SequentialCell(m_tail)

    def construct(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.parameters_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, mindspore.Parameter):
                    param = param.data
                try:
                    own_state[name].copy(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].shape, param.shape))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


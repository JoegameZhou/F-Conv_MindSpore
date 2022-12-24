# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import mindspore.ops as P
import mindspore.nn as nn
import mindspore


def make_model(args, parent=False):
    return Bicubic(args)


class Bicubic(nn.Cell):
    def __init__(self, args):
        super(Bicubic, self).__init__()
        self.r = args.scale[0]
        self.weights = mindspore.Parameter(P.StandardNormal()((1,1)), requires_grad=True)
      

    def construct(self, x):       
        w = self.weights
        w = w
        #mindspore1.7版本还不支持F.interpolate(x, scale_factor=self.r, mode='bicubic')功能
        #该文件在整个项目中没有用到，因此没去实现该功能
        return mindspore.nn.ResizeBilinear(x, scale_factor=self.r)

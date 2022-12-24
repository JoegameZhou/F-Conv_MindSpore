#还没有全部实现该文件功能
from model import common

import mindspore
import mindspore.nn as nn
import mindspore_hub as mshub

class VGG(nn.Cell):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        #download url: https://download.mindspore.cn/models/r1.7/vgg19_ascend_v170_imagenet2012_research_cv_top1acc74.29_top5acc91.99.ckpt
        vgg_features = mindspore.load_checkpoint("./vgg19_ascend_v170_imagenet2012_research_cv_top1acc74.29_top5acc91.99.ckpt").features
        # vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.SequentialCell(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.SequentialCell(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.get_parameters():
            p.requires_grad = False

    def construct(self, sr, hr):
        def _construct(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _construct(sr)
        vgg_hr = _construct(hr)

        loss = nn.MSELoss()(vgg_sr, vgg_hr)

        return loss

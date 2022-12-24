import os
from importlib import import_module

import mindspore
import mindspore.nn as nn
from mindspore import context
# import torch.utils.model_zoo

class Model(nn.Cell):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = args.device
        self.n_NPUs = args.n_NPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)

        if args.precision == 'half':
            self.model.to_float(mindspore.float16)

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def construct(self, x, idx_scale=0):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_NPUs > 1:
                context.set_auto_parallel_context(parallel_mode="data_parallel")
                return self.model(x)
            else:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.construct

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.ckpt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.ckpt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.ckpt'.format(epoch))
            )

        for s in save_dirs:
            mindspore.save_checkpoint(self.model, s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        if resume == -1:
            load_from = mindspore.load_checkpoint(os.path.join(apath, 'model_latest.ckpt'))
        elif resume == 1:
            load_from = mindspore.load_checkpoint(os.path.join(apath, 'model_best.ckpt'))
        elif resume == 0:
            # 还未实现此功能
            # if pre_train == 'download':
            #     print('Download the model')
            #     dir_model = os.path.join('..', 'models')
            #     os.makedirs(dir_model, exist_ok=True)
            #     load_from = torch.utils.model_zoo.load_url(
            #         self.model.url,
            #         model_dir=dir_model,
            #         **kwargs
            #     )
            # elif pre_train:
            #     print('Load the model from {}'.format(pre_train))
            #     load_from = mindspore.load_checkpoint('../experiment/'+pre_train+'/model/model_latest.ckpt', **kwargs)
            # else:
            #     load_from = False
            if pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = mindspore.load_checkpoint('../experiment/'+pre_train+'/model/model_latest.ckpt')
            else:
                load_from = False
        else:
            load_from = mindspore.load_checkpoint(os.path.join(apath, 'model_{}.ckpt'.format(resume)))

        if load_from:
            mindspore.load_param_into_net(self.model, load_from, strict_load=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        scale = 1 if self.input_large else self.scale[self.idx_scale]
        n_NPUs = min(self.n_NPUs, 4)
        # height, width
        h, w = args[0].shape[-2:]

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [mindspore.ops.Concat()([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_NPUs):
                x = [x_chop[i:(i + n_NPUs)] for x_chop in x_chops]
                y = self.model(*x)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in mindspore.ops.Split(axis=0, output_num=n_NPUs)(_y)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(mindspore.ops.Split(axis=0, output_num=n_NPUs)(_y))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].shape[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.astype(mindspore.float32, copy=False)

            v2np = v.asnumpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = mindspore.Tensor(tfnp)
            if self.precision == 'half': ret = ret.astype(mindspore.float16, copy=False)

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [mindspore.Concat(0)(_y).mean(axis=0, keep_dims=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y

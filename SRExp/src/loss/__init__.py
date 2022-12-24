import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as P

class Loss(nn.LossBase):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_NPUs = args.n_NPUs
        self.loss = []
        self.loss_module = nn.CellList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = mindspore.Tensor([])

        if args.precision == 'half': self.loss_module.to_float(mindspore.float16)
        if not args.cpu and args.n_NPUs > 1:
            mindspore.context.set_auto_parallel_context(parallel_mode="data_parallel")

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def construct(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum

        return loss_sum

    def start_log(self):
        self.log = P.Zeros()((1, len(self.loss)), mindspore.float32)

    def end_log(self, n_batches):
        self.log[-1] = P.Div()(self.log[-1], n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], float(c / n_samples)))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            if i>0:
                plt.plot(axis, self.log[:, i].asnumpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        mindspore.save_checkpoint(self, os.path.join(apath, 'loss.ckpt'))
        np.save(os.path.join(apath, 'loss_log.npy'), self.log.asnumpy())

    def load(self, apath, cpu=False):
        mindspore.load_param_into_net(self, mindspore.load_checkpoint(os.path.join(apath, 'loss.ckpt')))
        self.log = mindspore.Tensor(np.load(os.path.join(apath, 'loss_log.npy')))
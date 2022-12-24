import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
import scipy.io as sio    

import matplotlib
from skimage import io
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import mindspore

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = mindspore.Tensor([])
        self.ssim = mindspore.Tensor([])
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:#file name to load
            if not args.save:#file name to save
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = mindspore.load_checkpoint(self.get_path('psnr_log.ckpt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:#reset the training
            os.system('DEL /s ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        np.save(self.get_path('psnr_log.npy'), self.log.asnumpy())
        np.save(self.get_path('ssim_log.ckpt.npy'), self.ssim.asnumpy())
        
    def add_log(self, log):
        self.log = log
    def add_ssim(self, ssim):
        self.ssim = ssim

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].asnumpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.asnumpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) 
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

        
    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.name),
                str(filename)+'_x{}_'.format(scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = mindspore.ops.Mul()(v[0], 255 / self.args.rgb_range)
                tensor_cpu = normalized.astype(mindspore.uint8, copy=False)
                tensor_cpu = mindspore.ops.Transpose()(tensor_cpu, (1, 2, 0))
                io.imsave((str(filename)+'{}.png'.format(p)), tensor_cpu)
                if self.args.save_mat:
                    sio.savemat((str(filename)+'{}.mat'.format(p)), {'X':tensor_cpu})   
    def save_psnr(self, dataset, scale, psnr_all, ssim_all):

        filename1 = self.get_path(
            'results-{}'.format(dataset.name),
            'psrn_x{}_'.format(scale)
        )
        filename2 = self.get_path(
            'results-{}'.format(dataset.name),
            'ssim_x{}_'.format(scale)
        )
        print('{}.mat'.format(filename1))
        sio.savemat(('{}.mat'.format(filename1)), {'psnr_all':psnr_all})  
        sio.savemat(('{}.mat'.format(filename2)), {'ssim_all':ssim_all})

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    image1 = mindspore.ops.Mul()(img, pixel_range)
    image2 = mindspore.ops.clip_by_value(image1, mindspore.Tensor(0, mindspore.float32), mindspore.Tensor(255, mindspore.float32))
    image3 = mindspore.ops.Rint()(image2)
    image4 = mindspore.ops.Div()(image3, pixel_range)
    return image4

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.size == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.benchmark:
        shave = scale
        if diff.shape[1] > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            gray_coeffs = mindspore.Tensor(gray_coeffs, mindspore.float32)
            convert = gray_coeffs.view(1, 3, 1, 1) / 256
            diff = mindspore.ops.Mul()(diff, convert)
            diff = diff.sum(axis=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    valid = mindspore.ops.Pow()(valid, 2)
    mse = valid.mean()

    return -10 * math.log10(mse)

def calc_ssim(sr, hr, scale, rgb_range, dataset=None):
    if hr.size == 1: return 0

    sr = sr/rgb_range
    hr = hr/rgb_range
    if dataset and dataset.benchmark:
        shave = scale
        if sr.shape[1] > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            gray_coeffs = mindspore.Tensor(gray_coeffs, mindspore.float32)
            convert = gray_coeffs.view(1, 3, 1, 1) / 256
            sr = mindspore.ops.Mul()(sr, convert)
            sr = sr.sum(axis=1)
            sr = mindspore.ops.ExpandDims()(sr, 1)
            hr = mindspore.ops.Mul()(hr, convert)
            hr = hr.sum(axis=1)
            hr = mindspore.ops.ExpandDims()(hr, 1)
    else:
        shave = scale + 6

    sr = sr[..., shave:-shave, shave:-shave]
    hr = hr[..., shave:-shave, shave:-shave]
    ssim = mindspore.nn.SSIM()(sr,hr)
    return ssim


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    kwargs_optimizer = {'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = mindspore.nn.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = mindspore.nn.Adam
        kwargs_optimizer['beta1'] = args.betas[0]
        kwargs_optimizer['beta2'] = args.betas[1]
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = mindspore.nn.RMSProp
        kwargs_optimizer['epsilon'] = args.epsilon
    
    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    milestones.append(args.epochs)
    learning_rates = [args.lr*math.pow(args.gamma,i) for i in range(len(milestones))]
    scheduler = mindspore.nn.piecewise_constant_lr(milestones, learning_rates)
    kwargs_optimizer['learning_rate'] = scheduler


    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def save(self, save_dir):
            mindspore.save_checkpoint(self, self.get_dir(save_dir))

        def load(self, load_dir):
            self.load_state_dict(mindspore.load_checkpoint(self.get_dir(load_dir)))

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.ckpt')

        def get_learningrate(self):
            return float(self.get_lr())
    
    optimizer = CustomOptimizer(target.trainable_params(), **kwargs_optimizer)
    return optimizer
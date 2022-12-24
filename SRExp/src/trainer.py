import os
import math
from decimal import Decimal

import utility

import mindspore
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.train_dataset = loader.train_dataset
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self, epoch):
        lr = self.optimizer.get_learningrate()
        
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()

        loss_net = mindspore.nn.WithLossCell(self.model, self.loss)
        train_net = mindspore.nn.TrainOneStepCell(loss_net, self.optimizer)
        train_net.set_train(mode=True)

        timer_data, timer_model = utility.timer(), utility.timer()
        self.train_dataset.set_scale(0)
        for batch,data in enumerate(self.loader_train):
            lr = data['image']
            hr = data['label']
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            train_net(lr, hr)

            # sr = self.model(lr, 0)
            # loss = self.loss(sr, hr)
            if self.args.gclip > 0:
                mindspore.ops.clip_by_value(
                    filter(lambda x: x.requires_grad, self.model.get_parameters()),
                    clip_value_min = -self.args.gclip,
                    clip_value_max = self.args.gclip
                    )

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.train_dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(batch)
        self.error_last = self.loss.log[-1, -1]

    def test(self, epoch=0):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            mindspore.ops.Zeros()((1, len(self.loader_test), len(self.scale)), mindspore.float32)
        )
        self.ckp.add_ssim(
            mindspore.ops.Zeros()((1, len(self.loader_test), len(self.scale)), mindspore.float32)
        )
        self.model.set_train(mode=False)

        timer_test = utility.timer()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                testset = d[0]
                loader_test = d[1]
                testset.set_scale(idx_scale)
                psnr_all = []
                ssim_all = []
                for data in loader_test:
                    lr = data['image']
                    hr = data['label']
                    filename = data['filename']
                    lr, hr = self.prepare(lr, hr)

                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    temp_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=testset)
                    self.ckp.log[-1, idx_data, idx_scale] += temp_psnr
                    psnr_all.append(temp_psnr)
                    if self.args.ssim:
                        temp_ssim = utility.calc_ssim(sr, hr, scale, self.args.rgb_range, dataset=testset)
                        temp_ssim = float(temp_ssim)
                        self.ckp.ssim[-1, idx_data, idx_scale] += temp_ssim
                        ssim_all.append(temp_ssim)
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(testset, filename, save_list, scale)
                

                self.ckp.save_psnr(testset, scale, psnr_all, ssim_all)
                self.ckp.log[-1, idx_data, idx_scale] /= len(testset)
                self.ckp.ssim[-1, idx_data, idx_scale] /= len(testset)
                best = mindspore.ops.ArgMaxWithValue(0)(self.ckp.log)
                best_value = best[1][idx_data, idx_scale]
                best_idx = best[0][idx_data, idx_scale]
                
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: '.format(
                        testset.name,
                        scale
                    )+str(self.ckp.log[-1,idx_data,idx_scale])+" (Best: "+str(best_value)+" @epoch "+str(best_idx+1)+")"
                )
                if self.args.ssim:
                    self.ckp.write_log(
                        '[{} x{}]\tSSIM: '.format(
                            testset.name,
                            scale,
                        )+str(self.ckp.ssim[-1,idx_data, idx_scale])+" (Best: "+str(self.ckp.ssim[best_idx,idx_data,idx_scale])+" @epoch "+str(best_idx+1)+")"
                    )                   


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

#        if self.args.save_results:
#            self.ckp.end_background()

        if not self.args.test_only:
            self.model.set_train(mode=True)
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.astype(mindspore.float16, copy=False)
            return tensor

        return [_prepare(a) for a in args]
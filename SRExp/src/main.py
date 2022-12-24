import mindspore

import utility
import data
import model
import loss
from option_argument import args
from trainer import Trainer

mindspore.set_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    mindspore.context.set_context(device_target=args.device,mode=mindspore.context.PYNATIVE_MODE)
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            if args.test_only:
                t.test()
            else:
                for i in range(args.epochs):
                    t.train(i+1)
                    t.test(i+1)

            checkpoint.done()

if __name__ == '__main__':
    main()

from importlib import import_module
import mindspore.dataset as ds
class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                print(module_name.lower())
                m = import_module('data.' + module_name.lower())
                datasets = getattr(m, module_name)(args, name=d)
            print(datasets)

            self.train_dataset = datasets
            train_dataset = ds.GeneratorDataset(source=datasets, column_names=['image','label','filename'], num_parallel_workers=args.n_threads, shuffle=False)
            train_dataset = train_dataset.batch(batch_size=args.batch_size)
            self.loader_train = train_dataset.create_dict_iterator()
            # self.loader_train = dataloader.DataLoader(
            #     MyConcatDataset(datasets),
            #     batch_size=args.batch_size,
            #     shuffle=True,
            #     pin_memory=not args.cpu,
            #     num_workers=args.n_threads,
            # )
            

        self.loader_test = []
        for d in args.data_test:
            if d in ['Alldata','Alldata2','Set5','Set5_hat','Set14','Set14_hat','B100','B100_hat','Urban100','Urban100_hat','DIV2K_test_our','DIV2K_test_Rot','DIV2K_test_hat']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            test_dataset = ds.GeneratorDataset(source=testset, column_names=['image','label','filename'], num_parallel_workers=args.n_threads,shuffle=False)
            test_dataset = test_dataset.batch(batch_size=1)
            loader_test = test_dataset.create_dict_iterator()
            self.loader_test.append([testset, loader_test])
            # self.loader_test.append(
            #     dataloader.DataLoader(
            #         testset,
            #         batch_size=1,
            #         shuffle=False,
            #         pin_memory=not args.cpu,
            #         num_workers=args.n_threads,
            #     )
            # )

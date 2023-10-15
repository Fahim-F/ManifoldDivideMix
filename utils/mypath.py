class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'webvision':
            return '/datasets/webvision/'
        elif dataset == 'clothing':
            return '/data/clothing1M/'
        elif dataset == 'miniimagenet':
            return '/datasets/dataset/mini-imagenet/'
        elif dataset == 'cifar100':
            return 'samples/cifar100/'
        elif dataset == 'imagenet32':
            return '/datasets/'
        elif dataset == 'place365':
            return '/datasets/place365/'
        else:
            raise NotImplementedError('Dataset {} not available.'.format(dataset))
        
#!/usr/bin/env -S python3 -u

from torchvision.datasets import CIFAR10


class CIFAR11(CIFAR10):
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ['training-flickr', 'f59e7936f9ce356ee5b19e05b1931ec9'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ['testing-flickr', 'af8b6b74b97997334ecf6949feb23639'],
    ]


# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 16 2021, 10:54 [CST]

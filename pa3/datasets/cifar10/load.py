#!/usr/bin/env -S python3 -u

# Load cifar-10 dataset (32x32 3-channel images)


def unpickle(file: str):
    import pickle
    with open(file, 'rb') as f:
        ret = pickle.load(f, encoding='bytes')
    return ret


def main():
    flist = [
        "./cifar-10-batches-py/data_batch_1",
    ]
    for file in flist:
        print(unpickle(file)[b'file'])


if __name__ == "__main__":
    main()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 15 2021, 15:49 [CST]

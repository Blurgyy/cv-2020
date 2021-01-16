#!/usr/bin/env -S python3 -u

# Load cifar-10 dataset (32x32 3-channel images)


def unpickle(file: str):
    import pickle
    with open(file, 'rb') as f:
        ret = pickle.load(f, encoding='latin1')
    return ret


def main():
    flist = [
        "./cifar-10-batches-py/data_batch_1",
    ]
    for file in flist:
        data = unpickle(file)
        for key in data.keys():
            print(key, type(data[key]))
        print(data['data'].shape, data['data'].dtype)


if __name__ == "__main__":
    main()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 15 2021, 15:49 [CST]

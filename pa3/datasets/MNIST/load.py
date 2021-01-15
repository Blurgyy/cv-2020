#!/usr/bin/env -S python3 -u

# Load mnist dataset (28x28 single channel images)

import matplotlib.pyplot as plt
import numpy as np


def loadimg(imgfile: str, idxfile: str, nimages: int):
    size = 28
    # Get images
    with open(imgfile, 'rb') as f:
        image_header = f.read(16)  # useless header bits?
        with open("image_header", 'wb') as f2:
            f2.write(image_header)
        buf = f.read(size * size * nimages)
        data = np.frombuffer(buf, np.uint8)
    images = data.reshape(nimages, size, size, 1)
    # Get labels
    with open(idxfile, 'rb') as f:
        index_header = f.read(8)
        with open("index_header", 'wb') as f2:
            f2.write(index_header)
        buf = f.read(nimages)
        data = np.frombuffer(buf, np.uint8)
    labels = data.reshape(-1)
    return images, labels


def dumpimg(imgfile: str, idxfile: str, data: dict):
    with open("./image_header", 'rb') as f:
        image_header = f.read()
    with open("./index_header", 'rb') as f:
        index_header = f.read()


def main():
    train_set_param = {
        'imgfile': "./raw/train-images-idx3-ubyte",
        'idxfile': "./raw/train-labels-idx1-ubyte",
        'nimages': 60000,
    }
    test_set_param = {
        'imgfile': "./raw/t10k-images-idx3-ubyte",
        'idxfile': "./raw/t10k-labels-idx1-ubyte",
        "nimages": 10000,
    }

    # Augment training set
    images, labels = loadimg(**train_set_param)
    print(images.shape, labels.shape)


if __name__ == "__main__":
    main()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 15 2021, 15:59 [CST]

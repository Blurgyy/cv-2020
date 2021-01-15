#!/usr/bin/env -S python3 -u

# Load mnist dataset (28x28 single channel images)

import matplotlib.pyplot as plt
import numpy as np


def unpickle(file: str, nimages: int):
    size = 28
    with open(file, 'rb') as f:
        f.read(16)  # useless header bits?
        buf = f.read(size * size * nimages)
        data = np.frombuffer(buf, np.uint8)
    return data.reshape(nimages, size, size, 1)


def main():
    plist = [
        {
            'file': "./raw/train-images-idx3-ubyte",
            'nimages': 60000,
        },
    ]

    for param in plist:
        data = unpickle(**param)
        image = np.asarray(data[2]).squeeze()
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    main()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 15 2021, 15:59 [CST]

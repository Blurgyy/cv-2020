#!/usr/bin/env -S python3 -u

import os
import pickle
import numpy as np
from PIL import Image


def main():
    dirs = ["testing", "training"]
    for directory in dirs:
        data = {
            'batch_label': "",
            'labels': [],
            'data': np.array([], dtype=np.uint8),
            'filenames': []
        }
        nimg = 0
        for root, _, files in os.walk(directory):
            for file in files:
                nimg += 1
                fpath = os.path.join(root, file)
                img = Image.open(fpath)
                img = np.array(
                    img, dtype=np.uint8).transpose(2, 0, 1).reshape(-1)
                data['data'] = np.append(data['data'], img)
                data['labels'].append(10)
        data['data'] = data['data'].reshape(nimg, -1)
        print(data['data'].dtype)
        print(data['data'].shape)
        assert data['data'].shape[1] == 3072
        with open(directory + "-flicker", 'wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 16 2021, 12:14 [CST]

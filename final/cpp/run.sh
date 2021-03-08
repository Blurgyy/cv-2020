#!/usr/bin/env -S bash

if ./build/stereo data/Recycle-perfect/im0.png data/Recycle-perfect/im1.png data/Recycle-perfect/calib.txt; then
    scp *jpg 10.186.103.185:repos/cv/final/cpp
fi

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Mar 07 2021, 22:36 [CST]

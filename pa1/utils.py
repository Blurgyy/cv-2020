#!/usr/bin/env -S python3 -u

import cv2
import numpy


def downsample(img: numpy.ndarray, factor: int = 2) -> numpy.ndarray:
    return img[::factor, ::factor,]


def blend(
        imgs: numpy.ndarray,
        imgt: numpy.ndarray,
        winname: str,
        writer: cv2.VideoWriter,
        elapse: float = 1.0,  # in seconds
        fps: int = 24):
    assert imgs.shape == imgt.shape

    fcnt = int((elapse * fps + 1) / 2)  # ceil
    for f in range(fcnt):
        prog = f / fcnt
        inter = (imgs * (1 - prog)).astype(numpy.uint8)
        cv2.imshow(winname, inter)
        writer.write(inter)
        cv2.waitKey(int(1000 / fps))
    for f in range(fcnt):
        prog = f / fcnt
        inter = (imgt * prog).astype(numpy.uint8)
        cv2.imshow(winname, inter)
        writer.write(inter)
        cv2.waitKey(int(1000 / fps))


def set_pixel(img: numpy.ndarray, x: int, y: int,
              color: tuple = (255, 255, 255)):
    assert len(img.shape) == 3 and img.shape[2] == 3
    img[x, y] = color


def coscurve(frame: numpy.ndarray,
             pos: int,
             width: int = 1920,
             height: int = 1080):
    for i in range(height):
        j = int(pos + width / 8 * numpy.sin(2 * numpy.pi / (height * 2) *
                                         (i - height // 2.5)))
        set_pixel(frame, i, j)


# Author: Blurgy <gy@blurgy.xyz>
# Date:   Dec 01 2020, 13:51 [CST]

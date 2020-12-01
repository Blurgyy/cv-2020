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


def set_pixel(img: numpy.ndarray,
              x: int,
              y: int,
              color: tuple = (255, 255, 255)):
    assert len(img.shape) == 3 and img.shape[2] == 3
    # Swap R and B channel for OpenCV
    color = (color[2], color[1], color[0])
    img[x, y] = color


def coastline(frame: numpy.ndarray,
              pos: int,
              width: int = 1920,
              height: int = 1080) -> numpy.ndarray:
    costlinepos = lambda pos, width, height: int(pos + width / 8 * numpy.sin(
        2 * numpy.pi / (height * 2) * (i - height // 2.5)))
    lava_color = (0xff, 0x79, 0x1b)
    island_color = (0x23, 0x11, 0x0f)

    for i in range(height):
        for j in range(width):
            cp = costlinepos(pos, width, height)
            if j < cp:
                set_pixel(frame, i, j, lava_color)
            else:
                set_pixel(frame, i, j, island_color)
    return frame


def mustafar(width: int = 1920, height: int = 1080) -> list:
    ret = []
    frame = numpy.zeros((height, width, 3), dtype=numpy.uint8)

    ret.append(coastline(frame, width // 4, width, height))  # Background

    return ret


# Author: Blurgy <gy@blurgy.xyz>
# Date:   Dec 01 2020, 13:51 [CST]

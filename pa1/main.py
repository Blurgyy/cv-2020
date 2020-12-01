#!/usr/bin/env -S python3 -u
# Do not enable conda's virtualenv for this script

import utils

import cv2 as cv
import numpy as np
import sys


# :param prog: has possible values 0, 1, 2, 3, indicates drawing's
#              completeness
def draw(prog: int) -> np.ndarray:
    # Use black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # print(frame.shape)

    return frame


# :param prog: has possible values 0, 1.  0 for zju logo, 1 for personal info
def fanfare(width: int = 1920, height: int = 1080) -> np.ndarray:
    logofile = "./stuff/zjulogo.png"
    logo = utils.downsample(cv.imread(logofile), 3)
    logo_width = logo.shape[0]
    logo_height = logo.shape[1]
    # Use black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    hor_begin = (width - logo_width) // 2
    ver_begin = (height - logo_height) * 1 // 4
    # print(frame.shape)
    frame[ver_begin:ver_begin + logo_height,
          hor_begin:hor_begin + logo_width,] = logo

    text = "I have the high ground"
    fontface = cv.FONT_ITALIC
    origin = (hor_begin, ver_begin)
    color = (255, 255, 255)
    thickness = 2
    linetype = cv.LINE_AA

    cv.putText(frame, text, origin, fontface, 1, color, thickness, linetype)

    return frame


def credits(width: int = 1920, height: int = 1080) -> np.ndarray:
    # Use black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    return frame


def consume_key(key: int):
    if key == ord(' '):
        key = cv.waitKey()
    if key == ord('q'):
        cv.destroyAllWindows()
        sys.exit(0)


if __name__ == '__main__':
    # Global configs
    width = 1920
    height = 1080
    vidlen = 10
    fps = 24
    name = "kidsdrawing"

    # save file configs
    savfile = name + ".mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # Video writer
    out = cv.VideoWriter(savfile, fourcc, fps, (width, height))

    ############## Generate video pieces ##############
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    opening, kidsdrawing, ending = [], [], []
    # ObiWan's position
    ox, oy = width * 3 // 4, height // 3
    # Anakin's position
    ax, ay = width // 8, height * 2 // 3

    print("Generating fanfare ..")
    opening.append(fanfare())

    print("Drawing Mustafar ..")
    kidsdrawing.append(utils.Mustafar(frame, width // 4).copy())
    print("Drawing Anakin Skywalker ..")
    kidsdrawing.append(utils.Anakin(frame, (ax, ay)).copy())
    print("Drawing ObiWan Kenobi ..")
    kidsdrawing.append(utils.ObiWan(frame, (ox, oy)).copy())

    print("Generating credits ..")
    ending.append(credits())

    ############## Play by frame and save as video file ##############
    # Opening
    # print("Opening")

    # Kid's drawing
    # print("Content")
    for img in kidsdrawing:
        cv.imshow(name, img)
        cv.waitKey()

    # Ending
    # print("Ending")

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Nov 30 2020, 17:54 [CST]

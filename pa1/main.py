#!/usr/bin/env -S python3 -u
# Do not enable conda's virtualenv for this script

import utils

import cv2 as cv
import numpy as np
import sys


def fanfare_logo(width: int = 1920, height: int = 1080) -> np.ndarray:
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

    # Title text
    text = "PA1: Mustafar"
    fontface = cv.FONT_ITALIC
    origin = (width // 2 - 110, height // 8)
    color = (255, 255, 255)
    thickness = 2
    linetype = cv.LINE_AA

    cv.putText(
        frame,
        text,
        origin,
        fontface,
        1,
        color,
        thickness=thickness,
        lineType=linetype)

    return frame


def fanfare_me(width: int = 1920, height: int = 1080) -> np.ndarray:
    avatarfile = "./stuff/gy.jpg"
    avatar = utils.downsample(cv.imread(avatarfile), 2)
    avatar_width = avatar.shape[0]
    avatar_height = avatar.shape[1]
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    hor_begin = (width - avatar_width) // 2
    ver_begin = (height - avatar_height) * 1 // 4
    # print(frame.shape)
    frame[ver_begin:ver_begin + avatar_height,
          hor_begin:hor_begin + avatar_width,] = avatar

    # Title text
    text = "PA1: Mustafar"
    fontface = cv.FONT_ITALIC
    origin = (width // 2 - 110, height // 8)
    color = (255, 255, 255)
    thickness = 2
    linetype = cv.LINE_AA

    cv.putText(
        frame,
        text,
        origin,
        fontface,
        1,
        color,
        thickness=thickness,
        lineType=linetype)

    return frame


def credits_plot(width: int = 1920, height: int = 1080) -> np.ndarray:
    fontface = cv.FONT_HERSHEY_TRIPLEX
    color = (255, 255, 255)
    thickness = 2
    linetype = cv.LINE_AA
    # Use black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv.putText(
        frame,
        "Written and directed by", (width // 2 - 180, height // 2),
        fontface,
        1,
        color,
        thickness=thickness - 1,
        lineType=linetype)
    cv.putText(
        frame,
        "GEORGE LUCAS", (width // 2 - 110, height // 2 + 50),
        fontface,
        1,
        color,
        thickness=thickness,
        lineType=linetype)
    return frame


def credits_video(width: int = 1920, height: int = 1080) -> np.ndarray:
    fontface = cv.FONT_HERSHEY_TRIPLEX
    color = (255, 255, 255)
    thickness = 2
    linetype = cv.LINE_AA
    # Use black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv.putText(
        frame,
        "Video made by", (width // 2 - 150, height // 2),
        fontface,
        1,
        color,
        thickness=thickness - 1,
        lineType=linetype)
    cv.putText(
        frame,
        "Gaoyang Zhang", (width // 2 - 150, height // 2 + 50),
        fontface,
        1,
        color,
        thickness=thickness,
        lineType=linetype)
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
    opening.append(fanfare_logo())
    opening.append(fanfare_me())

    # print("Drawing Mustafar ..")
    # kidsdrawing.append(utils.Mustafar(frame, width // 4).copy())
    print("Drawing Anakin Skywalker ..")
    kidsdrawing.append(utils.Anakin(frame, (ax, ay)).copy())
    print("Drawing ObiWan Kenobi ..")
    kidsdrawing.append(utils.ObiWan(frame, (ox, oy)).copy())
    fullscene = frame.copy()
    print("Inserting quote ..")
    kidsdrawing.append(
        utils.quote_obiwan(fullscene.copy(), (ox, oy), 0).copy())
    kidsdrawing.append(
        utils.quote_obiwan(fullscene.copy(), (ox, oy), 1).copy())
    kidsdrawing.append(utils.quote_anakin(fullscene.copy(), (ax, ay)).copy())
    kidsdrawing.append(
        utils.quote_obiwan(fullscene.copy(), (ox, oy), 2).copy())
    print("Generating epilog ..")
    for i in range(2):
        kidsdrawing.append(utils.epilog(i))

    print("Generating credits ..")
    ending.append(credits_plot())
    ending.append(credits_video())

    ############## Play by frame and save as video file ##############
    # Opening
    # print("Opening")
    for img in opening:
        cv.imshow(name, img)
        cv.waitKey()

    # Kid's drawing
    # print("Content")
    for img in kidsdrawing:
        cv.imshow(name, img)
        cv.waitKey()

    # Ending
    # print("Ending")
    for img in ending:
        cv.imshow(name, img)
        cv.waitKey()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Nov 30 2020, 17:54 [CST]

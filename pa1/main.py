#!/usr/bin/env -S python3 -u
# Do not enable conda's virtualenv for this script

import utils

import cv2 as cv
import numpy as np
import time


def fanfare_logo(width: int = 1920, height: int = 1080) -> np.ndarray:
    logofile = "./stuff/zjulogo.png"
    logo = utils.downsample(cv.imread(logofile), 3)
    logo_height = logo.shape[0]
    logo_width = logo.shape[1]
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
    avatar_height = avatar.shape[0]
    avatar_width = avatar.shape[1]
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

    # Name, student id, date&time
    texts = (
        "Gaoyang Zhang",
        "12021052",
        time.asctime(),
    )
    fontface = cv.FONT_ITALIC
    for i, text in enumerate(texts):
        cv.putText(
            frame,
            text, (width * 13 // 16, height * 7 // 8 + i * 40),
            fontface,
            0.7,
            color,
            1,
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
    cv.putText(
        frame,
        "12021052", (width // 2 - 100, height // 2 + 100),
        fontface,
        1,
        color,
        thickness=thickness - 1,
        lineType=linetype)
    return frame


if __name__ == '__main__':
    # Global configs
    width = 1920
    height = 1080
    vidlen = 10
    fps = 24
    waitms = int(1000 / fps)
    name = "Mustafar"

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
    opening.append(fanfare_me())  # 1

    print("Drawing Mustafar ..")
    kidsdrawing.append(utils.Mustafar(frame, width // 4).copy())
    print("Drawing Anakin Skywalker ..")
    kidsdrawing.append(utils.Anakin(frame, (ax, ay)).copy())  # 2
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
        utils.quote_obiwan(fullscene.copy(), (ox, oy), 2).copy())  # 7
    print("Generating epilog ..")
    for i in range(2):
        kidsdrawing.append(utils.epilog(i))  # 9

    print("Generating credits ..")
    ending.append(credits_plot())  # 10
    ending.append(credits_video())

    ############## Play by frame and save as video file ##############
    full = opening + kidsdrawing + ending
    for f in range(fps):
        out.write(full[0])
        cv.imshow(name, full[0])
        key = cv.waitKey(waitms)
        utils.consume_key(key)

    for i in range(1, len(full)):
        utils.blend(full[i - 1], full[i], name, out, i <= 2 or i > 8)
        elapse = 1 if i <= 3 or i == 7 or i > 8 else 0.7
        for f in range(int(fps * elapse)):
            out.write(full[i])
            cv.imshow(name, full[i])
            key = cv.waitKey(waitms)
            utils.consume_key(key)

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Nov 30 2020, 17:54 [CST]

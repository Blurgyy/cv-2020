#!/usr/bin/env -S python3 -u

import cv2
import numpy


def downsample(img: numpy.ndarray, factor: int = 2) -> numpy.ndarray:
    assert factor > 0
    return img if factor == 1 else img[::factor, ::factor,]


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
    # print(img.shape)
    img[y, x] = color


def sqdistance(pa: tuple, pb: tuple) -> float:
    assert len(pa) == len(pb)
    ret = 0
    for i in range(len(pa)):
        ret += (pa[i] - pb[i]) * (pa[i] - pb[i])
    # print("dist is", ret)
    return ret


# opos = (hor, ver)
def Jedi(frame: numpy.ndarray, pos: tuple, saber_on: str, sabercolor: tuple):
    x, y = pos
    color = (255, 255, 255)

    # Head
    for i in range(x - 20, x + 20):
        for j in range(y - 20, y + 20):
            if sqdistance((i, j), pos) <= 20 * 20:
                # print(i, j, pos)
                set_pixel(frame, i, j, color)

    # Body and limbs
    thickness = 1
    linetype = cv2.LINE_AA
    neckpos = (pos[0], pos[1] + 20)
    lowerpos = (pos[0], pos[1] + 50)
    lhandpos = (neckpos[0] - 15, neckpos[1] + 20)
    rhandpos = (neckpos[0] + 15, neckpos[1] + 20)
    lfootpos = (lowerpos[0] - 15, lowerpos[1] + 20)
    rfootpos = (lowerpos[0] + 15, lowerpos[1] + 20)
    cv2.line(
        frame,
        neckpos,
        lowerpos,
        color,
        thickness=thickness,
        lineType=linetype)
    cv2.line(
        frame,
        neckpos,
        lhandpos,
        color,
        thickness=thickness,
        lineType=linetype)
    cv2.line(
        frame,
        neckpos,
        rhandpos,
        color,
        thickness=thickness,
        lineType=linetype)
    cv2.line(
        frame,
        lowerpos,
        lfootpos,
        color,
        thickness=thickness,
        lineType=linetype)
    cv2.line(
        frame,
        lowerpos,
        rfootpos,
        color,
        thickness=thickness,
        lineType=linetype)

    # Light saber
    saberthickness = 2
    sabercolor = (sabercolor[2], sabercolor[1], sabercolor[0])
    if saber_on == "righthand":
        sabertip = (rhandpos[0] + 20, rhandpos[1] - 55)
        cv2.line(
            frame,
            rhandpos,
            sabertip,
            sabercolor,
            thickness=saberthickness + 1,
            lineType=linetype)
        cv2.line(
            frame,
            rhandpos,
            sabertip,
            color,
            thickness=saberthickness,
            lineType=linetype)
    elif saber_on == "lefthand":
        sabertip = (lhandpos[0] - 20, rhandpos[1] - 55)
        cv2.line(
            frame,
            lhandpos,
            sabertip,
            sabercolor,
            thickness=saberthickness + 10,
            lineType=linetype)
        cv2.line(
            frame,
            lhandpos,
            sabertip,
            color,
            thickness=saberthickness,
            lineType=linetype)


# :param ipos: The middle point of the floating island's top
def FloatingIsland(frame: numpy.ndarray, ipos: tuple) -> numpy.ndarray:
    height, width = frame.shape[0:2]
    ix, iy = ipos
    assert 0 <= ix < width and 0 <= iy < height, f"ipos is {ipos}"
    island_color = (0x23, 0x11, 0x0f)
    topleft = (ix - 250, iy)
    botright = (ix + 200, iy + 60)
    for i in range(topleft[0], botright[0]):
        for j in range(topleft[1], botright[1]):
            set_pixel(frame, i, j, island_color)


def Mustafar(frame: numpy.ndarray, pos: int) -> numpy.ndarray:
    height, width = frame.shape[0:2]
    costlinepos = lambda pos, width, height: int(pos + width / 8 * numpy.sin(
        2 * numpy.pi / (height * 2) * (i - height // 2.5)))
    lava_color = (0xff, 0x79, 0x1b)
    island_color = (0x23, 0x11, 0x0f)

    for i in range(height):
        for j in range(width):
            cp = costlinepos(pos, width, height)
            if j < cp:
                set_pixel(frame, j, i, lava_color)
            else:
                set_pixel(frame, j, i, island_color)
    return frame


def Anakin(frame: numpy.ndarray, apos: tuple) -> numpy.ndarray:
    width, height = frame.shape[0:2]
    ax, ay = apos
    assert 0 <= ax < width and 0 <= ay < height, f"Anakin is at {apos}"
    FloatingIsland(frame, (ax, ay + 70))
    Jedi(frame, apos, saber_on="righthand", sabercolor=(0x9e, 0xa0, 0xd7))
    return frame


# opos = (hor, ver)
def ObiWan(frame: numpy.ndarray, opos: tuple) -> numpy.ndarray:
    height, width = frame.shape[0:2]
    ox, oy = opos
    assert 0 <= ox < width and 0 <= oy < height, f"ObiWan is at {opos}"
    Jedi(frame, opos, saber_on="lefthand", sabercolor=(0x9e, 0xa0, 0xd7))
    return frame


def quote_obiwan(frame: numpy.ndarray, opos: tuple,
                 quoteid: int) -> numpy.ndarray:
    height, width = frame.shape[0:2]
    ox, oy = opos
    assert 0 <= ox < width and 0 <= oy < height, f"ObiWan is at {opos}"
    quotes = [
        ("It's over", "Anakin"),
        ("I have the", "highground"),
        ("Don't try it",),
    ]
    current_quote = quotes[quoteid]
    fontface = cv2.FONT_HERSHEY_COMPLEX
    color = (255, 255, 255)
    linetype = cv2.LINE_AA

    for i, piece in enumerate(current_quote):
        cv2.putText(
            frame,
            piece, (ox - 150, oy - 60 + i * 25),
            fontface,
            0.8,
            color,
            thickness=1,
            lineType=linetype)
    return frame


def quote_anakin(frame: numpy.ndarray, apos: tuple) -> numpy.ndarray:
    height, width = frame.shape[0:2]
    ax, ay = apos
    assert 0 <= ax < width and 0 <= ay < height, f"Anakin is at {apos}"
    youunderestimatemypower = (
        "You underestimate",
        "my power",
    )
    fontface = cv2.FONT_HERSHEY_COMPLEX
    color = (255, 255, 255)
    linetype = cv2.LINE_AA

    for i, piece in enumerate(youunderestimatemypower):
        cv2.putText(
            frame,
            piece, (ax + 20, ay - 75 + i * 25),
            fontface,
            0.8,
            color,
            thickness=1,
            lineType=linetype)
    return frame


def epilog(epilogid: int,
           width: int = 1920,
           height: int = 1080) -> numpy.ndarray:
    frame = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    text = [
        ("Anakin tried anyway.",),
        ("Anakin lost his leg.",),
    ]
    fontface = cv2.FONT_HERSHEY_COMPLEX
    color = (255, 255, 255)
    linetype = cv2.LINE_AA

    for i, piece in enumerate(text[epilogid]):
        cv2.putText(
            frame,
            piece, (width // 2 - 125, height // 2),
            fontface,
            0.8,
            color,
            thickness=1,
            lineType=linetype)
    return frame


if __name__ == '__main__':
    width, height = 1920, 1080
    frame = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    frame = ObiWan(frame, (width // 2, height // 2))
    cv2.imshow("Jedi", frame)
    cv2.waitKey()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Dec 01 2020, 13:51 [CST]

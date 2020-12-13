# Assignment 1

## 作业描述

编程生成一个视频, 按一下空格, 视频暂停, 再按一下则继续. 程序运行结束后会把视频
文件自动保存到一个指定文件名的视频文件中.

## 使用方法

文件夹下的 `submit.mp4` 是生成的视频文件, 使用 `python3` 运行 `main.py` 可以再
次生成并播放视频:

```bash
$ cd pa1
$ # 不指定参数时输出视频文件为 Mustafar.mp4
$ python3 main.py
$ # 或指定输出文件名:
$ python3 main.py output # 将保存输出到 output.mp4
```

## 测试环境

OS: Arch Linux (5.9.13-zen1-1-zen)
python: 3.9.0
opencv: 4.5.0
numpy: 1.19.4

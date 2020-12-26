#include "pa2.hpp"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
    /* [Variables] */

    // Path to input file
    std::string ifile{""};
    // Treat input file as a video by default
    bool isimage{false};
    // Last pressed key
    char key{0};
    // Image
    cv::Mat img;
    // Frame rate
    int elapse;
    // If video is paused
    bool paused = false;
    // Object to store marked image
    cv::Mat detected;

    /* [/Variables] */
    /****************/
    /* [Parse args] */
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--image")) {
            isimage = true;
        }
        ifile = argv[i];
    }
    if (ifile.length() == 0) {
        fprintf(stderr, "No input file/device specified.\n");
        return 1;
    }
    /* [/Parse args] */

    if (isimage) {
        img      = cv::imread(ifile, cv::IMREAD_COLOR);
        detected = pa2::harris(img, 1);
        cv::imshow("Harris", detected);
        while (key != 'q') {
            key = cv::waitKey();
        }
    } else {
        cv::VideoCapture cap(ifile);

        if (!cap.isOpened()) {
            fprintf(stderr, "Failed to read video file '%s'\n",
                    ifile.c_str());
            return 1;
        }
        elapse = 1000 / cap.get(cv::CAP_PROP_FPS);
        while (cap.isOpened() && key != 'q') {
            if (key == ' ') {
                paused = !paused;
                if (paused) {
                    detected = pa2::harris(img, 1);
                    // directory `img/` should be created in function
                    // `pa2::harris()`
                    cv::imwrite("img/detected.png", detected);
                }
            }
            if (paused) {
                cv::imshow("Harris", detected);
                key = cv::waitKey();
            } else {
                if (!cap.read(img)) {
                    break;
                }
                cv::imshow("Harris", img);
                key = cv::waitKey(elapse);
            }
        }
        cap.release();
    }

    return 0;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Dec 24 2020, 09:29 [CST]

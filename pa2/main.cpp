#include "pa2.hpp"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <string>

int main(int argc, char **argv) {
    /* [Variables] */
    std::string ifile("");
    /* [/Variables] */
    /****************/
    /* [Parse args] */
    for (int i = 1; i < argc; ++i) {
        ifile = argv[i];
    }
    if (ifile.length() == 0) {
        fprintf(stderr, "No input file/device specified.\n");
        return 1;
    }
    /* [/Parse args] */

    cv::Mat img = cv::imread(ifile, cv::IMREAD_GRAYSCALE);

    pa2::harris(img);

    return 0;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Dec 24 2020, 09:29 [CST]

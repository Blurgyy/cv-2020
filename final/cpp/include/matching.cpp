#include "matching.hpp"

cv::Mat SAD(cv::Mat const &limg, cv::Mat const &rimg) {
    if (limg.rows != rimg.rows || //
        limg.cols != rimg.cols) {
        eprintf("Two input images has different sizes\n");
    }
    cv::Mat depth(limg.rows, limg.cols, CV_32FC1);

    return depth;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 17:21 [CST]

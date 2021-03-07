#include "matching.hpp"

#include <cmath>

#include <omp.h>

cv::Mat SAD(cv::Mat const &limg, cv::Mat const &rimg, int const &wr,
            flt const &fx, flt const &baseline) {
    if (limg.rows != rimg.rows || //
        limg.cols != rimg.cols) {
        eprintf("Two input images has different sizes\n");
    }
    int     rows = limg.rows;
    int     cols = limg.cols;
    cv::Mat depth(rows, cols, CV_32FC1, -1);

    flt maxd = std::numeric_limits<flt>::lowest();
    flt mind = std::numeric_limits<flt>::max();

    /* For every pixel on the left image .. */
#pragma omp parallel for
    for (int y = wr; y < rows - wr; ++y) {
        for (int x = wr; x < cols - wr; ++x) {
            /* Find the corresponding window that has minimal difference with
             * it on the right image.
             */
            uint32_t min_diff = std::numeric_limits<uint32_t>::max();
            int      pos      = -1;
            /* Iterate through the same row */
            for (int rx = wr; rx < cols - wr; ++rx) {
                uint32_t cur_diff = std::numeric_limits<uint32_t>::max();
                for (int i = -wr; i < wr; ++i) {
                    // cv::Vec3b lcol = limg.at<cv::Vec3b>(y, x);
                    for (int j = -wr; j < wr; ++j) {
                        cv::Vec3b lcol = limg.at<cv::Vec3b>(y + j, x + i);
                        cv::Vec3b rcol = rimg.at<cv::Vec3b>(y + j, rx + i);
                        cur_diff += std::abs(static_cast<int>(lcol[0]) -
                                             static_cast<int>(rcol[0]));
                        cur_diff += std::abs(static_cast<int>(lcol[1]) -
                                             static_cast<int>(rcol[1]));
                        cur_diff += std::abs(static_cast<int>(lcol[2]) -
                                             static_cast<int>(rcol[2]));
                    }
                }
                if (min_diff > cur_diff) {
                    min_diff = cur_diff;
                    pos      = rx;
                }
            }
            if (pos == x) {
                continue;
            }
            // vprintf("disparity = %d\n", std::abs(pos - x));
            flt d                 = fx * baseline / std::abs(pos - x);
            depth.at<float>(y, x) = d;
            maxd                  = std::max(maxd, d);
            mind                  = std::min(mind, d);
        }
    }

    vprintf("maxd is %f, mind is %f\n", maxd, mind);

    return depth;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 17:21 [CST]

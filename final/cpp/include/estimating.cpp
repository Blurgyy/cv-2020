#include "estimating.hpp"

#include <cmath>

#include <omp.h>

void pose_estimation(std::vector<cv::KeyPoint> const &kp1,
                     std::vector<cv::KeyPoint> const &kp2,
                     std::vector<cv::DMatch> const &matches, mat3 const &K,
                     mat3 &R, vec3 &t) {
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (int i = 0; i < matches.size(); ++i) {
        pts1.push_back(kp1[matches[i].queryIdx].pt);
        pts2.push_back(kp2[matches[i].trainIdx].pt);
    }

    cv::Point2d pp(K[0][2], K[1][2]);
    flt         f = (K[0][0] + K[1][1]) / 2;
    cv::Mat     E = cv::findEssentialMat(pts1, pts2, f, pp, cv::RANSAC);

    cv::Mat RMat, tMat;
    cv::recoverPose(E, pts1, pts2, RMat, tMat, f, pp);
    std::cout << RMat << std::endl;
    std::cout << tMat << std::endl;
    for (int i = 0; i < 3; ++i) {
        t[i] = tMat.at<flt>(i, 0);
        for (int j = 0; j < 3; ++j) {
            R[i][j] = RMat.at<flt>(i, j);
        }
    }
}

cv::Mat SAD(cv::Mat const &limg, cv::Mat const &rimg, int const &wr,
            MiscConf const &conf) {
    if (limg.rows != rimg.rows || //
        limg.cols != rimg.cols) {
        eprintf("Two input images has different sizes\n");
    }
    int     rows = limg.rows;
    int     cols = limg.cols;
    cv::Mat depth(rows, cols, CV_64FC1, -1);

    flt maxd = std::numeric_limits<flt>::lowest();
    flt mind = std::numeric_limits<flt>::max();

    progress p(rows - wr * 2);
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
            // for (int rx = wr; rx < cols - wr; ++rx) {
            for (int d = 0; d < conf.ndisp; ++d) {
                int      rx       = x - d;
                uint32_t cur_diff = 0;
                for (int i = -wr; i < wr; ++i) {
                    // cv::Vec3b lcol = limg.at<cv::Vec3b>(y, x);
                    if (!inrange(x + i, 0, cols) ||
                        !inrange(rx + i, 0, cols)) {
                        continue;
                    }
                    for (int j = -wr; j < wr; ++j) {
                        if (!inrange(y + j, 0, rows)) {
                            continue;
                        }
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
            flt d = conf.left.fx * conf.baseline / std::abs(pos - x);
            depth.at<flt>(y, x) = d;
            maxd                = std::max(maxd, d);
            mind                = std::min(mind, d);
        }
        p.advance();
    }

    return depth;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 17:21 [CST]

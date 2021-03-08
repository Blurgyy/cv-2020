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

cv::Mat SAD(cv::Mat const &left_image, cv::Mat const &right_image,
            int const &wr, MiscConf const &conf) {
    if (left_image.rows != right_image.rows || //
        left_image.cols != right_image.cols) {
        eprintf("Two input images has different sizes\n");
    }
    int     rows = left_image.rows;
    int     cols = left_image.cols;
    cv::Mat disparity(rows, cols, CV_64FC1, -1);

    cv::Mat limg, rimg;
    cv::cvtColor(left_image, limg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image, rimg, cv::COLOR_BGR2GRAY);

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
                        int lcol = limg.at<uint8_t>(y + j, x + i);
                        int rcol = rimg.at<uint8_t>(y + j, rx + i);
                        cur_diff += std::abs(lcol - rcol);
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
            flt d                   = x - pos;
            disparity.at<flt>(y, x) = d;
            maxd                    = std::max(maxd, d);
            mind                    = std::min(mind, d);
        }
        p.advance();
    }

    return disparity;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 17:21 [CST]

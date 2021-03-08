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
    cv::Mat disparity(rows, cols, CV_64FC1);
    disparity = 0;

    cv::Mat limg, rimg;
    cv::cvtColor(left_image, limg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image, rimg, cv::COLOR_BGR2GRAY);

    progress p(rows - wr * 2, "SAD");
    /* For every pixel on the left image .. */
#pragma omp parallel for
    for (int y = wr; y < rows - wr; ++y) {
        for (int x = wr + conf.ndisp; x < cols - wr; ++x) {
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
                    for (int j = -wr; j < wr; ++j) {
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
            // vprintf("disparity = %d\n", std::abs(pos - x));
            flt d                   = std::abs(x - pos);
            disparity.at<flt>(y, x) = d;
        }
        p.advance();
    }

    return disparity;
}

cv::Mat NCC(cv::Mat const &left_image, cv::Mat const &right_image,
            int const &wr, MiscConf const &conf) {
    if (left_image.rows != right_image.rows || //
        left_image.cols != right_image.cols) {
        eprintf("Two input images has different sizes\n");
    }
    int     rows = left_image.rows;
    int     cols = left_image.cols;
    cv::Mat disparity(rows, cols, CV_64FC1);
    disparity = 0;

    cv::Mat limg, rimg;
    cv::cvtColor(left_image, limg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image, rimg, cv::COLOR_BGR2GRAY);

    progress p(rows - wr * 2, "NCC");
#pragma omp parallel for
    for (int y = wr; y < rows - wr; ++y) {
        for (int x = wr + conf.ndisp; x < cols - wr; ++x) {
            int pos      = -1;
            flt max_corr = std::numeric_limits<flt>::lowest();

            for (int d = 0; d < conf.ndisp; ++d) {
                int rx = x - d;

                flt lavg = 0;
                flt ravg = 0;
                for (int i = -wr; i < wr; ++i) {
                    for (int j = -wr; j < wr; ++j) {
                        lavg += limg.at<uint8_t>(y + j, x + i);
                        ravg += rimg.at<uint8_t>(y + j, rx + i);
                    }
                }
                lavg /= sq(2 * wr + 1);
                ravg /= sq(2 * wr + 1);

                flt cur_corr = 0;
                flt lstd     = 0;
                flt rstd     = 0;
                for (int i = -wr; i < wr; ++i) {
                    for (int j = -wr; j < wr; ++j) {
                        flt lcol = limg.at<uint8_t>(y + j, x + i);
                        flt rcol = rimg.at<uint8_t>(y + j, rx + i);
                        cur_corr += (lcol - lavg) * (rcol - ravg);
                        lstd += sq(lcol - lavg);
                        rstd += sq(rcol - ravg);
                    }
                }
                cur_corr /= std::sqrt(lstd * rstd);

                if (max_corr < cur_corr) {
                    max_corr = cur_corr;
                    pos      = rx;
                }
            }
            // vprintf("disparity = %d\n", std::abs(pos - x));
            flt d                   = std::abs(x - pos);
            disparity.at<flt>(y, x) = d;
        }
        p.advance();
    }

    return disparity;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 17:21 [CST]

#include "GCoptimization.h"
#include "estimating.hpp"

#include <cmath>

#include <omp.h>

void pose_estimation(std::vector<cv::KeyPoint> const &kp1,
                     std::vector<cv::KeyPoint> const &kp2,
                     std::vector<cv::DMatch> const &matches, mat3 const &K,
                     mat3 &R, vec3 &t) {
    // eprintf("%d matches found\n", matches.size());
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (int i = 0; i < matches.size(); ++i) {
        pts1.push_back(kp1[matches[i].queryIdx].pt);
        pts2.push_back(kp2[matches[i].trainIdx].pt);
    }

    cv::Point2d pp(K[0][2], K[1][2]);
    flt         f = (K[0][0] + K[1][1]) / 2;
    /* Use RANSAC threshold as 0.1 for better results.
     * Reference: https://stackoverflow.com/a/48394798/13482274
     *
     * From function documentation in opencv2/calib3d.hpp:
     *> @param threshold Parameter used for RANSAC. It is the maximum distance
     *> from a point to an epipolar line in pixels, beyond which the point is
     *> considered an outlier and is not used for computing the final
     *> fundamental matrix. It can be set to something like 1-3, depending on
     *> the accuracy of the point localization, image resolution, and the
     *> image noise.
     */
    cv::Mat E =
        cv::findEssentialMat(pts1, pts2, f, pp, cv::RANSAC, 0.99, 0.1);

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

cv::Mat global_optimization(cv::Mat const &data, MiscConf const &conf,
                            int const &max_iter) {
    if (data.type() != CV_32SC1) {
        eprintf("Expected disparity map type is CV_32SC1 (%d), got %d\n",
                data.type());
    }
    int       rows              = data.rows;
    int       cols              = data.cols;
    int       n_labels          = conf.ndisp == 0 ? cols : conf.ndisp;
    int const default_data_cost = 10;

    try {
        vprintf("Initializing graph ..\n");
        GCoptimizationGridGraph *graph =
            new GCoptimizationGridGraph(cols, rows, n_labels);
        graph->setVerbosity(1);

        /* Set data cost */
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                int idx = y * cols + x;
                for (int l = 0; l < n_labels; ++l) {
                    if (l == data.at<int>(y, x)) {
                        graph->setDataCost(idx, l, 0);
                    } else {
                        graph->setDataCost(idx, l, default_data_cost);
                    }
                }
            }
        }
        /* Set smoothness cost */
        for (int l0 = 0; l0 < n_labels; ++l0) {
            for (int l1 = 0; l1 < n_labels; ++l1) {
                // graph->setSmoothCost(l0, l1, std::min(sq(l0 - l1), 4));
                /* Pott's model */
                int cost = 15 * (l0 != l1);
                graph->setSmoothCost(l0, l1, cost);
            }
        }

        vprintf("Initial energy in graph is %d, starting optimization via "
                "graph cuts ..\n",
                graph->compute_energy());
        graph->expansion(max_iter);
        vprintf("Done, energy after convergence is %d\n",
                graph->compute_energy());

        cv::Mat ret(rows, cols, CV_32SC1);
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                ret.at<int>(y, x) = graph->whatLabel(y * cols + x);
            }
        }
        return ret;
    } catch (GCException e) {
        e.Report();
        eprintf("Error encountered\n");
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
    cv::Mat disparity(rows, cols, CV_32SC1);
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
            int ndisp = conf.ndisp == 0 ? cols : conf.ndisp;
            for (int d = 0; d < ndisp; ++d) {
                int rx = x - d;
                if (rx < 0) {
                    break;
                }

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
            disparity.at<int>(y, x) = d;
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
    cv::Mat disparity(rows, cols, CV_32SC1);
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

            int ndisp = conf.ndisp == 0 ? cols : conf.ndisp;
            for (int d = 0; d < ndisp; ++d) {
                int rx = x - d;
                if (rx < 0) {
                    break;
                }

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
            disparity.at<int>(y, x) = d;
        }
        p.advance();
    }

    return disparity;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 17:21 [CST]

#include "pa2.hpp"

#include <omp.h>

#include <cmath>
#include <filesystem>

namespace pa2 {

// Constants
cv::Scalar const marker_color{20, 89, 200};

cv::Mat harris(cv::Mat const &frame, size_t const &wr) {
    cv::Mat img = frame.clone();
    cv::Mat ret = frame.clone();
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    if (img.channels() == 1) {
        img.convertTo(img, CV_64FC1);
    }
    assert(img.type() == CV_64FC1);

    size_t ws   = wr * 2 + 1;
    size_t rows = img.rows, row_ub = rows - ws;
    size_t cols = img.cols, col_ub = cols - ws;

    // Windows
    cv::Rect win_global  = cv::Rect{0, 0, static_cast<int>(cols - 1),
                                   static_cast<int>(rows - 1)};
    cv::Rect xwin_global = cv::Rect{1, 0, static_cast<int>(cols - 1),
                                    static_cast<int>(rows - 1)};
    cv::Rect ywin_global = cv::Rect{0, 1, static_cast<int>(cols - 1),
                                    static_cast<int>(rows - 1)};
    // Gradient images
    cv::Mat Ixmat = img(xwin_global) - img(win_global);
    cv::Mat Iymat = img(ywin_global) - img(win_global);
    // Precomputes
    cv::Mat Ixxmat = Ixmat.mul(Ixmat);
    cv::Mat Ixymat = Ixmat.mul(Iymat);
    cv::Mat Iyymat = Iymat.mul(Iymat);

    cv::Mat pIxxmat, pIxymat, pIyymat;
    cv::integral(Ixxmat, pIxxmat);
    cv::integral(Ixymat, pIxymat);
    cv::integral(Iyymat, pIyymat);

    // Minimum eigen value of each window
    cv::Mat eigenmin =
        cv::Mat(static_cast<int>(rows), static_cast<int>(cols),
                CV_64FC1); // Maximum eigen value of each window
    cv::Mat eigenmax =
        cv::Mat(static_cast<int>(rows), static_cast<int>(cols), CV_64FC1);

    printf("Computing max/min eigenvalues in each window..\n");
    // Iterate each window in image, `i` for y-direction (rows), `j` for
    // x-direction (cols).
    for (size_t i = 0; i < row_ub; ++i) {
#pragma omp parallel for
        for (size_t j = 0; j < col_ub; ++j) {
            // Generate covariance matrix
            double Ixx{getsum(pIxxmat, i, j, i + ws, j + ws)};
            double Ixy{getsum(pIxymat, i, j, i + ws, j + ws)};
            double Iyy{getsum(pIyymat, i, j, i + ws, j + ws)};
            // Compute eigenvalues of covariace matrix
            auto [mine, maxe] = eigen(Ixx, Ixy, Iyy);
            // Assign eigenvalues to corresponding matrices
            eigenmin.at<double>(i, j) = mine;
            eigenmax.at<double>(i, j) = maxe;
        }
    }

    printf("Saving max/min eigenvalues ..\n");
    // Create directory `img` if it does not exist.
    if (!std::filesystem::exists("img")) {
        std::filesystem::create_directory("img");
    }
    // Save max/min eigenvalue images to file, before non-maximum suppression.
    cv::imwrite("img/eigenmax.png", eigenmax);
    cv::imwrite("img/eigenmin.png", eigenmin);

    // Non-maximum suppression
    eigenmin = nms(eigenmin);
    // // No need of NMS for eigenmax
    // eigenmax = nms(eigenmax);

    printf("Drawing markers at detected corners ..\n");
    // Draw markers at detected corners
    for (size_t i = 0; i < rows; ++i) {
#pragma omp parallel for
        for (size_t j = 0; j < cols; ++j) {
            if (eigenmin.at<unsigned char>(i, j) == 255) {
                cv::circle(ret, cv::Point2i(j, i), wr * 10, marker_color);
            }
        }
    }

    return ret;
}

std::tuple<double, double> eigen(double const &a, double const &b,
                                 double const &c) {
    double A = a + c;
    double B = std::sqrt(sq(a - c) + sq(b + b));
    // Return {minimum eigenvalue, maximum eigenvalue}.
    return {.5 * (A - B), .5 * (A + B)};
}

double getsum(cv::Mat const &from, size_t const &rlb, size_t const &clb,
              size_t const &rub, size_t const &cub) {
    assert(from.type() == CV_64FC1);
    if (rlb < rub && clb < cub) {
        double L   = from.at<double>(rub, cub) - from.at<double>(rlb, clb);
        double hor = from.at<double>(rub, cub) - from.at<double>(rlb, cub);
        double ver = from.at<double>(rub, cub) - from.at<double>(rub, clb);
        return hor + ver - L;
    } else {
        return 0;
    }
}

cv::Mat nms(cv::Mat const &frame) {
    assert(frame.type() == CV_64FC1);

    int rows = frame.rows;
    int cols = frame.cols;
    int size = rows * cols;

    cv::Mat ret(frame.rows, frame.cols, CV_8UC1);

    double median    = 0;
    double threshold = -1;

    std::vector<double> array;
    for (int i = 0; i < frame.rows; ++i) {
        auto x = frame.ptr<double>(i);
        for (int j = 0; j < frame.cols; ++j) {
            array.push_back(x[j]);
        }
    }
    int reserved = std::max(30, (int)(.0001 * size));
    int offset   = size - reserved;
    std::nth_element(array.begin(), array.begin() + offset, array.end());
    median    = array[offset];
    threshold = median;

    // Thresholding.
    for (int i = 0; i < ret.rows; ++i) {
#pragma omp parallel for
        for (int j = 0; j < ret.cols; ++j) {
            if (frame.at<double>(i, j) > threshold) {
                ret.at<unsigned char>(i, j) = 255;
            } else {
                ret.at<unsigned char>(i, j) = 0;
            }
        }
    }

    return ret;
}

} // namespace pa2

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Dec 24 2020, 09:31 [CST]

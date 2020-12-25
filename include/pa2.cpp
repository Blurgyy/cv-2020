#include "pa2.hpp"

#include <cmath>

#include <filesystem>

namespace pa2 {

// Constants
cv::Scalar const marker_color{254, 3, 151};

cv::Mat harris(cv::Mat const &frame, size_t const &wr) {
    printf("Harris ..\n");
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

    // Minimum eigen value of each window
    cv::Mat eigenmin =
        cv::Mat(static_cast<int>(rows), static_cast<int>(cols), CV_64FC1);
    // Maximum eigen value of each window
    cv::Mat eigenmax =
        cv::Mat(static_cast<int>(rows), static_cast<int>(cols), CV_64FC1);

    // Windows
    cv::Rect win_global  = cv::Rect{0, 0, static_cast<int>(cols - 1),
                                   static_cast<int>(rows - 1)};
    cv::Rect xwin_global = cv::Rect{1, 0, static_cast<int>(cols - 1),
                                    static_cast<int>(rows - 1)};
    cv::Rect ywin_global = cv::Rect{0, 1, static_cast<int>(cols - 1),
                                    static_cast<int>(rows - 1)};
    // Gradient images
    cv::Mat Ix = img(xwin_global) - img(win_global);
    cv::Mat Iy = img(ywin_global) - img(win_global);
    // Precomputes
    cv::Mat Ixxmat = Ix.mul(Ix);
    cv::Mat Ixymat = Ix.mul(Iy);
    cv::Mat Iyymat = Iy.mul(Iy);
    cv::Mat pIxxmat, pIxymat, pIyymat;
    cv::integral(Ixxmat, pIxxmat);
    cv::integral(Ixymat, pIxymat);
    cv::integral(Iyymat, pIyymat);

    // Iterate each window in image, `i` for y-direction (rows), `j` for
    // x-direction (cols).
    for (size_t i = 0; i < row_ub; ++i) {
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

    // Non-maximum suppression with slightly larger window radius
    eigenmin = nms(eigenmin);
    eigenmax = nms(eigenmax);

    // Create directory `img` if it does not exist
    std::filesystem::create_directory("img");
    // Save max/min eigenvalue images to file
    cv::imwrite("img/eigenmax.png", eigenmax);
    cv::imwrite("img/eigenmin.png", eigenmin);

    // Draw markers at detected corners
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (eigenmin.at<unsigned char>(i, j) >= 128) {
                printf("Drawing circle at (%zu, %zu)\n", i, j);
                break;
                cv::circle(ret, cv::Point2i(i, j), wr * 10, marker_color);
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
    printf("NMS ..\n");

    cv::Mat ret = frame.clone();
    cv::normalize(ret, ret, 255.5, 0, cv::NORM_MINMAX);
    unsigned char th = 127;
    cv::threshold(ret, ret, th, 255, cv::THRESH_BINARY);

    cv::imshow("Harris", ret);
    cv::waitKey();

    return ret;

    // cv::Mat inter, ret;

    // cv::dilate(frame, inter, cv::Mat());
    // cv::compare(frame, inter, inter, cv::CMP_GE);

    // cv::erode(frame, ret, cv::Mat());
    // cv::compare(frame, ret, ret, cv::CMP_GT);
    // cv::bitwise_and(inter, ret, ret);

    // return ret;

    // size_t  rows = frame.rows;
    // size_t  cols = frame.cols;
    // size_t  ws   = std::min(rows, cols) / 50;
    // cv::Mat ret(rows, cols, CV_8UC1);

    // for (size_t x = 0; x < cols; x += ws) {
    // for (size_t y = 0; y < rows; y += ws) {
    // // Maximum value in current window
    // double maxx = -std::numeric_limits<double>::infinity();
    // for (size_t i = y; i < std::min(rows, y + ws); ++i) {
    // for (size_t j = x; j < std::min(cols, x + ws); ++j) {
    // maxx = std::max(maxx, frame.at<double>(i, j));
    // }
    // }
    // // Suppress all non-maximum values
    // for (size_t i = y; i < std::min(rows, y + ws); ++i) {
    // for (size_t j = x; j < std::min(rows, x + ws); ++j) {
    // // printf("value is %f, maxx is %f\n",
    // // frame.at<double>(i, j), maxx);
    // if (frame.at<double>(i, j) == maxx) {
    // ret.at<unsigned char>(i, j) = 255;
    // } else {
    // ret.at<unsigned char>(i, j) = 0;
    // }
    // }
    // }
    // }
    // }

    // cv::imshow("Harris", ret);
    // cv::waitKey();

    return ret;
}

} // namespace pa2

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Dec 24 2020, 09:31 [CST]

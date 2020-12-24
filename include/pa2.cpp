#include "pa2.hpp"

#include <cmath>

namespace pa2 {

void harris_naive(cv::Mat const &frame, size_t const &wr) {
    cv::Mat img = frame.clone();
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

    // double sum = 0;
    // int    ilb = 1, iub = 50;
    // int    jlb = 1, jub = 50;
    // for (int i = ilb; i < iub; ++i) {
    // for (int j = jlb; j < jub; ++j) {
    // sum += Ixxmat.at<double>(i, j);
    // }
    // }
    // printf("%f \\sim %f\n", sum, getsum(pIxxmat, ilb, jlb, iub, jub));
    // exit(0);

    // Iterate each window in image, `i` for y-direction (rows), `j` for
    // x-direction (cols).
    for (size_t i = 0; i < row_ub; ++i) {
        for (size_t j = 0; j < col_ub; ++j) {
            // Initialize windows
            cv::Rect win =
                cv::Rect(static_cast<int>(j), static_cast<int>(i),
                         static_cast<int>(ws), static_cast<int>(ws));
            // Generate covariance matrix
            // cv::Mat winIx = Ix(win);
            // cv::Mat winIy = Iy(win);
            double Ixx{getsum(pIxxmat, i, j, i + ws, j + ws)};
            double Ixy{getsum(pIxymat, i, j, i + ws, j + ws)};
            double Iyy{getsum(pIyymat, i, j, i + ws, j + ws)};
            /* double  Ixx{cv::sum(Ix.mul(Ix))[0]}; */
            /* double  Ixy{cv::sum(Ix.mul(Iy))[0]}; */
            /* double  Iyy{cv::sum(Iy.mul(Iy))[0]}; */
            // Compute eigenvalues of covariace matrix
            auto [mine, maxe] = eigen(Ixx, Ixy, Iyy);
            // printf("(%lu, %lu)\n", i, j);
            eigenmin.at<double>(i, j) = mine;
            eigenmax.at<double>(i, j) = maxe;
        }
        // printf("Row %zu is processed\n", i);
    }
    printf("Done\n");
    cv::imshow("max", eigenmax);
    cv::waitKey();
    cv::imshow("min", eigenmin);
    cv::waitKey();
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
    // printf("row-dir: [%zu, %zu)\n", rlb, rub);
    // printf("col-dir: [%zu, %zu)\n", clb, cub);
    if (rlb < rub && clb < cub) {
        double L   = from.at<double>(rub, cub) - from.at<double>(rlb, clb);
        double hor = from.at<double>(rub, cub) - from.at<double>(rlb, cub);
        double ver = from.at<double>(rub, cub) - from.at<double>(rub, clb);
        return hor + ver - L;
    } else {
        return 0;
    }
}

} // namespace pa2

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Dec 24 2020, 09:31 [CST]

#include <opencv2/opencv.hpp>

#include <tuple>

namespace pa2 {

// Naive Harris corner detector (super slow)
void harris_naive(cv::Mat const &frame, size_t const &wr = 1);

// Compute eigenvalues of given symmetric matrix [a, b; b, c].
// @return A tuple with {minEvalue, maxEvalue}.
std::tuple<double, double> eigen(double const &a, double const &b,
                                 double const &c);

// @param from: Integral image.
// @param rlb, clb: Lower-bound in row/column direction (inclusive)
// @param rub, cub: Upper-bound in row/column direction (exclusive)
double getsum(cv::Mat const &from, size_t const &rlb, size_t const &clb,
              size_t const &rub, size_t const &cub);

// @return square of `x`
template <typename T> T constexpr sq(T const &x) { return x * x; }

}; // namespace pa2

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Dec 24 2020, 09:31 [CST]

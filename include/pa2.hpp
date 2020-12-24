#include <opencv2/opencv.hpp>

#include <tuple>

namespace pa2 {

// @brief Harris corner detector
void harris(cv::Mat const &frame, size_t const &wr = 1);

// @brief Compute eigenvalues of given symmetric matrix [a, b; b, c].
// @return A tuple with {minEvalue, maxEvalue}.
std::tuple<double, double> eigen(double const &a, double const &b,
                                 double const &c);

// @brief Get sum of given block range.
// @param from: Integral image.
// @param rlb, clb: Lower-bound in row/column direction (inclusive)
// @param rub, cub: Upper-bound in row/column direction (exclusive)
// @return Value of sum of block specified by (rlb, clb) and (rub, cub).
double getsum(cv::Mat const &from, size_t const &rlb, size_t const &clb,
              size_t const &rub, size_t const &cub);

// @return square of `x`
template <typename T> T constexpr sq(T const &x) { return x * x; }

// @brief Perfrom non-maximum suppression
cv::Mat nms(cv::Mat &frame, size_t const &wr = 2);

}; // namespace pa2

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Dec 24 2020, 09:31 [CST]

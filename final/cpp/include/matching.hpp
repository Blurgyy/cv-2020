#pragma once

#include "globla.hpp"

/* Sum of absolute difference (SAD)
 * @return Depth map.
 */
cv::Mat SAD(cv::Mat const &limg, cv::Mat const &rimg);

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 17:20 [CST]

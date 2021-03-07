#include "estimating.hpp"
#include "geometry.hpp"
#include "globla.hpp"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

void Usage() { fprintf(stderr, "Usage:\n"); }

int main(int argc, char **argv) {
    /* [Testing area] */
    // std::cout << std::boolalpha;
    // std::cout << std::is_trivial<CamConf>::value << std::endl;
    // std::cout << std::is_trivial<SpatialPoint>::value << std::endl;
    /* [/Testing area] */

    /* [Variables] */
    cv::Mat limg, rimg;
    /* [/Variables] */

    /* [Parse args] */
    if (argc < 4) {
        Usage();
        return 1;
    }
    limg     = cv::imread(argv[1], cv::IMREAD_COLOR);
    rimg     = cv::imread(argv[2], cv::IMREAD_COLOR);
    int rows = limg.rows;
    int cols = rimg.cols;
    if (rows != rimg.rows || cols != rimg.cols) {
        eprintf("The given 2 stereo images has different sizes\n");
    }
    auto [lconf, rconf] = read_calib(argv[3]);
    dump(lconf);
    dump(rconf);
    /* [/Parse args] */

    std::vector<cv::KeyPoint> kp1, kp2;
    std::vector<cv::DMatch>   matches;
    get_matches(limg, rimg, kp1, kp2, matches);
    // clang-format off
    mat3 K{
        1171, 0, 269.5,
        0, 1171, 479.5,
        0,    0,     1,
    };
    // clang-format on
    mat3 R;
    vec3 t;
    pose_estimation(kp1, kp2, matches, K, R, t);
    printf("e rotation:\n");
    dump(R);
    printf("e translation\n");
    dump(t);

    /* [Stereo rectification] */
    cv::Mat          l_rect, r_rect;
    flt              baseline  = std::numeric_limits<flt>::lowest();
    std::vector<ppp> pixel_map = stereo_rectification(
        limg, rimg, lconf, rconf, l_rect, r_rect, baseline);
    cv::imwrite("l_rect.jpg", l_rect);
    cv::imwrite("r_rect.jpg", r_rect);
    /* [/Stereo rectification] */

    /* [SAD] */
    cv::Mat dep_SAD = SAD(l_rect, r_rect, 3, lconf.fx, baseline);
    dep_SAD         = map_back(pixel_map, rows, cols, dep_SAD);
    cv::imwrite("dep_SAD.jpg", dep_SAD);
    /* [/SAD] */

    return 0;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 25 2021, 15:34 [CST]

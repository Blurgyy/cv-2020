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
    limg                              = cv::imread(argv[1], cv::IMREAD_COLOR);
    rimg                              = cv::imread(argv[2], cv::IMREAD_COLOR);
    std::pair<CamConf, CamConf> confs = read_cam(argv[3]);
    CamConf                     lconf = confs.first;
    CamConf                     rconf = confs.second;
    /* [/Parse args] */

    /* [Spatial transformation] */
    cv::Mat l_rect, r_rect;
    stereo_rectification(limg, rimg, lconf, rconf, l_rect, r_rect);
    /* [/Spatial transformation] */

    cv::imwrite("l_rect.jpg", l_rect);
    cv::imwrite("r_rect.jpg", r_rect);

    return 0;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 25 2021, 15:34 [CST]

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
    std::vector<SpatialPoint> lpts, rpts;
    for (int x = 0; x < limg.cols; ++x) {
        cv::Vec3b *row = limg.ptr<cv::Vec3b>(x);
        for (int y = 0; y < limg.rows; ++y) {
            lpts.push_back({
                {x, y, 1},                         // position
                {row[y][0], row[y][1], row[y][2]}, // color
            });
        }
    }
    for (int x = 0; x < rimg.cols; ++x) {
        cv::Vec3b *row = rimg.ptr<cv::Vec3b>(x);
        for (int y = 0; y < rimg.rows; ++y) {
            rpts.push_back({
                {x, y, 1},                         // position
                {row[y][0], row[y][1], row[y][2]}, // color
            });
        }
    }
    assert(lpts.size() == rpts.size());
    int len = lpts.size();
    for (int i = 0; i < len; ++i) {
        SpatialPoint p = to_left_camera(lconf, rconf, rpts[i]);
        int          x = p.pos[0];
        int          y = p.pos[1];
        if (0 <= x && x < rimg.cols && 0 <= y && y < rimg.rows) {
            rimg.at<cv::Vec3b>(x, y) =
                cv::Vec3b(p.color[0], p.color[1], p.color[2]);
        }
    }
    /* [/Spatial transformation] */

    cv::imwrite("rimg.jpg", rimg);

    return 0;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 25 2021, 15:34 [CST]

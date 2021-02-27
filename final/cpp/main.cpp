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
    for (int y = 0; y < limg.rows; ++y) {
        cv::Vec3b *row = limg.ptr<cv::Vec3b>(y);
        for (int x = 0; x < limg.cols; ++x) {
            lpts.push_back({
                {x, y, 1},                         // position
                {row[x][0], row[x][1], row[x][2]}, // color
            });
        }
    }
    for (int y = 0; y < rimg.rows; ++y) {
        cv::Vec3b *row = rimg.ptr<cv::Vec3b>(y);
        for (int x = 0; x < rimg.cols; ++x) {
            rpts.push_back({
                {x, y, 1},                         // position
                {row[x][0], row[x][1], row[x][2]}, // color
            });
        }
    }
    assert(lpts.size() == rpts.size());
    int     len = lpts.size();
    cv::Mat out(rimg.rows, rimg.cols, rimg.type());
    for (int i = 0; i < len; ++i) {
        CamConf repconf = get_reprojection_conf(rconf, lconf);
        vec3    ncoord  = to_camera_space(rconf, rpts[i]).pos * repconf.rot;
        SpatialPoint p  = {ncoord, rpts[i].color};
        p               = to_image_space(rconf, p);
        int x           = std::round(p.pos[0]);
        int y           = std::round(p.pos[1]);
        if (0 <= x && x < out.cols && //
            0 <= y && y < out.rows) {
            out.at<cv::Vec3b>(y, x) =
                cv::Vec3b(p.color[0], p.color[1], p.color[2]);
        } else {
            vprintf("Out of range\n");
        }
    }
    /* [/Spatial transformation] */

    cv::imwrite("rimg.jpg", out);

    return 0;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 25 2021, 15:34 [CST]

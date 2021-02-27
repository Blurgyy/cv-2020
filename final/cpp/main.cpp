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
    cv::Mat inter_right(rimg.rows, rimg.cols, rimg.type());
    cv::Mat l_rect(limg.rows, limg.cols, limg.type());
    cv::Mat r_rect(rimg.rows, rimg.cols, rimg.type());
    CamConf repconf = get_reprojection_conf(rconf, lconf);
    /* 1. Rotate right image to be parallel with left image */
    for (int i = 0; i < len; ++i) {
        vec3 ncoord    = to_camera_space(rconf, rpts[i]).pos * repconf.rot;
        SpatialPoint p = {ncoord, rpts[i].color};
        p              = to_image_space(rconf, p);
        int x          = std::round(p.pos[0]);
        int y          = std::round(p.pos[1]);
        if (0 <= x && x < inter_right.cols && //
            0 <= y && y < inter_right.rows) {
            inter_right.at<cv::Vec3b>(y, x) =
                cv::Vec3b(p.color[0], p.color[1], p.color[2]);
        } else {
            vprintf("Out of range\n");
        }
    }
    rpts.clear();
    for (int y = 0; y < inter_right.rows; ++y) {
        cv::Vec3b *row = inter_right.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inter_right.cols; ++x) {
            rpts.push_back({
                {x, y, 1},                         // position
                {row[x][0], row[x][1], row[x][2]}, // color
            });
        }
    }
    /* 2. Rotate both images by R_{rect} */
    vec3 row1 = glm::normalize(repconf.trans);
    vec3 row2 = vec3{-row1.y, row1.x, 0} / std::sqrt(sq(row1.x) + sq(row1.y));
    vec3 row3 = glm::cross(row1, row2);
    // clang-format off
    mat3 R_rect{
        row1.x, row1.y, row1.z,
        row2.x, row2.y, row2.z,
        row3.x, row3.y, row3.z,
    };
    // clang-format on
    /* 2.1 rotate left image plane */
    for (int i = 0; i < len; ++i) {
        vec3         lcoord = to_camera_space(lconf, lpts[i]).pos * R_rect;
        SpatialPoint lp     = {lcoord, lpts[i].color};
        lp                  = to_image_space(lconf, lp);
        int lx              = std::round(lp.pos[0]);
        int ly              = std::round(lp.pos[1]);
        if (0 <= lx && lx < l_rect.cols && //
            0 <= ly && ly < l_rect.rows) {
            l_rect.at<cv::Vec3b>(ly, lx) =
                cv::Vec3b(lp.color[0], lp.color[1], lp.color[2]);
        } else {
            vprintf("Out of range\n");
        }
        vec3         rcoord = to_camera_space(rconf, rpts[i]).pos * R_rect;
        SpatialPoint rp     = {rcoord, rpts[i].color};
        rp                  = to_image_space(rconf, rp);
        int rx              = std::round(rp.pos[0]);
        int ry              = std::round(rp.pos[1]);
        if (0 <= rx && rx < r_rect.cols && //
            0 <= ry && ry < r_rect.rows) {
            r_rect.at<cv::Vec3b>(ry, rx) =
                cv::Vec3b(rp.color[0], rp.color[1], rp.color[2]);
        } else {
            vprintf("Out of range\n");
        }
    }
    /* [/Spatial transformation] */

    cv::imwrite("inter_right.jpg", inter_right);
    cv::imwrite("l_rect.jpg", l_rect);
    cv::imwrite("r_rect.jpg", r_rect);

    return 0;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 25 2021, 15:34 [CST]

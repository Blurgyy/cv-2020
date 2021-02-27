#pragma once

using flt = double;

#include <boost/format.hpp>
using format = boost::format;

#include <glm/glm.hpp>
using vec3 = glm::vec<3, flt, glm::defaultp>;
using mat3 = glm::mat<3, 3, flt, glm::defaultp>;

#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>

// messaging functions `dprintf` (debug messages), `eprintf` (error messages)
// Reference: https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html
#ifndef NDEBUG
#define DEBUGGING -1
#else
#define DEBUGGING 0
#endif
#define dprintf(fmt, ...)                                                    \
    do {                                                                     \
        if (DEBUGGING)                                                       \
            fprintf(stdout, " [*] %s::%d::%s(): " fmt, __FILE__, __LINE__,   \
                    __func__, ##__VA_ARGS__);                                \
    } while (0)
#define eprintf(fmt, ...)                                                    \
    do {                                                                     \
        if (DEBUGGING)                                                       \
            fprintf(stderr, " [X] %s::%d::%s(): " fmt, __FILE__, __LINE__,   \
                    __func__, ##__VA_ARGS__);                                \
        else                                                                 \
            fprintf(stderr, " [X] " fmt, ##__VA_ARGS__);                     \
        exit(-1);                                                            \
    } while (0)
#define vprintf(fmt, ...)                                                    \
    do {                                                                     \
        fprintf(stderr, " [v] " fmt, ##__VA_ARGS__);                         \
    } while (0)

/* Structs */
struct CamConf {
    flt  fx, fy;
    flt  cx, cy;
    mat3 rot;
    vec3 trans;
};
struct SpatialPoint {
    vec3 pos;
    vec3 color;
};

/* Struct helper functions */
inline void dump(CamConf const &x) {
    printf("fx = %f, fy = %f\n", x.fx, x.fy);
    printf("cx = %f, cy = %f\n", x.cx, x.cy);
    printf("rotation:\n");
    for (int i = 0; i < 3; ++i) {
        printf("%f", x.rot[i][0]);
        for (int j = 1; j < 3; ++j) {
            printf(" %f", x.rot[i][j]);
        }
        printf("\n");
    }
    printf("translation:\n");
    printf("%f %f %f\n", x.trans[0], x.trans[1], x.trans[2]);
}
inline void dump(SpatialPoint const &x) {
    printf("position: %f %f %f\n", x.pos[0], x.pos[1], x.pos[2]);
    printf("color:    %f %f %f\n", x.color[0], x.color[1], x.color[2]);
}
inline void dump(vec3 const &x) { printf("%f %f %f\n", x.x, x.y, x.z); }

/* Functions */
template <typename T> T sq(T const &x) { return x * x; }

inline std::pair<CamConf, CamConf> read_cam(std::string const &filename) {
    CamConf       lret, rret;
    std::ifstream from{filename};
    if (from.fail()) {
        eprintf("Failed opening file %s\n", filename.c_str());
    }
    /* Check if reading is successful */
    int six = 0;
    for (std::string line; std::getline(from, line);) {
        std::istringstream in{line};
        std::string        token;
        in >> token;
        if (token == "fx") {
            flt fx;
            in >> fx;
            lret.fx = rret.fx = fx;
            ++six;
        } else if (token == "cx") {
            flt cx;
            in >> cx;
            lret.cx = rret.cx = cx;
            ++six;
        } else if (token == "cy") {
            flt cy;
            in >> cy;
            lret.cy = rret.cy = cy;
            ++six;
        } else if (token == "fy") {
            flt fy;
            in >> fy;
            lret.fy = rret.fy = fy;
            ++six;
        } else if (token == "left" || token == "Left") {
            for (int i = 0; i < 3; ++i) {
                std::getline(from, line);
                in = std::istringstream{line};
                in >> lret.rot[i][0] >> lret.rot[i][1] >> lret.rot[i][2] >>
                    lret.trans[i];
            }
            ++six;
        } else if (token == "right" || token == "Right") {
            for (int i = 0; i < 3; ++i) {
                std::getline(from, line);
                in = std::istringstream{line};
                in >> rret.rot[i][0] >> rret.rot[i][1] >> rret.rot[i][2] >>
                    rret.trans[i];
            }
            ++six;
        }
    }
    if (six != 6) {
        dump(lret);
        dump(rret);
        eprintf("Failed reading camera configs\n");
    }
    return std::make_pair(lret, rret);
}

inline SpatialPoint to_camera_space(CamConf const &     conf,
                                    SpatialPoint const &point) {
    SpatialPoint ret;
    ret.pos = vec3{
        (point.pos.x - conf.cx) / conf.fx,
        (point.pos.y - conf.cy) / conf.fy,
        1,
    };
    ret.color = point.color;
    return ret;
}
inline SpatialPoint to_image_space(CamConf const &     conf,
                                   SpatialPoint const &point) {
    SpatialPoint ret;
    ret.pos = {
        point.pos.x * conf.fx / point.pos.z + conf.cx,
        point.pos.y * conf.fy / point.pos.z + conf.cy,
        1,
    };
    ret.color = point.color;
    return ret;
}

/* Get the reprojection matrix from camera `from` to camera `to`.
 */
inline CamConf get_reprojection_conf(CamConf const &from, CamConf const &to) {
    // dump(from);
    // dump(to);
    // eprintf();
    CamConf ret;
    ret.rot   = glm::transpose(from.rot) * to.rot;
    ret.trans = to.trans - from.trans * glm::transpose(from.rot) * to.rot;
    return ret;
}

/* Reproject a point with given camera configuration.
 * Note: Get reprojection conf with function `get_reprojection_conf`.
 */
inline SpatialPoint reproject(CamConf const &     conf,
                              SpatialPoint const &rpoint) {
    vec3 pos = rpoint.pos * conf.rot + conf.trans;
    // printf("new pos: %f %f %f\n", pos.x, pos.y, pos.z);
    pos.x /= pos.z;
    pos.y /= pos.z;
    return SpatialPoint{pos, rpoint.color};
}

/* Stereo rectification */
inline void stereo_rectification(cv::Mat const &left_image,
                                 cv::Mat const &right_image,
                                 CamConf const &left_camera,
                                 CamConf const &right_camera,
                                 cv::Mat &      rectified_left_image,
                                 cv::Mat &      rectified_right_image) {
    std::vector<SpatialPoint> lpts, rpts;
    for (int y = 0; y < left_image.rows; ++y) {
        cv::Vec3b const *row = left_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < left_image.cols; ++x) {
            lpts.push_back({
                {x, y, 1},                         // position
                {row[x][0], row[x][1], row[x][2]}, // color
            });
        }
    }
    for (int y = 0; y < right_image.rows; ++y) {
        cv::Vec3b const *row = right_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < right_image.cols; ++x) {
            rpts.push_back({
                {x, y, 1},                         // position
                {row[x][0], row[x][1], row[x][2]}, // color
            });
        }
    }
    assert(lpts.size() == rpts.size());
    int len = lpts.size();
    rectified_left_image =
        cv::Mat(left_image.rows, left_image.cols, left_image.type());
    rectified_right_image =
        cv::Mat(right_image.rows, right_image.cols, right_image.type());
    CamConf repconf = get_reprojection_conf(right_camera, left_camera);
    /* 1. Rotate right image to be parallel with left image */
    for (int i = 0; i < len; ++i) {
        vec3 ncoord =
            to_camera_space(right_camera, rpts[i]).pos * repconf.rot;
        SpatialPoint p = {ncoord, rpts[i].color};
        p              = to_image_space(right_camera, p);
        rpts[i]        = p;
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
        vec3 lcoord     = to_camera_space(left_camera, lpts[i]).pos * R_rect;
        SpatialPoint lp = {lcoord, lpts[i].color};
        lp              = to_image_space(left_camera, lp);
        int lx          = std::round(lp.pos[0]);
        int ly          = std::round(lp.pos[1]);
        if (0 <= lx && lx < rectified_left_image.cols && //
            0 <= ly && ly < rectified_left_image.rows) {
            rectified_left_image.at<cv::Vec3b>(ly, lx) =
                cv::Vec3b(lp.color[0], lp.color[1], lp.color[2]);
        }
        vec3 rcoord     = to_camera_space(right_camera, rpts[i]).pos * R_rect;
        SpatialPoint rp = {rcoord, rpts[i].color};
        rp              = to_image_space(right_camera, rp);
        int rx          = std::round(rp.pos[0]);
        int ry          = std::round(rp.pos[1]);
        if (0 <= rx && rx < rectified_right_image.cols && //
            0 <= ry && ry < rectified_right_image.rows) {
            rectified_right_image.at<cv::Vec3b>(ry, rx) =
                cv::Vec3b(rp.color[0], rp.color[1], rp.color[2]);
        }
    }
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Jan 25 2021, 18:43 [CST]

#pragma once

using flt = double;

#include <boost/format.hpp>
using format = boost::format;

#include <glm/glm.hpp>
using vec3 = glm::vec<3, flt, glm::defaultp>;
using mat3 = glm::mat<3, 3, flt, glm::defaultp>;

#include <fstream>
#include <sstream>

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
        fprintf(stdout, " [v] " fmt, ##__VA_ARGS__);                         \
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

/* Functions */
template <typename T> T sq(T const &x) { return x * x; }

inline std::pair<CamConf, CamConf> read_cam(std::string const &filename) {
    CamConf       lret, rret;
    std::ifstream from{filename};
    if (from.fail()) {
        eprintf("Failed opening file %s\n", filename.c_str());
    }
    for (std::string line; std::getline(from, line);) {
        std::istringstream in{line};
        std::string        token;
        in >> token;
        if (token == "fx") {
            flt fx;
            in >> fx;
            lret.fx = rret.fx = fx;
        } else if (token == "cx") {
            flt cx;
            in >> cx;
            lret.cx = rret.cx = cx;
        } else if (token == "cy") {
            flt cy;
            in >> cy;
            lret.cy = rret.cy = cy;
        } else if (token == "fy") {
            flt fy;
            in >> fy;
            lret.fy = rret.fy = fy;
        } else if (token == "left") {
            for (int i = 0; i < 3; ++i) {
                in >> lret.rot[i][0] >> lret.rot[i][1] >> lret.rot[i][2] >>
                    lret.trans[i];
            }
        } else if (token == "right") {
            for (int i = 0; i < 3; ++i) {
                in >> rret.rot[i][0] >> rret.rot[i][1] >> rret.rot[i][2] >>
                    rret.trans[i];
            }
        }
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

inline SpatialPoint to_left_camera(CamConf const &lconf, CamConf const &rconf,
                                   SpatialPoint const &rpoint) {
    vec3 pos =
        (rpoint.pos - rconf.trans) * glm::transpose(rconf.rot) * lconf.rot +
        lconf.trans;
    pos.x /= pos.z;
    pos.y /= pos.z;
    return SpatialPoint{pos, rpoint.color};
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Jan 25 2021, 18:43 [CST]

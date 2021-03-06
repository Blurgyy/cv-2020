#pragma once

using flt = double;

#include <boost/format.hpp>
using format = boost::format;

#include <glm/glm.hpp>
using vec3 = glm::vec<3, flt, glm::defaultp>;
using mat3 = glm::mat<3, 3, flt, glm::defaultp>;

#include <fstream>
#include <sstream>
#include <tuple>

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

inline std::tuple<CamConf, CamConf> read_cam(std::string const &filename) {
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
    return {lret, rret};
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Jan 25 2021, 18:43 [CST]

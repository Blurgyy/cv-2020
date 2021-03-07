#pragma once

using flt = double;

#include <glm/glm.hpp>
using vec3 = glm::vec<3, flt, glm::defaultp>;
using mat3 = glm::mat<3, 3, flt, glm::defaultp>;

#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>

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
struct MiscConf {
    CamConf  left, right;
    flt      doffs;
    flt      baseline;
    uint32_t width, height;
    uint32_t ndisp;
    bool     isint;
    uint32_t vmin, vmax;
    flt      dyavg, dymax;
};
using ppp = std::pair<SpatialPoint, SpatialPoint>;

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
inline void dump(mat3 const &x) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            printf("%f ", x[i][j]);
        }
        printf("\n");
    }
}

/* Functions */
template <typename T> T sq(T const &x) { return x * x; }

std::tuple<CamConf, CamConf> read_cam(std::string const &filename);
MiscConf                     read_calib(std::string const &filename);

cv::Mat map_back(std::vector<ppp> const &pixel_map, int const &rows,
                 int const &cols, cv::Mat const &dep);

void get_matches(cv::Mat const &limg, cv::Mat const &rimg,
                 std::vector<cv::KeyPoint> &kp1,
                 std::vector<cv::KeyPoint> &kp2,
                 std::vector<cv::DMatch> &  matches);

struct progress {
    progress(int const &tot = 1) : now(0), tot(tot) {}
    int  now;
    int  tot;
    void advance() {
        printf("\33[2K");
        ++now;
        vprintf("%g%%\r", 100.0 * now / tot);
    }
};

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Jan 25 2021, 18:43 [CST]

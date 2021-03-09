#include "estimating.hpp"
#include "geometry.hpp"

SpatialPoint to_camera_space(CamConf const &conf, SpatialPoint const &point) {
    SpatialPoint ret;
    ret.pos = vec3{
        (point.pos.x - conf.cx) / conf.fx,
        (point.pos.y - conf.cy) / conf.fy,
        1,
    };
    ret.color = point.color;
    return ret;
}

SpatialPoint to_image_space(CamConf const &conf, SpatialPoint const &point) {
    SpatialPoint ret;
    ret.pos = {
        point.pos.x * conf.fx / point.pos.z + conf.cx,
        point.pos.y * conf.fy / point.pos.z + conf.cy,
        1,
    };
    ret.color = point.color;
    return ret;
}

CamConf get_reprojection_conf(CamConf const &from, CamConf const &to) {
    // dump(from);
    // dump(to);
    // eprintf();
    CamConf ret;
    ret.rot   = glm::transpose(from.rot) * to.rot;
    ret.trans = to.trans - from.trans * glm::transpose(from.rot) * to.rot;
    return ret;
}

std::vector<ppp> stereo_rectification(cv::Mat const &left_image,
                                      cv::Mat const &right_image,
                                      CamConf const &left_camera,
                                      CamConf const &right_camera,
                                      cv::Mat &      rectified_left_image,
                                      cv::Mat &      rectified_right_image) {
    std::vector<SpatialPoint> lpts, rpts;
    std::vector<ppp>          ret;
    /* Generate points on both images' imaging planes */
    for (int y = 0; y < left_image.rows; ++y) {
        cv::Vec3b const *row = left_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < left_image.cols; ++x) {
            lpts.push_back({
                {x + 0.5, y + 0.5, 1},             // position
                {row[x][0], row[x][1], row[x][2]}, // color
            });
        }
    }
    for (int y = 0; y < right_image.rows; ++y) {
        cv::Vec3b const *row = right_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < right_image.cols; ++x) {
            rpts.push_back({
                {x + 0.5, y + 0.5, 1},             // position
                {row[x][0], row[x][1], row[x][2]}, // color
            });
        }
    }
    assert(lpts.size() == rpts.size());

    int len = lpts.size();
    // clang-format off
    mat3 K{
        left_camera.fx, 0, left_camera.cx,
        0, left_camera.fy, left_camera.cy,
        0,              0,              1,
    };
    // clang-format on
    mat3                      R;
    vec3                      t;
    std::vector<cv::KeyPoint> kp1, kp2;
    std::vector<cv::DMatch>   matches;
    get_matches(left_image, right_image, kp1, kp2, matches);
    pose_estimation(kp1, kp2, matches, K, R, t);
    mat3 R2    = glm::transpose(R);
    vec3 trans = -t * glm::transpose(R);

    /* Get rectification matrix */
    vec3 row1 = glm::normalize(trans);
    vec3 row2 = glm::normalize(vec3{-row1.y, row1.x, 0});
    vec3 row3 = glm::normalize(glm::cross(row1, row2));
    // clang-format off
    mat3 R_rect{
        row1.x, row1.y, row1.z,
        row2.x, row2.y, row2.z,
        row3.x, row3.y, row3.z,
    };
    // clang-format on

    R2      = R2 * R_rect;
    mat3 R1 = R_rect;

    flt maxx = std::numeric_limits<flt>::lowest();
    flt maxy = std::numeric_limits<flt>::lowest();
    flt minx = std::numeric_limits<flt>::max();
    flt miny = std::numeric_limits<flt>::max();

    std::vector<SpatialPoint> limgpts, rimgpts;
    for (int i = 0; i < len; ++i) {
        vec3         ncoord = to_camera_space(right_camera, rpts[i]).pos * R2;
        SpatialPoint p      = {ncoord, rpts[i].color};
        p                   = to_image_space(right_camera, p);
        rimgpts.push_back(p);

        ncoord = to_camera_space(left_camera, lpts[i]).pos * R1;
        p      = {ncoord, lpts[i].color};
        p      = to_image_space(left_camera, p);
        limgpts.push_back(p);

        SpatialPoint lp = limgpts[i];
        maxx            = std::max(maxx, lp.pos.x);
        minx            = std::min(minx, lp.pos.x);
        maxy            = std::max(maxy, lp.pos.y);
        miny            = std::min(miny, lp.pos.y);

        SpatialPoint rp = rimgpts[i];
        maxx            = std::max(maxx, rp.pos.x);
        minx            = std::min(minx, rp.pos.x);
        maxy            = std::max(maxy, rp.pos.y);
        miny            = std::min(miny, rp.pos.y);
    }

    int cols              = std::round(maxx) - std::round(minx) + 1;
    int rows              = std::round(maxy) - std::round(miny) + 1;
    rectified_left_image  = cv::Mat(rows, cols, left_image.type());
    rectified_right_image = cv::Mat(rows, cols, right_image.type());

    rectified_left_image  = 0;
    rectified_right_image = 0;

    for (int i = 0; i < len; ++i) {
        SpatialPoint lp = limgpts[i];
        lp.pos -= vec3(minx, miny, 0);
        ret.push_back(std::make_pair(lpts[i], lp));
        int lx = lp.pos.x;
        int ly = lp.pos.y;
        if (0 <= lx && lx < rectified_left_image.cols && //
            0 <= ly && ly < rectified_left_image.rows) {
            rectified_left_image.at<cv::Vec3b>(ly, lx) =
                cv::Vec3b(lp.color[0], lp.color[1], lp.color[2]);
        }

        SpatialPoint rp = rimgpts[i];
        rp.pos -= vec3(minx, miny, 0);
        int rx = rp.pos.x;
        int ry = rp.pos.y;
        if (0 <= rx && rx < rectified_right_image.cols && //
            0 <= ry && ry < rectified_right_image.rows) {
            rectified_right_image.at<cv::Vec3b>(ry, rx) =
                cv::Vec3b(rp.color[0], rp.color[1], rp.color[2]);
        }
    }
    interpolate(rectified_left_image);
    interpolate(rectified_right_image);

    return ret;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 16:19 [CST]

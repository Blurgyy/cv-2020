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

SpatialPoint to_image_space(CamConf const &conf, SpatialPoint const &point,
                            flt const &scale, flt const &hor_offset,
                            flt const &ver_offset) {
    SpatialPoint ret;
    ret.pos = {
        point.pos.x * conf.fx * scale / point.pos.z + conf.cx + hor_offset,
        point.pos.y * conf.fy * scale / point.pos.z + conf.cy + ver_offset,
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

SpatialPoint reproject(CamConf const &conf, SpatialPoint const &rpoint) {
    vec3 pos = rpoint.pos * conf.rot + conf.trans;
    // printf("new pos: %f %f %f\n", pos.x, pos.y, pos.z);
    pos.x /= pos.z;
    pos.y /= pos.z;
    return SpatialPoint{pos, rpoint.color};
}

void stereo_rectification(cv::Mat const &left_image,
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
// Date:   Feb 27 2021, 16:19 [CST]

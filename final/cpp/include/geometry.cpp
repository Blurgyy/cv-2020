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
    rectified_left_image =
        cv::Mat(left_image.rows, left_image.cols, left_image.type());
    rectified_right_image =
        cv::Mat(right_image.rows, right_image.cols, right_image.type());
    CamConf repconf = get_reprojection_conf(right_camera, left_camera);

    /* 1. Rotate right image to let it be parallel with left image */
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
    flt lmaxx = std::numeric_limits<flt>::lowest();
    flt rmaxx = std::numeric_limits<flt>::lowest();
    flt lmaxy = std::numeric_limits<flt>::lowest();
    flt rmaxy = std::numeric_limits<flt>::lowest();
    flt lminx = std::numeric_limits<flt>::max();
    flt rminx = std::numeric_limits<flt>::max();
    flt lminy = std::numeric_limits<flt>::max();
    flt rminy = std::numeric_limits<flt>::max();

    std::vector<vec3> lcamps, rcamps;
    for (int i = 0; i < len; ++i) {
        lcamps.push_back(to_camera_space(left_camera, lpts[i]).pos * R_rect);
        SpatialPoint lp =
            to_image_space(left_camera, {lcamps[i], lpts[i].color});
        lmaxx = std::max(lmaxx, lp.pos.x);
        lminx = std::min(lminx, lp.pos.x);
        lmaxy = std::max(lmaxy, lp.pos.y);
        lminy = std::min(lminy, lp.pos.y);

        rcamps.push_back(to_camera_space(right_camera, rpts[i]).pos * R_rect);
        SpatialPoint rp =
            to_image_space(right_camera, {rcamps[i], rpts[i].color});
        rmaxx = std::max(rmaxx, rp.pos.x);
        rminx = std::min(rminx, rp.pos.x);
        rmaxy = std::max(rmaxy, rp.pos.y);
        rminy = std::min(rminy, rp.pos.y);
    }

    flt lmidx = (lminx + lmaxx) / 2;
    flt rmidx = (rminx + rmaxx) / 2;
    flt lmidy = (lminy + lmaxy) / 2;
    flt rmidy = (rminy + rmaxy) / 2;

    flt left_scale =
        std::min(static_cast<flt>(left_image.cols) / (lmaxx - lminx),
                 static_cast<flt>(left_image.rows) / (lmaxy - lminy));
    flt right_scale =
        std::min(static_cast<flt>(right_image.cols) / (rmaxx - rminx),
                 static_cast<flt>(right_image.rows) / (rmaxy - rminy));
    flt left_hor_offset  = static_cast<flt>(left_image.cols) / 2 - lmidx;
    flt left_ver_offset  = static_cast<flt>(left_image.rows) / 2 - lmidy;
    flt right_hor_offset = static_cast<flt>(right_image.cols) / 2 - rmidx;
    flt right_ver_offset = static_cast<flt>(right_image.rows) / 2 - rmidy;

    flt scale      = std::min(left_scale, right_scale);
    flt hor_offset = (left_hor_offset + right_hor_offset) / 2;
    flt ver_offset = (left_ver_offset + right_ver_offset) / 2;

    vprintf("scale is %f\n", scale);
    vprintf("hor_offset is %f\n", hor_offset);
    vprintf("ver_offset is %f\n", ver_offset);

    // vprintf("left_scale is %f\n", left_scale);
    // vprintf("right_scale is %f\n", right_scale);
    // vprintf("left_hor_offset = %f\n", left_hor_offset);
    // vprintf("right_hor_offset = %f\n", right_hor_offset);
    // vprintf("left_ver_offset = %f\n", left_ver_offset);
    // eprintf("right_ver_offset = %f\n", right_ver_offset);

    for (int i = 0; i < len; ++i) {
        SpatialPoint lp =
            to_image_space(left_camera, {lcamps[i], lpts[i].color}, scale,
                           hor_offset, ver_offset);
        ret.push_back(std::make_pair(lpts[i], lp));
        int lx = lp.pos.x;
        int ly = lp.pos.y;
        if (0 <= lx && lx < rectified_left_image.cols && //
            0 <= ly && ly < rectified_left_image.rows) {
            rectified_left_image.at<cv::Vec3b>(ly, lx) =
                cv::Vec3b(lp.color[0], lp.color[1], lp.color[2]);
        }

        SpatialPoint rp =
            to_image_space(right_camera, {rcamps[i], rpts[i].color}, scale,
                           hor_offset, ver_offset);
        int rx = rp.pos.x;
        int ry = rp.pos.y;
        if (0 <= rx && rx < rectified_right_image.cols && //
            0 <= ry && ry < rectified_right_image.rows) {
            rectified_right_image.at<cv::Vec3b>(ry, rx) =
                cv::Vec3b(rp.color[0], rp.color[1], rp.color[2]);
        }
    }
    return ret;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 16:19 [CST]

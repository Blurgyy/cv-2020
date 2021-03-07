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
    // 1   │ cam0=[2945.377 0 1284.862; 0 2945.377 954.52; 0 0 1]
    // 2   │ cam1=[2945.377 0 1455.543; 0 2945.377 954.52; 0 0 1]
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
    CamConf repconf;
    repconf.rot   = R;
    repconf.trans = t;

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
    flt maxx = std::numeric_limits<flt>::lowest();
    flt maxy = std::numeric_limits<flt>::lowest();
    flt minx = std::numeric_limits<flt>::max();
    flt miny = std::numeric_limits<flt>::max();

    std::vector<SpatialPoint> limgpts, rimgpts;
    for (int i = 0; i < len; ++i) {
        vec3 lcampt = to_camera_space(left_camera, lpts[i]).pos * R_rect;
        SpatialPoint lp =
            to_image_space(left_camera, {lcampt, lpts[i].color});
        limgpts.push_back(lp);
        maxx = std::max(maxx, lp.pos.x);
        minx = std::min(minx, lp.pos.x);
        maxy = std::max(maxy, lp.pos.y);
        miny = std::min(miny, lp.pos.y);

        vec3 rcampt = (to_camera_space(right_camera, rpts[i]).pos * R_rect);
        SpatialPoint rp =
            to_image_space(right_camera, {rcampt, rpts[i].color});
        rimgpts.push_back(rp);
        maxx = std::max(maxx, rp.pos.x);
        minx = std::min(minx, rp.pos.x);
        maxy = std::max(maxy, rp.pos.y);
        miny = std::min(miny, rp.pos.y);
    }

    int cols              = std::round(maxx) - std::round(minx) + 1;
    int rows              = std::round(maxy) - std::round(miny) + 1;
    rectified_left_image  = cv::Mat(rows, cols, left_image.type());
    rectified_right_image = cv::Mat(rows, cols, right_image.type());

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
    return ret;
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 16:19 [CST]

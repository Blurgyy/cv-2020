#pragma once

#include "globla.hpp"

SpatialPoint to_camera_space(CamConf const &conf, SpatialPoint const &point);
SpatialPoint to_image_space(CamConf const &conf, SpatialPoint const &point,
                            flt const &scale      = 1.0,
                            flt const &hor_offset = 0.0,
                            flt const &ver_offset = 0.0);

/* Get the reprojection matrix from camera `from` to camera `to`.
 */
CamConf get_reprojection_conf(CamConf const &from, CamConf const &to);

/* Stereo rectification */
std::vector<ppp> stereo_rectification(cv::Mat const &left_image,
                                      cv::Mat const &right_image,
                                      CamConf const &left_camera,
                                      CamConf const &right_camera,
                                      cv::Mat &      rectified_left_image,
                                      cv::Mat &      rectified_right_image);

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Feb 27 2021, 16:19 [CST]

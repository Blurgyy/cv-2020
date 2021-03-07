#include "globla.hpp"

std::tuple<CamConf, CamConf> read_cam(std::string const &filename) {
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

MiscConf read_calib(std::string const &filename) {
    MiscConf      ret;
    std::ifstream from{filename};
    if (from.fail()) {
        eprintf("Failed opening file %s\n", filename.c_str());
    }
    flt dummy;
    for (std::string line; std::getline(from, line);) {
        for (int i = 0; i < line.length(); ++i) {
            if (line[i] == '=' || line[i] == '[' || line[i] == ']' ||
                line[i] == ';') {
                line[i] = ' ';
            }
        }
        std::istringstream in{line};
        std::string        token;
        in >> token;
        if (token == "cam0") {
            in >> ret.left.fx >> dummy >> ret.left.cx >> dummy >>
                ret.left.fy >> ret.left.cy;
        } else if (token == "cam1") {
            in >> ret.right.fx >> dummy >> ret.right.cx >> dummy >>
                ret.right.fy >> ret.right.cy;
        } else if (token == "doffs") {
            in >> ret.doffs;
        } else if (token == "baseline") {
            in >> ret.baseline;
        } else if (token == "width") {
            in >> ret.width;
        } else if (token == "height") {
            in >> ret.height;
        } else if (token == "ndisp") {
            in >> ret.ndisp;
        } else if (token == "ndisp") {
            in >> ret.ndisp;
        } else if (token == "isint") {
            in >> ret.isint;
        } else if (token == "vmin") {
            in >> ret.vmin;
        } else if (token == "vmax") {
            in >> ret.vmax;
        } else if (token == "dyavg") {
            in >> ret.dyavg;
        } else if (token == "dymax") {
            in >> ret.dymax;
        }
    }
    return ret;
}

cv::Mat map_back(std::vector<ppp> const &pixel_map, int const &rows,
                 int const &cols, cv::Mat const &dep) {
    int     drows = dep.rows;
    int     dcols = dep.cols;
    cv::Mat ret   = cv::Mat(rows, cols, CV_32FC1, -1);

    flt mind = std::numeric_limits<flt>::max();
    flt maxd = std::numeric_limits<flt>::lowest();
    for (ppp const &item : pixel_map) {
        SpatialPoint dep_p = item.second;
        int          dep_x = dep_p.pos.x;
        int          dep_y = dep_p.pos.y;
        if (0 <= dep_x && dep_x < dcols && //
            0 <= dep_y && dep_y < drows) {
            SpatialPoint ori_p          = item.first;
            int          ori_x          = ori_p.pos.x;
            int          ori_y          = ori_p.pos.y;
            flt          depth          = dep.at<float>(dep_y, dep_x);
            ret.at<float>(ori_y, ori_x) = depth;

            mind = std::min(mind, depth);
            maxd = std::max(maxd, depth);
        }
    }

    /* [Normalize] */
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float &value = ret.at<float>(y, x);
            if (value < 0) {
                continue;
            }
            value = (value - mind) / (maxd - mind);
            value = std::pow(value, 0.3);
            value = value * 256 - 0.5;
        }
    }
    /* [/Normalize] */

    return ret;
}

void get_matches(cv::Mat const &limg, cv::Mat const &rimg,
                 std::vector<cv::KeyPoint> &kp1,
                 std::vector<cv::KeyPoint> &kp2,
                 std::vector<cv::DMatch> &  matches) {
    kp1.clear();
    kp2.clear();
    cv::Mat          desc1, desc2;
    cv::Ptr<cv::ORB> orb =
        cv::ORB::create(500, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    /* 1. Detect key points */
    orb->detect(limg, kp1);
    orb->detect(rimg, kp2);

    /* 2. Compute descriptors */
    orb->compute(limg, kp1, desc1);
    orb->compute(rimg, kp2, desc2);

    /* 3. Match descriptors with hamming distance */
    std::vector<cv::DMatch> all_matches;
    cv::BFMatcher           matcher(cv::NORM_HAMMING);
    matcher.match(desc1, desc2, all_matches);

    /* 4. Filter keypoints */
    flt maxd = std::numeric_limits<flt>::lowest();
    flt mind = std::numeric_limits<flt>::max();
    for (int y = 0; y < desc1.rows; ++y) {
        mind = std::min<flt>(mind, all_matches[y].distance);
        maxd = std::max<flt>(maxd, all_matches[y].distance);
    }

    for (int y = 0; y < desc1.rows; ++y) {
        if (all_matches[y].distance <= std::max<flt>(2 * mind, 30)) {
            matches.push_back(all_matches[y]);
        }
    }

    // cv::Mat img;
    // cv::drawMatches(limg, kp1, rimg, kp2, matches, img);
    // cv::imwrite("matched.png", img);
}

// Author: Blurgy <gy@blurgy.xyz>
// Date:   Mar 07 2021, 16:15 [CST]

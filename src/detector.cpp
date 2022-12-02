//
// Created by hitchicris on 11/06/22.
//

#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>

#include "detector.hpp"
#include "utils.hpp"
#include "random.hpp"

std::vector<cv::Mat> cgra::read_imgs() {
    std::vector<cv::String> file_names;
    cv::glob("res/Frame*.jpg", file_names);

    std::vector<cv::Mat> out;
    for (const auto& f : file_names) {
        out.emplace_back(cgra::readImg(f, 1));
    }

    return out;
}

void cgra::detector::task_1() {
    if (!feature_pairs.has_value()) {
        feature_pairs = get_feature_pairs(39, 41);
    }

    // draw match
    cv::Mat task1_out {frames[0].rows * 2, frames[0].cols, CV_8UC3, cv::Scalar(0, 0,0)};
    frames[39].copyTo(cv::Mat {task1_out, cv::Rect {0, 0, frames[0].cols, frames[0].rows}});
    frames[41].copyTo(cv::Mat {task1_out, cv::Rect {0, frames[0].rows, frames[0].cols, frames[0].rows}});

    for (const auto& p : *feature_pairs) {
        auto top = p.first;
        auto bottom = p.second + cv::Point2f {0, static_cast<float>(frames[0].rows)};
        cv::line(task1_out, top, bottom, cv::Scalar(0, 255, 0));
    }

    cgra::ezshow("task 1", task1_out);
//    cv::imwrite("out/task 1.png", task1_out);
    cv::waitKey(0);
}

void cgra::detector::task_2() {
    if (!feature_pairs.has_value()) {
        feature_pairs = get_feature_pairs(39, 41);
    }
    if (!ransac.has_value()) {
        ransac = get_ransac_homography(*feature_pairs);
    }

    // draw match
    cv::Mat task2_out {frames[0].rows * 2, frames[0].cols, CV_8UC3, cv::Scalar(0, 0,0)};
    frames[39].copyTo(cv::Mat {task2_out, cv::Rect {0, 0, frames[0].cols, frames[0].rows}});
    frames[41].copyTo(cv::Mat {task2_out, cv::Rect {0, frames[0].rows, frames[0].cols, frames[0].rows}});

    for (const auto& p : *feature_pairs) {
        auto top = p.first;
        auto bottom = p.second + cv::Point2f {0, static_cast<float>(frames[0].rows)};
        if (std::find(ransac->inliers.begin(), ransac->inliers.end(), p) != ransac->inliers.end()) {
            cv::line(task2_out, top, bottom, cv::Scalar(0, 255, 0));
        } else {
            cv::line(task2_out, top, bottom, cv::Scalar(0, 0, 255));
        }
    }

    cgra::ezshow("task 2", task2_out);
//    cv::imwrite("out/task 2.png", task2_out);
    cv::waitKey(0);
}

void cgra::detector::task_3() {
    if (!feature_pairs.has_value()) {
        feature_pairs = get_feature_pairs(39, 41);
    }
    if (!ransac.has_value()) {
        ransac = get_ransac_homography(*feature_pairs);
    }

    cv::Mat warped;
    cv::warpPerspective(frames[41], warped, ransac->homography, cv::Size {frames[0].cols + 20, frames[0].rows + 20});

    cv::Mat task3_out;
    cv::copyMakeBorder(frames[39], task3_out, 20, 20, 20, 20, cv::BORDER_CONSTANT, cv::Scalar(0, 255, 0));
    warped.copyTo(cv::Mat {task3_out, cv::Rect {20, 20, frames[0].cols + 20, frames[0].rows + 20 }}, warped);

    cgra::ezshow("task 3", task3_out);
//    cv::imwrite("out/task 3.png", task3_out);
    cv::waitKey(0);
}

void cgra::detector::task_4() {
    const int start = 0;
    auto o_frames = stabilise_frames(start, 100);
    std::cout << o_frames.size() << std::endl;

    int i = start;
    for (const auto& f : o_frames) {
        std::stringstream n;
        n << "out/Stable" << std::setw(3) << std::setfill('0') << i << ".png";
        cv::imwrite(n.str(), f);
        i++;
    }
}

const float gaussian[] = {0.1, 0.3, 0.5, 0.3, 0.1};

std::vector<cv::Mat> cgra::detector::stabilise_frames(int start, int end) {
    assert(start < end);
    assert(end <= frames.size());

    // calc first homography
    cv::Mat first_homograpy = find_homography(start + 1, start + 0);
    auto running_homography = first_homograpy.clone();
    std::vector<cv::Mat> homographies { first_homograpy };
    for (int i = start + 0; i <= end; i++) {
        // running H = running H * H[i-1 -> i]
        running_homography *= find_homography(i + 1, i + 0);
        // store running H in vec
        homographies.emplace_back(running_homography.clone());
    }
    std::vector<cv::Mat> out_frames { frames[start] };
    for (int i = start + 1, idx = 1; i <= end; i++, idx++) {
        // do gaussian smoothing
        cv::Mat smooth_transform = {};

        // gaussian sums to 1, no division needed
        smooth_transform = (gaussian[0] * cgra::clamp_at(homographies, idx - 2))
                         + (gaussian[1] * cgra::clamp_at(homographies, idx - 1))
                         + (gaussian[2] * cgra::clamp_at(homographies, idx + 0))
                         + (gaussian[3] * cgra::clamp_at(homographies, idx + 1))
                         + (gaussian[4] * cgra::clamp_at(homographies, idx + 2));

        // then apply the U = H'^-1 * H calculation
        cv::Mat u = smooth_transform.inv() * homographies[idx];

        // and apply to frame, append to vec
        cv::Mat warped;
        cv::warpPerspective(frames[i], warped, u, cv::Size {frames[0].cols, frames[0].rows}, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 255, 0));
        out_frames.emplace_back(warped.clone());
    }

    return out_frames;
}

cv::Mat cgra::detector::find_homography(int original, int target) {
    auto features = cgra::vecpair2pairvec(get_feature_pairs(original, target));
    return cv::findHomography(features.first, features.second, cv::RANSAC);
}

std::vector<pair_vec2f> cgra::detector::get_feature_pairs(int index_l, int index_r) {
    std::vector<cv::KeyPoint> pts_l;
    sifter->detect(frames[index_l], pts_l);
    cv::Mat descriptor_l;
    sifter->compute(frames[index_l], pts_l, descriptor_l);

    std::vector<cv::KeyPoint> pts_r;
    sifter->detect(frames[index_r], pts_r);
    cv::Mat descriptor_r;
    sifter->compute(frames[index_r], pts_r, descriptor_r);

    auto matcher = cv::BFMatcher::create(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptor_l, descriptor_r, matches);

    //left is pt on f39, right is pt on f41
    std::vector<pair_vec2f> match_locs;
    for (auto m : matches) {
        auto l = pts_l[m.queryIdx].pt;
        auto r = pts_r[m.trainIdx].pt;

        pair_vec2f p { l, r };

        match_locs.emplace_back(p);
    }
    return match_locs;
}

cgra::ransac_data cgra::detector::get_ransac_homography(std::vector<pair_vec2f> features) const {
    // RANSAC
    ransac_data best;
    for (int i = 0; i < ransac_iters; i++) {
        std::vector<pair_vec2f> ransac_pts;
        while (ransac_pts.size() < 4) {
            pair_vec2f proposed_pt = features.at(cgra::random::integer(0, features.size() - 1));
            if (std::find(ransac_pts.begin(), ransac_pts.end(), proposed_pt) == ransac_pts.end()) {
                ransac_pts.emplace_back(proposed_pt);
            }
        }
        assert(ransac_pts.size() == 4);

        // make homography
        auto nu_ransac_pts = cgra::vecpair2pairvec(ransac_pts);
        cv::Mat homography = cv::findHomography(nu_ransac_pts.second, nu_ransac_pts.first, 0);

        // find inliers by |(left - H * right)| < eps
        int count_matches = 0;
        std::vector<pair_vec2f> inliers = {};

        auto match_locs_decons = cgra::vecpair2pairvec(features);
        std::vector<cv::Point2f> transformed_right_pts;
        cv::perspectiveTransform(match_locs_decons.second, transformed_right_pts, homography);

        for (unsigned long j = 0; j < features.size(); j++) {
            if (cv::norm(match_locs_decons.first[j] - transformed_right_pts[j]) < ransac_epsilon) {
                count_matches++;
                inliers.emplace_back(features[j]);
            }
        }

        // if there's more inliers than last then woo
        if (count_matches > best.num_inliers) {
            best.num_inliers = count_matches;
            best.inliers = inliers;
            best.homography = homography;
        }
    }

    auto best_pairs_decons = cgra::vecpair2pairvec(best.inliers);
    cv::Mat final_homography = cv::findHomography(best_pairs_decons.second, best_pairs_decons.first, 0);
    best.homography = final_homography;

    return best;
}









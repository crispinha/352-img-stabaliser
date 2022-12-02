//
// Created by hitchicris on 11/06/22.
//

#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>

#include <optional>

using pair_vec2f = std::pair<cv::Point2f, cv::Point2f>;


namespace cgra {
    std::vector<cv::Mat> read_imgs();


    struct ransac_data {
        int num_inliers;
        std::vector<pair_vec2f> inliers;
        cv::Mat homography;

        void reset() {
            num_inliers = 0;
            inliers.clear();
            homography = cv::Mat {};
        }
    };

    class detector {
    public:
        const int ransac_iters = 1000;
        const float ransac_epsilon = 10.f;

        std::vector<cv::Mat> frames;
        detector() {
            frames = read_imgs();
        }

        void task_1();
        void task_2();
        void task_3();
        void task_4();

     private:
        cv::Ptr<cv::SIFT> sifter = cv::SIFT::create();

        std::vector<pair_vec2f> get_feature_pairs(int index_l, int index_r);
        [[nodiscard]] ransac_data get_ransac_homography(std::vector<pair_vec2f> features) const;

        std::vector<cv::Mat> stabilise_frames(int start, int end);

        cv::Mat find_homography(int original, int target);

        std::optional<std::vector<pair_vec2f>> feature_pairs;
        std::optional<ransac_data> ransac;

    };



}
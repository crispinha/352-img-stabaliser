//
// Created by crispin on 24/04/22.
//

#pragma once

#include <opencv2/core.hpp>

namespace cgra {
    cv::Mat readImg(const std::string& path, int flags = 1);
    void ezshow(const std::string& name, const cv::Mat& img);

    template<typename T>
    std::pair<std::vector<T>, std::vector<T>> vecpair2pairvec(std::vector<std::pair<T, T>> vec) {
        std::vector<T> left, right;
        for (auto p : vec) {
            T l = p.first;
            T r = p.second;
            left.template emplace_back(l);
            right.template emplace_back(r);
        }

        return std::pair {left, right};
    }

    cv::Mat clamp_at(std::vector<cv::Mat> v, unsigned long i);

    template<typename T>
    T clamp_at(std::vector<T> v, unsigned long i) {
        if (i < 0) {
            return v[0];
        }
        if (i >= v.size()) {
            return v[v.size()-1];
        }
        return v[i];
    }
}


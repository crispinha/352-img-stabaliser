//
// Created by crispin on 24/04/22.
//

#include "utils.hpp"

#include "utils.hpp"

#include <opencv2/highgui.hpp>
#include <iostream>

void cgra::ezshow(const std::string& name, const cv::Mat& img) {
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img);
}

cv::Mat cgra::readImg(const std::string& path, int flags) {
    // read the file
    cv::Mat image;
    image = cv::imread(path, flags);

    // check for invalid input
    if(!image.data) {
        std::cerr << "Could not open or find the image" << std::endl;
        abort();
    }

    return image;
}

cv::Mat cgra::clamp_at(std::vector<cv::Mat> v, unsigned long i) {
    if (i < 0) {
        return cv::Mat {v[0]}.setTo(0);
    }
    if (i >= v.size()) {
        return cv::Mat {v[0]}.setTo(0);
    }
    return v[i];
}


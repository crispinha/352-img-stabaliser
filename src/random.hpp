//
// Created by crispin on 21/04/22.
//

#pragma once

#include <random>

namespace cgra {

    class random {
        static std::random_device rd;
        static std::mt19937* gen;

     public:
        static int integer(int lo, int hi);
    };

}
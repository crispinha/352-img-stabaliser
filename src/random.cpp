//
// Created by crispin on 21/04/22.
//

#include "random.hpp"

namespace cgra {
//    std::mt19937 random::gen {  };
    std::random_device random::rd {};
    std::mt19937* random::gen = nullptr;

    int random::integer(int lo, int hi) {
        if (gen == nullptr) {
            gen = new std::mt19937 {rd()};
        }
        std::uniform_int_distribution<int> d {lo, hi};
        return d(*gen);
    }
} // cgra
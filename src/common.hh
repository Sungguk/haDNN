//File: common.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once
#include <Halide.h>
#include <iostream>
#include <unordered_map>
#include "layers/shape.hh"
#include "lib/utils.hh"

namespace hadnn {

Halide::Image<float> random_image(const Shape& shape, std::string name="");

void write_tensor(const Halide::Image<float>& v, std::ostream& os);

std::unordered_map<std::string, Halide::Image<float>> read_params(std::string fname);

}

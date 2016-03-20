//File: common.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once
#include <Halide.h>
#include <iostream>
#include "layers/shape.hh"
#include "lib/utils.hh"

namespace hadnn {

Halide::Image<float> random_image(const Shape& shape, std::string name="");

void write_tensor(const Halide::Image<float>& v, std::ostream& os);

}
